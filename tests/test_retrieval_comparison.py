import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import late_interaction_baselines as legacy
from experiments import retrieval_comparison as comparison
from ras.late_interaction import CrossEncoderScorer


def arguments(root, *extra):
    return comparison.parse_args(['--synthetic', '--output-dir', str(root), '--queries', '2',
        '--k', '5', '--pool-size', '50', '--candidates', '10', '1000', '--repetitions', '2',
        '--partition-bits', '2', '--fde-dim', '32', '--timing-repeats', '2', *extra])


def test_cross_encoder_preserves_pairs_and_handles_empty_candidates():
    class FakeModel:
        def predict(self, pairs, **kwargs):
            assert pairs == [('query', 'second'), ('query', 'first')]
            assert kwargs['activation_fn'] == 'identity'
            return np.array([-3., 5.])
    model = CrossEncoderScorer.__new__(CrossEncoderScorer)
    model.model, model.batch_size, model.activation = FakeModel(), 2, 'identity'
    assert model.score('query', []).shape == (0,)
    np.testing.assert_array_equal(model.score('query', ['second', 'first']), [-3., 5.])
    model.model.predict = lambda *a, **k: np.array([[1., 2.], [3., 4.]])
    with pytest.raises(ValueError, match='one finite'):
        model.score('query', ['second', 'first'])


@pytest.mark.parametrize('backend', ['flat', 'hnsw'])
def test_separate_experiments_costs_and_full_budget_parity(tmp_path, backend):
    if backend == 'hnsw':
        pytest.importorskip('faiss')
    args = arguments(tmp_path, '--backend', backend)
    comparison.run(args)
    quality = pd.read_csv(tmp_path / 'ranking_quality.csv')
    assert set(quality.method) == {'dense_dot_product', 'colbert_exact', 'cross_encoder',
                                  'binary_predicates', 'linear_predicates'}
    assert set(quality.scope) == {'shared_minilm_pool'}
    assert 'colbert_topk_recall_mean' not in quality
    approximation = pd.read_csv(tmp_path / 'approximation.csv')
    assert 'ndcg_mean' not in approximation and 'cross_encoder' not in set(approximation.method)
    rows = json.loads((tmp_path / 'rankings.json').read_text())
    for qid in {row['query_id'] for row in rows}:
        shared = [row for row in rows if row['query_id'] == qid and row['scope'] == 'shared_minilm_pool']
        assert all(row['scored_rows'] == shared[0]['scored_rows'] for row in shared)
        full = [row for row in rows if row['query_id'] == qid and row['scope'] == 'full_corpus']
        oracle = next(row['selected_rows'] for row in full if row['method'] == 'colbert_exact')
        assert all(row['selected_rows'] == oracle for row in full if row['candidate_budget'] == 1000)
    per_query = pd.read_csv(tmp_path / 'per_query.csv')
    assert set(per_query.repeat) == {0, 1}  # Warmup never leaks into sample counts.
    assert (per_query.runtime_pool_overlap.dropna() == 1).all()
    stage_total = per_query[['query_encoding_ms', 'pool_search_ms', 'fde_ms', 'search_ms', 'scoring_ms']].sum(axis=1)
    assert (per_query.total_ms >= stage_total).all()
    latency = pd.read_csv(tmp_path / 'latency.csv')
    assert (latency.samples == 4).all()
    assert (latency.concurrency == 1).all()
    storage = pd.read_csv(tmp_path / 'storage.csv').set_index('pipeline')
    assert storage.loc['dense_pool_cross_encoder', 'corpus_disk_bytes'] == storage.loc['dense_dot_product', 'corpus_disk_bytes']
    components = pd.read_csv(tmp_path / 'storage_components.csv').set_index('artifact').disk_bytes
    if backend == 'hnsw':
        expected = components[['metadata.json', 'documents.npz', 'fde_hnsw.faiss', 'fde_parameters.npz']].sum()
        assert storage.loc['muvera_hnsw_maxsim', 'corpus_disk_bytes'] == expected
    matched = pd.read_csv(tmp_path / 'matched_fidelity.csv')
    assert (matched.status == 'reached_on_this_run').all()
    assert (matched.observed_recall >= matched.target_colbert_topk_recall).all()
    import zipfile
    with zipfile.ZipFile(tmp_path / 'late_interaction_results.zip') as archive:
        assert archive.testzip() is None
        assert 'build_time.csv' in archive.namelist()


def test_v1_reuse_validates_manifest_and_preparation_config(tmp_path):
    source = tmp_path / 'v1'
    source.mkdir()
    old = legacy.parse_args(['--synthetic', '--output-dir', str(source), '--queries', '2'])
    legacy.run(old)
    target = tmp_path / 'v2'
    target.mkdir()
    args = arguments(target, '--prepared-from', str(source))
    comparison.prepare_inputs(args, target)
    assert (target / 'documents.npz').read_bytes() == (source / 'documents.npz').read_bytes()
    assert not (target / 'summary.csv').exists()
    incompatible = tmp_path / 'incompatible'
    incompatible.mkdir()
    args.seed = 9
    with pytest.raises(ValueError, match='different seed'):
        comparison.prepare_inputs(args, incompatible)
    with (source / 'documents.npz').open('ab') as file:
        file.write(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        comparison.verify_prepared(source)


def test_empty_filter_stays_in_quality_and_does_not_call_pair_model(tmp_path):
    args = arguments(tmp_path)
    legacy.prepare(args, tmp_path)
    path = tmp_path / 'queries.json'
    specs = json.loads(path.read_text())
    specs[0]['exact'] = [['baseColour', 'NoSuchColour']]
    path.write_text(json.dumps(specs))
    comparison.evaluate(tmp_path, args)
    rows = pd.read_csv(tmp_path / 'per_query.csv')
    empty = rows[rows.query_id == specs[0]['query_id']]
    assert (empty.returned == 0).all()
    assert (empty.precision_at_k == 0).all()
    assert empty.recall.isna().all()
    assert empty.colbert_topk_recall.isna().all()


def test_paired_uncertainty_uses_queries_not_timing_repeats(tmp_path):
    frame = pd.DataFrame([dict(query_id=q, method=m, scope='shared_minilm_pool',
                              ndcg=v, recall=v, precision_at_k=v)
                          for q in ['a', 'b'] for m, v in [('one', .7), ('two', .2)] for _ in range(4)])
    comparison.write_paired_deltas(frame, tmp_path)
    deltas = pd.read_csv(tmp_path / 'paired_quality_deltas.csv')
    assert (deltas.paired_queries == 2).all()
    np.testing.assert_allclose(deltas['mean'], .5)
    np.testing.assert_allclose(deltas.lo, .5)
    np.testing.assert_allclose(deltas.hi, .5)


def test_online_encoding_drift_does_not_change_approximation_reference(tmp_path, monkeypatch):
    args = arguments(tmp_path, '--warmup', '0', '--timing-repeats', '1')
    legacy.prepare(args, tmp_path)
    monkeypatch.setattr(comparison.Models, 'token_query', lambda self, text, qi: -self.queries[qi])
    comparison.evaluate(tmp_path, args)
    full = pd.read_csv(tmp_path / 'per_query.csv')
    exact = full[(full.scope == 'full_corpus') & (full.method == 'colbert_exact')]
    np.testing.assert_array_equal(exact.colbert_topk_recall, 1.)
    exhaustive = full[(full.scope == 'full_corpus') & (full.candidate_budget == 1000)]
    np.testing.assert_array_equal(exhaustive.colbert_topk_recall, 1.)


def test_modern_profile_selects_matched_models_and_rejects_overrides():
    args = arguments('/tmp/modern', '--model-family', 'modernbert-base')
    assert args.checkpoint == comparison.MODERN_MODELS['colbert']
    assert args.cross_encoder_checkpoint == comparison.MODERN_MODELS['cross_encoder']
    assert comparison.load_config(args.config)['retrieval']['model'] == comparison.MODERN_MODELS['dense']
    assert args.query_maxlen == 48 and args.doc_maxlen == 300 and args.cross_encoder_maxlen == 512
    assert comparison.pool_scope(args) == 'shared_modernbert_pool'
    for flag, value in [('--checkpoint', 'colbert-ir/colbertv2.0'),
                        ('--cross-encoder-checkpoint', 'cross-encoder/ms-marco-MiniLM-L6-v2'),
                        ('--config', 'configs/binary_bbq.yaml')]:
        with pytest.raises(SystemExit):
            arguments('/tmp/modern', '--model-family', 'modernbert-base', flag, value)


def test_modern_profile_runs_and_cannot_reuse_legacy_embeddings(tmp_path):
    legacy_dir = tmp_path / 'legacy'
    comparison.run(arguments(legacy_dir, '--warmup', '0', '--timing-repeats', '1'))
    modern_dir = tmp_path / 'modern'
    modern_dir.mkdir()
    incompatible = arguments(modern_dir, '--model-family', 'modernbert-base', '--prepared-from', str(legacy_dir))
    with pytest.raises(ValueError, match='different model family'):
        comparison.prepare_inputs(incompatible, modern_dir)
    args = arguments(modern_dir, '--model-family', 'modernbert-base', '--warmup', '0', '--timing-repeats', '1')
    comparison.run(args)
    quality = pd.read_csv(modern_dir / 'ranking_quality.csv')
    assert set(quality.scope) == {'shared_modernbert_pool'}
    assert set(quality.model_family) == {'modernbert-base'}
    rows = json.loads((modern_dir / 'rankings.json').read_text())
    for qi in {r['query_id'] for r in rows}:
        shared = [r for r in rows if r['query_id'] == qi and r['scope'] == 'shared_modernbert_pool']
        assert len(shared) == 5
        assert all(r['scored_rows'] == shared[0]['scored_rows'] for r in shared)
    again = tmp_path / 'modern_reuse'
    again.mkdir()
    comparison.prepare_inputs(arguments(again, '--model-family', 'modernbert-base', '--prepared-from', str(modern_dir)), again)
    assert (again / 'documents.npz').read_bytes() == (modern_dir / 'documents.npz').read_bytes()


def test_backbone_guard_checks_architecture_precision_and_size():
    rows = [dict(method=m, model_type='modernbert', parameters=p, hidden_size=768,
                 layers=22, attention_heads=12, weight_dtype='torch.float32')
            for m, p in [('dot', 149000000), ('colbert', 149100000), ('ce', 149600000)]]
    comparison.validate_modern_backbones(rows)
    for field, value in [('model_type', 'bert'), ('layers', 12), ('hidden_size', 384),
                         ('weight_dtype', 'torch.float16'), ('parameters', 23000000)]:
        altered = [dict(r) for r in rows]
        altered[-1][field] = value
        with pytest.raises(ValueError):
            comparison.validate_modern_backbones(altered)


def test_modern_encoder_keeps_query_mode_ragged_tokens_and_input_order():
    from ras.late_interaction import ModernColBERTEncoder
    class PylateModel:
        def encode(self, texts, **kwargs):
            assert kwargs['padding'] is False
            assert kwargs['normalize_embeddings'] is True
            assert kwargs['convert_to_numpy'] is True
            length = 4 if kwargs['is_query'] else None
            return [np.full((length or int(t) + 1, 128), int(t), dtype=np.float32) for t in texts]
    encoder = ModernColBERTEncoder.__new__(ModernColBERTEncoder)
    encoder.batch_size, encoder.model = 2, PylateModel()
    docs = encoder.encode(['2', '0', '3'], query=False)
    assert docs.offsets.tolist() == [0, 3, 4, 8]
    assert np.all(docs[2] == 3)
    queries = encoder.encode(['2', '0', '3'], query=True)
    assert queries.offsets.tolist() == [0, 4, 8, 12]
