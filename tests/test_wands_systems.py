import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments import wands_comparison as w
from experiments import wands_systems as s
from ras.late_interaction import MuveraFDE, RaggedEmbeddings, maxsim


def test_disk_token_packing_preserves_negative_maxsim_and_order(tmp_path):
    matrices = [np.array([[-1., 0.]], dtype='float32'),
                np.array([[-.3, -.7], [-.7, -.3]], dtype='float32')]
    paths = []
    for i, matrix in enumerate(matrices):
        path = tmp_path / f'{i}.npz'
        np.savez(path, values=matrix, offsets=[0, len(matrix)])
        paths.append(path)
    s.combine_tokens(paths, tmp_path)
    tokens = s.Tokens(tmp_path)
    assert isinstance(tokens.values, np.memmap)
    for i in range(2):
        np.testing.assert_array_equal(tokens[i], matrices[i])
    np.testing.assert_allclose(maxsim(np.array([[1., 0.]], dtype='float32'), tokens), [-1., -.3])


def test_fde_full_budget_reranking_recovers_exact_and_index_metric(tmp_path):
    faiss = pytest.importorskip('faiss')
    rng = np.random.default_rng(7)
    docs = RaggedEmbeddings.from_list([rng.normal(size=(i % 4+1, 8)).astype('float32') for i in range(40)])
    query = rng.normal(size=(3, 8)).astype('float32')
    fde = MuveraFDE(8, repetitions=2, partition_bits=2, final_dim=32, seed=7)
    fde.save(tmp_path / 'map.npz')
    restored = MuveraFDE.load(tmp_path / 'map.npz')
    values, q = fde.encode_many(docs, query=False), restored.encode(query, query=True)
    np.testing.assert_array_equal(q, fde.encode(query, query=True))
    ids = [str(i) for i in range(len(docs))]
    scores = maxsim(query, docs)
    exact = w.ranked_ids(scores, ids, 10)
    for index in [faiss.IndexFlatIP(32), faiss.IndexHNSWFlat(32, 16, faiss.METRIC_INNER_PRODUCT)]:
        index.add(values)
        retrieved = s.retrieve(index, q, 100, 100)
        assert len(retrieved) == len(docs)
        actual = w.ranked_ids(maxsim(query, docs, retrieved), [ids[i] for i in retrieved], 10)
        assert actual == exact
    flat = faiss.IndexFlatIP(32)
    flat.add(values)
    np.testing.assert_array_equal(s.retrieve(flat, q, 7, 100), np.argsort(-(values @ q))[:7])


def toy_reference(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('faiss')
    pytest.importorskip('ir_measures')
    reference, root = tmp_path / 'reference', tmp_path / 'systems'
    reference.mkdir()
    corpus = {str(i): f'product {i}' for i in range(20)}
    queries = {'a': 'product 1', 'b': 'product 6'}
    data = dict(corpus=corpus, queries=queries, qrels={'a': {'1': 2, '2': 1}, 'b': {'6': 2, '5': 1}})
    def vector(text):
        n = int(text.split()[-1])
        x = np.array([np.cos(n), np.sin(n), .1], dtype='float32')
        return x / np.linalg.norm(x)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def encode_query(self, texts, **kw):
            return np.stack([vector(t) for t in texts])
        encode_document = encode_query
        def encode(self, texts, **kw):
            return [vector(t)[None, :] for t in texts]
        def predict(self, pairs, **kw):
            return np.array([vector(q) @ vector(d) for q, d in pairs])
    monkeypatch.setattr(w, 'load_model', lambda arm, *args: (Toy(), dict(repo=w.MODELS[arm][0], revision=w.MODELS[arm][1], max_seq_length=512)))
    args = s.parse_args(['--reference-dir', str(reference), '--output-dir', str(root), '--device', 'cpu',
        '--dimensions', '16', '--dimension', '16', '--candidates', '10', '20', '--shard-size', '7',
        '--timing-queries', '2', '--timing-repeats', '1', '--warmup', '1', '--parity-queries', '2',
        '--repetitions', '2', '--partition-bits', '2'])
    values = np.stack([vector(t) for t in corpus.values()])
    scores = np.stack([vector(t) for t in queries.values()]) @ values.T
    for arm in ['dense', 'colbert']:
        (reference / arm).mkdir()
        w.save(reference / arm / 'model.json', w.load_model(arm)[1])
        w.array_save(reference / arm / 'scores_000000.npy', scores)
    w.save(reference / 'dataset.json', data)
    w.save(reference / 'manifest.json', dict(models=w.MODELS, dataset_sha256=w.digest(data), max_length=512,
        pool_size=5, versions={p: s.importlib.metadata.version(p) for p in ['pylate', 'sentence-transformers', 'transformers']}))
    w.save(reference / 'complete.json', dict(status='complete'))
    pools = w.make_pools(data, scores, w.lexical_scores(data), 5)
    w.save(reference / 'candidate_ids.json', pools)
    (reference / 'quality.csv').write_text('method,ndcg10\ndense,1\n')
    return args, data


def test_pipeline_fidelity_cost_reporting_and_resume_without_real_models(tmp_path, monkeypatch):
    args, data = toy_reference(tmp_path, monkeypatch)
    s.prepare(args)
    for arm in ['dense', 'colbert']:
        args.arm = arm
        s.encode(args)
        # Completion marker must avoid encoding again.
        before = w.sha_file(Path(args.output_dir) / arm / 'values.npy')
        s.encode(args)
        assert w.sha_file(Path(args.output_dir) / arm / 'values.npy') == before
    s.parity(args)
    s.transform(args)
    for arm in ['flat', 'hnsw']:
        args.arm = arm
        s.build_index(args)
        s.fidelity(args)
    # Timed scoring must not read reference score shards.
    monkeypatch.setattr(s, 'reference_scores', lambda *a: pytest.fail('Reference scores in timed path'))
    for arm in ['dense', 'colbert', 'bm25', 'ce', 'flat', 'hnsw']:
        args.arm = arm
        s.latency(args)
    s.report(args)
    root = Path(args.output_dir)
    fidelity = pd.read_csv(root / 'fidelity.csv')
    assert (fidelity[fidelity.candidate_budget == 20].colbert_topk_recall == 1).all()
    times = pd.read_csv(root / 'latency.csv')
    assert set(times.method) == {'dense', 'colbert', 'bm25', 'ce', 'flat', 'hnsw'}
    assert (times.total_ms_p50 > 0).all()
    assert (pd.read_csv(root / 'memory.csv').rss_sampled_peak_B > 0).all()
    storage = pd.read_csv(root / 'storage_build.csv')
    assert storage[storage.method == 'muvera_flat'].serving_corpus_disk_B.iloc[0] > storage[storage.method == 'colbert'].serving_corpus_disk_B.iloc[0]
    assert (root / 'latency_at_fidelity.csv').exists()


def test_reference_guard_rejects_modified_dataset(tmp_path, monkeypatch):
    args, data = toy_reference(tmp_path, monkeypatch)
    data['queries']['a'] = 'changed'
    w.save(Path(args.reference_dir) / 'dataset.json', data)
    with pytest.raises(ValueError, match='differs from its manifest'):
        s.prepare(args)


@pytest.mark.skipif(os.environ.get('WANDS_REAL_MODELS') != '1', reason='Opt-in real checkpoint integration')
def test_actual_checkpoints_in_separate_serving_processes(tmp_path, monkeypatch):
    """Tiny artificial corpus tests compatibility only; not benchmark evidence."""
    data = dict(corpus={'0': 'red wooden chair', '1': 'blue armchair', '2': 'oak dining table', '3': 'white desk'},
                queries={'a': 'red chair', 'b': 'dining table'}, qrels={'a': {'0': 2}, 'b': {'2': 2}})
    reference, root = tmp_path / 'reference', tmp_path / 'systems'
    monkeypatch.setattr(w, 'load_data', lambda *a: (data, dict(synthetic=True)))
    w.run(w.parse_args(['--output-dir', str(reference), '--device', 'cpu', '--batch-size', '2', '--pool-size', '2']))
    args = s.parse_args(['--reference-dir', str(reference), '--output-dir', str(root), '--device', 'cpu',
        '--dimensions', '16', '--candidates', '2', '4', '--k', '2', '--batch-size', '2', '--shard-size', '2',
        '--timing-queries', '1', '--timing-repeats', '1', '--warmup', '1', '--parity-queries', '2',
        '--repetitions', '2', '--partition-bits', '2'])
    s.run(args)
    assert s.read(root / 'complete.json')['status'] == 'complete'
    fidelity = pd.read_csv(root / 'fidelity.csv')
    assert (fidelity[fidelity.candidate_budget == 4].colbert_topk_recall == 1).all()
