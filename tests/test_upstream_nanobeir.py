"""Native evaluator integration uses deterministic embeddings and no downloads."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from experiments.upstream_nanobeir import CARD, digest, paired, restrict, save, prepare_data, audit_data, check_manifest


def missing_document_parts():
    return dict(corpus=[{'_id': 'a', 'text': 'red'}, {'_id': 'empty', 'text': ''}],
                queries=[{'_id': 'q', 'text': 'red'}, {'_id': 'only_missing', 'text': 'blue'}],
                qrels=[{'query-id': 'q', 'corpus-id': d} for d in ['a', 'empty', 'absent']] +
                      [{'query-id': 'only_missing', 'corpus-id': 'absent'}])


def test_missing_documents_remain_in_qrels():
    data = prepare_data(missing_document_parts(), 'fixture', 'revision')
    assert data['corpus'] == {'a': 'red'}
    assert data['qrels'] == {'q': ['a', 'absent', 'empty'], 'only_missing': ['absent']}
    assert set(data['queries']) == {'q', 'only_missing'}
    audit = audit_data('fixture', data)
    assert audit['missing_qrel_pairs'] == 3
    assert audit['affected_queries'] == 2


def test_source_fix_can_resume_preparation_but_not_scores(tmp_path):
    old = dict(script_sha256='old', seed=7)
    new = dict(script_sha256='new', seed=7)
    check_manifest(tmp_path, old)
    check_manifest(tmp_path, new)
    assert json.loads((tmp_path / 'manifest.json').read_text()) == new
    assert json.loads(next((tmp_path / 'manifest_history').glob('*.json')).read_text()) == old
    with pytest.raises(ValueError, match='Configuration changed'):
        check_manifest(tmp_path, dict(new, seed=8))
    save(tmp_path / 'scidocs/reference/complete.json', {})
    with pytest.raises(ValueError, match='Configuration changed'):
        check_manifest(tmp_path, dict(new, script_sha256='third'))


def test_missing_document_semantics_match_native_loader(monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('pylate')
    import datasets
    from pylate.evaluation import NanoBEIREvaluator
    from experiments.upstream_nanobeir import native_metrics
    parts = missing_document_parts()
    monkeypatch.setattr(datasets, 'load_dataset', lambda path, name, **kw: parts[name])
    upstream = NanoBEIREvaluator(dataset_names=['scidocs']).evaluators[0]
    data = prepare_data(parts, 'fixture', 'revision')
    assert upstream.relevant_docs == {q: set(ids) for q, ids in data['qrels'].items()}
    assert upstream.queries_ids == list(data['queries'])
    ranks = {q: [{'corpus_id': 'a', 'score': 1.}] for q in data['queries']}
    metrics = native_metrics(data, ranks)
    expected = upstream.compute_metrics([ranks[q] for q in upstream.queries_ids])
    assert metrics['ndcg@k'][10] == expected['ndcg@k'][10]
    assert metrics['recall@k'][10] == pytest.approx(1/6)


def test_pool_restriction_preserves_scores_and_checks_candidates():
    ranks = {'q': [{'corpus_id': 'a', 'score': 3}, {'corpus_id': 'b', 'score': 2},
                   {'corpus_id': 'c', 'score': 1}]}
    assert restrict(ranks, {'q': ['c', 'a']}) == {'q': [ranks['q'][0], ranks['q'][2]]}
    with pytest.raises(ValueError, match='Missing'):
        restrict(ranks, {'q': ['missing']})
    with pytest.raises(ValueError, match='Duplicate'):
        restrict(ranks, {'q': ['a', 'a']})


def test_paired_queries_are_matched_by_id():
    rows = [dict(dataset='toy', query_id=q, method=m, scope='shared_dense_pool', ndcg10=s)
            for q, m, s in [('x', 'a', .9), ('y', 'a', .4), ('y', 'b', .2), ('x', 'b', .7)]]
    result, = paired(rows)
    assert result['mean_a_minus_b'] == pytest.approx(.2)
    assert result['lo'] == pytest.approx(.2)
    with pytest.raises(ValueError, match='Unpaired'):
        paired(rows[:-1])


def test_atomic_json_and_card_subset(tmp_path):
    payload = {'corpus': {'a': 'one'}, 'qrels': {'q': ['a']}}
    path = tmp_path / 'test.json'
    save(path, dict(payload, sha256=digest(payload)))
    actual = json.loads(path.read_text())
    assert actual.pop('sha256') == digest(actual)
    assert len(CARD) == 13
    assert np.mean(list(CARD.values())) == pytest.approx(.6758, abs=.0001)


def test_native_evaluation_and_candidate_loss(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('pylate')
    from experiments.upstream_nanobeir import native_retrieval, native_metrics
    torch.set_num_threads(1)
    data = dict(corpus={'a': 'red', 'b': 'blue', 'c': 'red blue'},
                queries={'q': 'red'}, qrels={'q': ['a', 'c']})
    class Toy:
        model_card_data = SimpleNamespace(set_evaluation_metrics=lambda *a: None)
        def __init__(self, multi):
            self.multi = multi
            self.calls = []
        def encode_query(self, texts, **kwargs):
            self.calls.append('query')
            return self.encode(texts, **kwargs)
        def encode_document(self, texts, **kwargs):
            self.calls.append('document')
            return self.encode(texts, **kwargs)
        def encode(self, texts, **kwargs):
            if self.multi:
                self.calls.append('query' if kwargs['is_query'] else 'document')
            vectors = {'red': [1., 0.], 'blue': [0., 1.], 'red blue': [.8, .6]}
            tensors = [torch.tensor(vectors[t]) for t in texts]
            return [t.unsqueeze(0) for t in tensors] if self.multi else torch.stack(tensors)
    for multi in [False, True]:
        model = Toy(multi)
        metrics, ranks = native_retrieval(model, data, tmp_path / str(multi),
            colbert=multi, batch_size=2, chunk_size=2)
        assert model.calls == ['query', 'document', 'document']
        assert {r['corpus_id'] for r in ranks['q']} == set(data['corpus'])
        key = 'MaxSim_ndcg@10' if multi else 'cosine_ndcg@10'
        assert metrics[key] == pytest.approx(1)
        selected = restrict(ranks, {'q': ['a', 'b']})
        conditional = native_metrics(data, selected)
        assert conditional['recall@k'][10] == .5
        assert conditional['ndcg@k'][10] < 1
