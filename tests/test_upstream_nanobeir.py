"""Native evaluator integration uses deterministic embeddings and no downloads."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from experiments.upstream_nanobeir import CARD, digest, paired, restrict, save


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
        def encode(self, texts, **kwargs):
            vectors = {'red': [1., 0.], 'blue': [0., 1.], 'red blue': [.8, .6]}
            tensors = [torch.tensor(vectors[t]) for t in texts]
            return [t.unsqueeze(0) for t in tensors] if self.multi else torch.stack(tensors)
    for multi in [False, True]:
        metrics, ranks = native_retrieval(Toy(multi), data, tmp_path / str(multi),
            colbert=multi, batch_size=2, chunk_size=2)
        assert {r['corpus_id'] for r in ranks['q']} == set(data['corpus'])
        key = 'MaxSim_ndcg@10' if multi else 'cosine_ndcg@10'
        assert metrics[key] == pytest.approx(1)
        selected = restrict(ranks, {'q': ['a', 'b']})
        conditional = native_metrics(data, selected)
        assert conditional['recall@k'][10] == .5
        assert conditional['ndcg@k'][10] < 1
