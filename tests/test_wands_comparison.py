import json
import numpy as np
import pandas as pd
import pytest

from experiments import wands_comparison as w


def fixture():
    products = pd.DataFrame([dict(product_id=d, product_name=t, product_class='furniture',
        product_features='', product_description='') for d, t in [('a', 'red chair'), ('b', 'blue chair'), ('c', 'table')]])
    queries = pd.DataFrame([dict(query_id='q', query='red chair'), dict(query_id='z', query='unknown')])
    labels = pd.DataFrame([dict(query_id=q, product_id=d, label=g) for q, d, g in
        [('q', 'a', 'Exact'), ('q', 'a', 'Exact'), ('q', 'b', 'Partial'), ('q', 'b', 'Irrelevant'),
         ('z', 'c', 'Irrelevant')]])
    return products, queries, labels


def test_data_duplicates_conflicts_and_empty_relevance_are_explicit():
    data, audit = w.prepare_data(*fixture())
    assert data['qrels'] == {'q': {'a': 2, 'b': 0}, 'z': {'c': 0}}
    assert audit['duplicate_rows'] == 2
    assert len(audit['conflicting_pairs']) == 1
    assert audit['no_positive_queries'] == ['z']
    assert 'nan' not in ''.join(data['corpus'].values())
    small, _ = w.prepare_data(*fixture(), query_limit=1)
    assert len(small['queries']) == 1 and small['corpus'] == data['corpus']


def test_graded_metrics_keep_full_denominator_and_unjudged_results():
    pytest.importorskip('ir_measures')
    data = dict(queries={'q': 'chair'}, corpus={'a': 'a', 'b': 'b', 'c': 'c', 'u': 'u'},
                qrels={'q': {'a': 2, 'b': 1, 'c': 2}})
    result, _ = w.evaluate(data, {'q': ['b', 'u', 'a']}, 'shared_union_pool', 'test')
    dcg = 1 + 2/np.log2(4)
    ideal = 2 + 2/np.log2(3) + 1/np.log2(4)
    assert result['ndcg10'] == pytest.approx(dcg/ideal)
    assert result['exact_mrr10'] == pytest.approx(1/3)
    assert result['exact_recall100'] == .5
    assert result['judged_at10'] == .2
    data['queries']['zero'] = 'no positives'
    data['qrels']['zero'] = {'u': 0}
    _, rows = w.evaluate(data, {'q': ['a'], 'zero': ['u']}, 'full_wands', 'test')
    assert len(rows) == 2 and rows[1]['ndcg10'] == 0


def test_native_maxsim_is_batch_invariant_even_with_negative_matches():
    torch = pytest.importorskip('torch')
    pytest.importorskip('pylate')
    from pylate.scores import colbert_scores_pairwise
    q = [np.array([[1., 0.]], dtype='float32'), np.array([[0., 1.], [1., 0.]], dtype='float32')]
    d = [np.array([[-1., 0.]], dtype='float32'), np.array([[-.5, -.5], [-.8, -.2]], dtype='float32'),
         np.array([[.5, .5]], dtype='float32')]
    values = w.colbert_block(q, d, 'cpu', query_batch=2, document_batch=32)
    np.testing.assert_allclose(values, w.colbert_block(q, d, 'cpu', query_batch=1, document_batch=1))
    for i, query in enumerate(q):
        for j, doc in enumerate(d):
            expected = colbert_scores_pairwise([torch.tensor(query)], [torch.tensor(doc)], backend='torch')[0]
            assert values[i, j] == pytest.approx(expected.item())
    assert values[0, 0] == -1  # No phantom zero padding wins.


def test_pools_are_union_without_label_injection_and_ties_are_stable():
    data, _ = w.prepare_data(*fixture())
    dense = np.array([[1., 2., 3.], [1., 2., 3.]])
    lexical = np.array([[1., 3., 2.], [1., 3., 2.]])
    assert w.make_pools(data, dense, lexical, 1) == {'q': ['b', 'c'], 'z': ['b', 'c']}
    assert w.ranked_ids([1, 1], ['b', 'a']) == ['a', 'b']


def test_manifest_refuses_mixed_results_and_array_roundtrip(tmp_path):
    w.lock_manifest(tmp_path, {'model': 'a', 'seed': 7})
    w.lock_manifest(tmp_path, {'model': 'a', 'seed': 7})
    with pytest.raises(ValueError, match='new output'):
        w.lock_manifest(tmp_path, {'model': 'b', 'seed': 7})
    a = np.array([[1., 2.]], dtype='float32')
    w.array_save(tmp_path / 'scores.npy', a)
    np.testing.assert_array_equal(a, np.load(tmp_path / 'scores.npy'))


def test_paired_intervals_are_separate_for_each_scope():
    rows = [dict(scope=s, method=m, query_id=q, ndcg10=v) for s in ['full_wands', 'shared_union_pool']
            for m, v in [('a', .8), ('b', .5)] for q in ['1', '2']]
    out = w.paired_deltas(rows, 7)
    assert len(out) == 2
    assert all(r['paired_queries'] == 2 and r['mean_a_minus_b'] == pytest.approx(.3) for r in out)
