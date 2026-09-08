import numpy as np
import pytest
from ras.late_interaction import RaggedEmbeddings, MuveraFDE, maxsim, topk_ids
from experiments.late_interaction_baselines import ranking_metrics, parse_args


def test_maxsim_matches_naive_with_negative_scores_and_variable_lengths():
    q = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    docs = [np.array([[-1., -2.]]), np.array([[1., -4.], [-3., 2.]])]
    ragged = RaggedEmbeddings.from_list(docs)
    expected = np.array([(q @ d.T).max(axis=1).sum() for d in docs])
    np.testing.assert_allclose(maxsim(q, ragged, batch_size=1), expected)
    np.testing.assert_allclose(maxsim(q, ragged, [1, 0]), expected[[1, 0]])
    assert maxsim(q, ragged, []).shape == (0,)
    assert expected[0] == -3  # zero padding must not turn a negative match into zero


def test_ragged_roundtrip_and_invalid_offsets(tmp_path):
    d = RaggedEmbeddings.from_list([np.eye(2), np.ones((1, 2))])
    d.save(tmp_path / 'tokens.npz')
    loaded = RaggedEmbeddings.load(tmp_path / 'tokens.npz')
    np.testing.assert_array_equal(loaded[1], d[1])
    for offsets in [[0, 0, 3], [0, 4], [0., 3.]]:
        with pytest.raises(ValueError):
            RaggedEmbeddings(d.values, np.array(offsets))
    with pytest.raises(ValueError):
        RaggedEmbeddings.from_list([np.empty((0, 2))])


def test_fde_asymmetric_sum_mean_and_singleton_exactness():
    # A singleton document fills every bucket with that token. Without final
    # projection the FDE dot must equal exact MaxSim, for any query and seed.
    rng = np.random.default_rng(12)
    for seed in [3, 17, 29]:
        fde = MuveraFDE(7, repetitions=5, partition_bits=3, final_dim=None, seed=seed)
        q, d = rng.normal(size=(4, 7)), rng.normal(size=(1, 7))
        score = fde.encode(q, query=True) @ fde.encode(d, query=False)
        np.testing.assert_allclose(score, (q @ d.T).sum(), rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(fde.encode(np.repeat(d, 3, axis=0), query=False),
                                   fde.encode(d, query=False), atol=1e-6)
        np.testing.assert_allclose(fde.encode(np.tile(q, (2, 1)), query=True),
                                   2 * fde.encode(q, query=True), atol=1e-6)


def test_fde_zero_bits_is_sum_dot_mean_and_projection_is_reproducible():
    q = np.array([[1., 0.], [0., 2.]])
    d = np.array([[3., 0.], [0., 4.]])
    fde = MuveraFDE(2, repetitions=1, partition_bits=0, final_dim=None)
    assert fde.encode(q, query=True) @ fde.encode(d, query=False) == 5.5
    a = MuveraFDE(2, final_dim=16, seed=91)
    b = MuveraFDE(2, final_dim=16, seed=91)
    np.testing.assert_array_equal(a.encode(d, query=False), b.encode(d, query=False))
    assert np.isfinite(a.encode(d, query=False)).all()
    with pytest.raises(ValueError):
        a.encode(np.empty((0, 2)), query=False)


def test_topk_ties_use_global_ids_and_do_not_fabricate_results():
    np.testing.assert_array_equal(topk_ids([1., 1., 2.], [8, 3, 9], 9), [9, 3, 8])
    assert len(topk_ids([], [], 4)) == 0


def test_metrics_penalize_underfill_and_preserve_zero_truth_queries():
    row = ranking_metrics(np.array([True, False, True]), [0], 2)
    assert row['recall'] == .5 and row['precision_at_k'] == .5
    assert row['fill_rate'] == .5 and row['ndcg'] < 1
    empty = ranking_metrics(np.zeros(3, dtype=bool), [], 2)
    assert empty['precision_at_k'] == 0 and empty['fill_rate'] == 0
    assert np.isnan(empty['recall']) and np.isnan(empty['ndcg'])
    with pytest.raises(ValueError):
        ranking_metrics(np.ones(3, dtype=bool), [0, 0], 2)


def test_candidate_budget_cannot_be_less_than_k():
    with pytest.raises(SystemExit):
        parse_args(['--output-dir', '/tmp/example', '--k', '50', '--candidates', '10'])


def test_fde_saved_maps_replay_exactly(tmp_path):
    fde = MuveraFDE(3, repetitions=3, partition_bits=2, final_dim=10)
    path = tmp_path / 'fde.npz'
    fde.save(path)
    restored = MuveraFDE.load(path)
    tokens = np.eye(3, dtype=np.float32)
    for is_query in [True, False]:
        np.testing.assert_array_equal(fde.encode(tokens, query=is_query),
                                      restored.encode(tokens, query=is_query))


def test_full_candidate_budget_and_empty_filters(tmp_path):
    import json
    import pandas as pd
    from experiments.late_interaction_baselines import prepare, evaluate
    args = parse_args(['--synthetic', '--output-dir', str(tmp_path), '--queries', '2',
                       '--k', '5', '--candidates', '1000', '--repetitions', '2',
                       '--partition-bits', '2', '--fde-dim', '32'])
    prepare(args, tmp_path)
    # Force an empty exact-filter result for one query. It must still be reported.
    path = tmp_path / 'queries.json'
    queries = json.loads(path.read_text())
    queries[0]['exact'] = [['baseColour', 'NoSuchColour']]
    path.write_text(json.dumps(queries))
    evaluate(tmp_path, args)
    frame = pd.read_csv(tmp_path / 'per_query.csv')
    empty = frame[frame.query_id == queries[0]['query_id']]
    assert len(empty) == 7
    assert (empty.returned == 0).all()
    assert (empty.precision_at_k == 0).all()
    assert empty.recall.isna().all()
    rows = json.loads((tmp_path / 'rankings.json').read_text())
    actual = {r['method']: r['selected_rows'] for r in rows
              if r['query_id'] == queries[1]['query_id'] and r['scope'] == 'full_corpus'}
    assert actual['muvera_flat_maxsim'] == actual['colbert_exact']


def notebook_process_runner():
    import ast
    import json
    from pathlib import Path
    notebook = json.loads((Path(__file__).parents[1] /
        'notebooks/ras_late_interaction_colab.ipynb').read_text())
    source = ast.parse(''.join(notebook['cells'][1]['source']))
    function = next(node for node in source.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'run_logged')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<notebook-runner>', 'exec'), namespace)
    return namespace['run_logged']


def test_notebook_failure_surfaces_stderr_and_saves_log(tmp_path, capsys):
    import sys
    runner = notebook_process_runner()
    log = tmp_path / 'nested' / 'failure.log'
    with pytest.raises(RuntimeError, match='underlying failure'):
        runner([sys.executable, '-u', '-c',
                "import sys; print('progress', flush=True); raise ValueError('underlying failure')"],
               log_path=log)
    assert 'ValueError: underlying failure' in log.read_text()
    assert 'ValueError: underlying failure' in capsys.readouterr().out


def test_notebook_success_streams_output(tmp_path, capsys):
    import sys
    assert notebook_process_runner()([sys.executable, '-c', "print('finished')"],
                                     log_path=tmp_path / 'success.log') == 0
    assert 'finished' in capsys.readouterr().out
