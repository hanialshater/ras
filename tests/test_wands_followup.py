import numpy as np
import pandas as pd
import pytest

from ras.muvera_diagnostics import ProjectedFDE, TensorMaxSim, topk_numpy, topk_tensor
from ras.late_interaction import RaggedEmbeddings, MuveraFDE, maxsim
from experiments import wands_comparison as w
from experiments import wands_systems as s
from experiments import wands_followup as f


def test_inner_projection_identity_matches_uncompressed_construction(tmp_path):
    rng = np.random.default_rng(123)
    tokens = rng.normal(size=(7, 8)).astype('float32')
    reference = MuveraFDE(8, repetitions=3, partition_bits=3, final_dim=None, seed=7)
    projected = ProjectedFDE(8, 3, 3, 8, 7)
    for query in [False, True]:
        np.testing.assert_allclose(projected.encode(tokens, query=query), reference.encode(tokens, query=query), atol=1e-6)
    projected = ProjectedFDE(8, 3, 3, 2, 7)
    # Independently project the original buckets; verifies projection order/scaling.
    raw = reference.encode(tokens, query=False).reshape(3, 8, 8)
    expected = np.concatenate([(raw[r] @ projected.projections[r].T).ravel() for r in range(3)])
    np.testing.assert_allclose(projected.encode(tokens, query=False), expected, atol=1e-6)
    projected.save(tmp_path / 'map.npz')
    np.testing.assert_array_equal(ProjectedFDE.load(tmp_path / 'map.npz').encode(tokens, query=False), projected.encode(tokens, query=False))


def test_partial_topk_preserves_cutoff_ties():
    torch = pytest.importorskip('torch')
    scores = np.array([1., 1., 3., 1., -1., 2.], dtype='float32')
    ids = ['z', 'c', 'd', 'a', 'x', 'b']
    order = np.argsort(np.argsort(ids))
    for k in [1, 3, 4, 6, 100]:
        expected = w.ranked_ids(scores, ids, k)
        assert [ids[i] for i in topk_numpy(scores, order, k)] == expected
        assert [ids[i] for i in topk_tensor(torch.tensor(scores), torch.tensor(order), k)] == expected


@pytest.mark.parametrize('resident', [True, False])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_segmented_scoring_without_padding_preserves_negative_scores_and_candidate_order(resident, device):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA runtime unavailable')
    rng = np.random.default_rng(8)
    docs = RaggedEmbeddings.from_list([-abs(rng.normal(size=(n, 8))).astype('float32') for n in [1, 3, 5, 2]])
    query = abs(rng.normal(size=(2, 8))).astype('float32')
    scorer = TensorMaxSim(docs, device, resident=resident, batch_docs=2)
    for candidates in [None, [3, 0, 2], [1], [1, 1, 3]]:
        np.testing.assert_allclose(scorer.score(query, candidates).cpu().numpy(), maxsim(query, docs, candidates), atol=1e-5)


def test_cached_followup_end_to_end(tmp_path, monkeypatch):
    from test_wands_systems import toy_reference
    args, _ = toy_reference(tmp_path, monkeypatch)
    args.repetitions, args.partition_bits = 8, 4
    s.prepare(args)
    for arm in ['dense', 'colbert']:
        args.arm = arm
        s.encode(args)
    s.parity(args)
    follow = f.parse_args(['--reference-dir', args.reference_dir, '--cache-dir', args.output_dir,
        '--output-dir', str(tmp_path / 'followup'), '--configs', 'raw_r8_b4', 'paper_r20_b5_p8',
        '--arms', 'dense_cpu', 'shared_late_cpu', '--device', 'cpu', '--diagnostic-queries', '2',
        '--timing-queries', '2', '--timing-repeats', '1', '--warmup', '1', '--candidates', '10', '20'])
    f.prepare(follow)
    f.geometry(follow)
    for config in follow.configs:
        follow.config = config
        f.diagnose(follow)
    for arm in follow.arms:
        follow.arm = arm
        f.serving(follow)
    f.report(follow)
    root = tmp_path / 'followup'
    fidelity = pd.read_csv(root / 'diagnostic_fidelity.csv')
    assert (fidelity[fidelity.candidate_budget == 20].topk_recall == 1).all()
    assert set(pd.read_csv(root / 'latency.csv').method) == {'dense_cpu', 'shared_late_cpu'}
    assert (pd.read_csv(root / 'memory.csv').rss_sampled_peak_B > 0).all()
    assert all(row['status'] == 'passed' for row in s.read(root / 'serving_status.json'))
