"""Native integration tests: GOOGLE_FDE_LIBRARY is required to execute them."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ras.google_fde import GoogleFDE, CONFIGS
from ras.late_interaction import RaggedEmbeddings, maxsim
from experiments import wands_comparison as w
from experiments import wands_google_fde as quick


@pytest.fixture
def library():
    path = os.environ.get('GOOGLE_FDE_LIBRARY')
    if not path:
        pytest.skip('Set GOOGLE_FDE_LIBRARY to run real C++ integration')
    assert Path(path).is_file(), 'Build the official library first'
    return path


def test_official_identity_sum_mean_algebra_and_validation(library):
    mapping = GoogleFDE(library, 3, repetitions=4, bits=0, projection=0)
    q = np.array([[1., 2., -1.], [0., -1., 3.]], dtype='float32')
    d = np.array([[-2., 1., 1.], [4., -3., 0.]], dtype='float32')
    assert mapping.encode(q, query=True) @ mapping.encode(d, query=False) == pytest.approx(4*q.sum(0).dot(d.mean(0)))
    for bad in [np.empty((0, 3)), np.array([[np.nan, 1, 2]]), np.ones((2, 4))]:
        with pytest.raises(ValueError):
            mapping.encode(bad, query=False)


@pytest.mark.parametrize('config', list(CONFIGS))
def test_native_batch_parallelism_and_query_document_configuration(library, config):
    rng = np.random.default_rng(3)
    docs = RaggedEmbeddings.from_list([rng.normal(size=(n, 8)).astype('float32') for n in [1, 4, 9]])
    parallel = GoogleFDE(library, 8, **CONFIGS[config], threads=2)
    serial = GoogleFDE(library, 8, **CONFIGS[config], threads=1)
    for query in [True, False]:
        batch = parallel.encode_batch(docs.values, docs.offsets, query=query)
        separate = np.stack([serial.encode(d, query=query) for d in docs])
        np.testing.assert_array_equal(batch, separate)
    # With one document token and filling enabled, exact uncompressed FDE score
    # equals R times query/document dot product regardless of partition choices.
    identity = GoogleFDE(library, 8, repetitions=8, bits=4, projection=0)
    assert identity.encode(docs[1], query=True) @ identity.encode(docs[0], query=False) == pytest.approx(
        8*docs[1].sum(0).dot(docs[0][0]), rel=1e-5, abs=1e-5)


def test_quick_full_corpus_report_and_resume_without_reencoding(tmp_path, library, monkeypatch):
    rng = np.random.default_rng(12)
    ref, cache = tmp_path/'reference', tmp_path/'cache'
    ref.mkdir(); cache.mkdir()
    data = dict(corpus={str(i): f'document {i}' for i in range(12)},
        queries={'q0': 'one', 'q1': 'two', 'q2': 'three'},
        qrels={f'q{i}': {str(i): 2, str(i+1): 1} for i in range(3)})
    pools = {q: list(data['corpus'])[:6] for q in data['queries']}
    w.save(ref/'dataset.json', data)
    w.save(ref/'candidate_ids.json', pools)
    w.save(ref/'manifest.json', {'test': True})
    w.save(cache/'manifest.json', {'reference_hashes': {n: w.sha_file(ref/n)
        for n in ['manifest.json', 'dataset.json', 'candidate_ids.json']}})
    w.save(cache/'parity.json', {'status': 'passed'})
    w.save(cache/'serving.json', {k: data[k] for k in ['corpus', 'queries']})
    docs = RaggedEmbeddings.from_list([rng.normal(size=(n, 8)).astype('float32') for n in range(1, 13)])
    queries = RaggedEmbeddings.from_list([rng.normal(size=(2, 8)).astype('float32') for _ in range(3)])
    w.array_save(cache/'colbert/values.npy', docs.values)
    w.array_save(cache/'colbert/offsets.npy', docs.offsets)
    queries.save(cache/'colbert/queries.npz')
    exact = np.stack([maxsim(q, docs) for q in queries])
    w.array_save(ref/'colbert/scores_000000.npy', exact)
    w.array_save(ref/'dense/scores_000000.npy', rng.normal(size=exact.shape).astype('float32'))
    args = quick.parse_args(['--reference-dir', str(ref), '--cache-dir', str(cache),
        '--output-dir', str(tmp_path/'report'), '--fde-cache-dir', str(tmp_path/'fdes'),
        '--library', library, '--queries', '2', '--k', '3', '--candidates', '3', '12', '--block-docs', '5'])
    quick.run(args)
    result = pd.read_csv(tmp_path/'report/fidelity.csv')
    assert result[result.method.str.startswith('google_') & (result.candidate_budget == 12)].topk_recall.item() == 1
    quality = pd.read_csv(tmp_path/'report/quality.csv')
    expected = quality[quality.method == 'exact_lateon'].ndcg10.item()
    assert quality[quality.method.str.startswith('google_') & (quality.candidate_budget == 12)].ndcg10.item() == expected
    assert set(quality.queries) == {2}
    def forbidden(*a, **kw):
        raise AssertionError('Cached documents must not be re-encoded')
    monkeypatch.setattr(quick, 'document_fdes', forbidden)
    quick.run(args)  # Existing query scores avoid all FDE work.
    # Expanding the query set should use the existing document matrix.
    monkeypatch.undo()
    original = GoogleFDE.encode_batch
    def queries_only(self, *a, query, **kw):
        if not query:
            forbidden()
        return original(self, *a, query=query, **kw)
    monkeypatch.setattr(GoogleFDE, 'encode_batch', queries_only)
    args.queries = 3
    args.output_dir = str(tmp_path/'expanded')
    quick.run(args)
    assert json.loads((tmp_path/'expanded/manifest.json').read_text())['query_ids'][:2] == json.loads((tmp_path/'report/manifest.json').read_text())['query_ids']
    # A mismatched reference cannot silently reuse the cache.
    w.save(ref/'candidate_ids.json', {})
    with pytest.raises(ValueError, match='different reference'):
        quick.run(args)
