"""Short official-FDE diagnostic: cached tokens, full corpus, no model loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import wands_comparison as w
from experiments import wands_systems as s
from ras.google_fde import CONFIGS, REVISION, GoogleFDE
from ras.muvera_diagnostics import topk_numpy


def fingerprint(path, memo):
    """Hash large immutable inputs once per local file version, not per query loop."""
    path = Path(path).resolve()
    stat = path.stat()
    key = str(path)
    stamp = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    if key not in memo or memo[key]['stamp'] != stamp:
        print(f'CHECK_INPUT {path.name} ({stat.st_size/1e6:.1f} MB)', flush=True)
        memo[key] = dict(stamp=stamp, sha256=w.sha_file(path))
    return memo[key]['sha256']


def document_fdes(mapping, tokens, folder, key, block_docs):
    folder.mkdir(parents=True, exist_ok=True)
    w.lock_manifest(folder, key)
    progress = folder / 'progress.json'
    count = s.read(progress)['completed_documents'] if progress.exists() else 0
    path = folder / 'documents.npy'
    shape = (len(tokens), mapping.output_dim)
    if not 0 <= count <= len(tokens):
        raise ValueError('Invalid document cache progress')
    if path.exists():
        matrix = np.load(path, mmap_mode='r+')
        if matrix.shape != shape or matrix.dtype != np.float32:
            raise ValueError('Wrong FDE cache shape or dtype')
    else:
        if count:
            raise ValueError('FDE cache file missing; use a new FDE cache directory')
        matrix = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=shape)
    tick, initial = time.perf_counter(), count
    if count == len(tokens):
        print('REUSE_OFFICIAL_DOCUMENT_FDES', flush=True)
    for start in range(count, len(tokens), block_docs):
        end = min(start+block_docs, len(tokens))
        offsets = tokens.offsets[start:end+1]
        matrix[start:end] = mapping.encode_batch(tokens.values[offsets[0]:offsets[-1]],
            offsets-offsets[0], query=False)
        # Resume at the next block only after its data reaches the file.
        matrix.flush()
        w.save(progress, dict(completed_documents=end))
        if (start-count) % (block_docs*4) == 0 or end == len(tokens):
            elapsed = time.perf_counter()-tick
            remaining = elapsed*(len(tokens)-end)/(end-initial)
            print(f'OFFICIAL_FDE {end}/{len(tokens)} elapsed={elapsed:.1f}s remaining~{remaining:.1f}s', flush=True)
    return matrix


def run(args):
    root, cache, reference = map(Path, [args.output_dir, args.cache_dir, args.reference_dir])
    root.mkdir(parents=True, exist_ok=True)
    work = Path(args.fde_cache_dir)
    work.mkdir(parents=True, exist_ok=True)
    memo_path = work / 'input_hashes.json'
    memo = s.read(memo_path) if memo_path.exists() else {}
    old = s.read(cache / 'manifest.json')
    if s.read(cache / 'parity.json')['status'] != 'passed':
        raise ValueError('Cached embeddings have not passed reference parity')
    for name in ['manifest.json', 'dataset.json', 'candidate_ids.json']:
        if fingerprint(reference / name, memo) != old['reference_hashes'][name]:
            raise ValueError(f'Cached embeddings belong to a different reference: {name}')
    data = s.read(reference / 'dataset.json')
    if s.read(cache / 'serving.json') != dict(corpus=data['corpus'], queries=data['queries']):
        raise ValueError('Cached token row order/text differs from reference')
    build_info = s.read(Path(args.library).parent / 'build_info.json')
    vendor = Path(__file__).resolve().parents[1] / 'third_party/google_fde'
    expected = s.read(vendor / 'upstream.json')
    if (build_info['upstream'] != expected or expected['revision'] != REVISION or
            build_info['library_sha256'] != fingerprint(args.library, memo) or
            build_info['adapter_sha256'] != w.sha_file(vendor / 'bridge.cc') or
            build_info['cmake_sha256'] != w.sha_file(vendor / 'CMakeLists.txt')):
        raise ValueError('Native library provenance mismatch; rerun build_google_fde')
    tokens, queries = s.Tokens(cache / 'colbert'), s.query_tokens(cache)
    ids, all_qids = list(data['corpus']), list(data['queries'])
    if len(tokens) != len(ids) or len(queries) != len(all_qids):
        raise ValueError('Wrong number of cached embeddings')
    if not 0 < args.k <= len(ids):
        raise ValueError('k outside corpus size')
    # Prefix of the existing 60-query diagnostic sample; expansion preserves IDs.
    sample = np.random.default_rng(args.query_seed).choice(len(all_qids), min(60, len(all_qids)), replace=False)
    qidx = sample[:args.queries]
    qids = [all_qids[i] for i in qidx]
    source_hashes = {name: fingerprint(cache / ('colbert/'+name), memo)
                     for name in ['values.npy', 'offsets.npy', 'queries.npz']}
    map_config = CONFIGS[args.config] | dict(seed=args.fde_seed)
    mapping = GoogleFDE(args.library, tokens.values.shape[1], threads=args.threads, **map_config)
    fde_key = dict(upstream=expected, adapter=build_info['adapter_sha256'],
        library=build_info['library_sha256'], dimension=tokens.values.shape[1], config=map_config,
        wrapper=w.sha_file(Path(__file__).resolve().parents[1] / 'src/ras/google_fde.py'),
        inputs={k: source_hashes[k] for k in ['values.npy', 'offsets.npy']})
    folder = work / w.digest(fde_key)[:20]
    score_files = {arm: sorted((reference / arm).glob('scores_*.npy')) for arm in ['colbert', 'dense']}
    if not all(score_files.values()):
        raise ValueError('Full cached dense and ColBERT reference scores are required')
    ref_hashes = {str(p.relative_to(reference)): fingerprint(p, memo)
                  for files in score_files.values() for p in files}
    w.save(memo_path, memo)
    manifest = dict(fde=fde_key, query_ids=qids, query_tokens=source_hashes['queries.npz'],
        reference_manifest=old['reference_hashes']['manifest.json'], reference_scores=ref_hashes,
        candidates=args.candidates, k=args.k, runner=w.sha_file(__file__),
        evaluator=w.sha_file(w.__file__))
    w.lock_manifest(root, manifest)
    w.save(root / 'scope.json', dict(
        construction='Unmodified pinned Google C++ FDE source, called through a ctypes/OpenMP adapter',
        queries='Prefix of the original seeded 60-query diagnostic sample; exploratory, not held-out',
        corpus='Entire reference corpus; no filtering or label-based candidate injection',
        scoring='Exact inner-product FDE search, then saved exact LateOn MaxSim scores for reranking',
        fidelity='Candidate overlap with exact LateOn top-k; distinct from semantic relevance',
        quality='Same selected queries and full-corpus qrels for all methods; Exact=2, Partial=1',
        timing='Offline diagnostic only; no serving latency, QPS, or memory claims',
        reuse='Local document FDE cache independent of query count and candidate budgets',
        first_run='Native compilation and full-corpus FDE generation required once per configuration',
        published_results='Tests official FDE generation on LateOn/WANDS, not the paper DiskANN/PQ serving stack'))
    score_key = w.digest(dict(fde=fde_key, query_ids=qids, query_tokens=source_hashes['queries.npz']))
    scores_path = work / (score_key + '_scores.npy')
    tick = time.perf_counter()
    if scores_path.exists():
        scores = np.load(scores_path)
        print('REUSE_OFFICIAL_QUERY_SCORES', flush=True)
    else:
        matrix = document_fdes(mapping, tokens, folder, fde_key, args.block_docs)
        qfde = np.stack([mapping.encode(queries[int(i)], query=True) for i in qidx])
        scores = np.empty((len(qidx), len(ids)), dtype='float32')
        for start in range(0, len(ids), 2048):
            scores[:, start:start+2048] = qfde @ matrix[start:start+2048].T
        w.array_save(scores_path, scores)
    if scores.shape != (len(qidx), len(ids)) or not np.isfinite(scores).all():
        raise ValueError('Invalid official score cache')
    exact = np.stack([s.reference_scores(reference, 'colbert', int(i)) for i in qidx])
    dense = np.stack([s.reference_scores(reference, 'dense', int(i)) for i in qidx])
    if exact.shape != scores.shape or dense.shape != scores.shape or not np.isfinite(exact).all() or not np.isfinite(dense).all():
        raise ValueError('Reference scores do not match the full corpus')
    small = dict(corpus=data['corpus'], queries={q: data['queries'][q] for q in qids},
                 qrels={q: data['qrels'][q] for q in qids})
    pools = s.read(reference / 'candidate_ids.json')
    pos = {d: i for i, d in enumerate(ids)}
    tie = np.argsort(np.argsort(np.asarray(ids)))
    fidelity, per_query, quality = [], [], []
    method_specs = [('exact_lateon', 0), ('dense_full', 0), ('dense100_lateon', 100), ('union_lateon', 0)]
    method_specs += [('google_'+args.config, b) for b in args.candidates]
    for method, budget in method_specs:
        ranks, rows = {}, []
        for qi, qid in enumerate(qids):
            gold = topk_numpy(exact[qi], tie, args.k)
            if method == 'exact_lateon':
                chosen = topk_numpy(exact[qi], tie, min(1000, len(ids)))
                candidates = np.arange(len(ids))
            elif method == 'dense_full':
                chosen = topk_numpy(dense[qi], tie, min(1000, len(ids)))
                # Dense full measures top-k ranking overlap; rerank controls measure candidate recall.
                candidates = chosen[:args.k]
            else:
                if method == 'dense100_lateon':
                    candidates = topk_numpy(dense[qi], tie, budget)
                elif method == 'union_lateon':
                    candidates = np.array([pos[d] for d in pools[qid]], dtype=np.int64)
                else:
                    candidates = topk_numpy(scores[qi], tie, budget)
                chosen = candidates[topk_numpy(exact[qi, candidates], tie[candidates], min(1000, len(candidates)))]
            ranks[qid] = [ids[i] for i in chosen]
            rows.append(dict(method=method, candidate_budget=budget, query_id=qid,
                candidates=len(candidates), topk_recall=len(set(gold)&set(candidates))/len(gold),
                top1_candidate_recall=float(gold[0] in candidates)))
        summary, _ = w.evaluate(small, ranks, 'full_wands_quick_queries', method)
        quality.append(summary | dict(candidate_budget=budget))
        per_query.extend(rows)
        fidelity.append(dict(method=method, dimension=mapping.output_dim if method.startswith('google_') else 0,
            candidate_budget=budget, queries=len(qids), k=args.k,
            mean_candidates=float(np.mean([r['candidates'] for r in rows])),
            topk_recall=float(np.mean([r['topk_recall'] for r in rows])),
            top1_candidate_recall=float(np.mean([r['top1_candidate_recall'] for r in rows]))))
    for name, rows in [('fidelity.csv', fidelity), ('fidelity_queries.csv', per_query), ('quality.csv', quality)]:
        w.table(root / name, rows)
    w.save(root / 'complete.json', dict(status='complete', documents=len(ids), queries=len(qids),
        dimension=mapping.output_dim, diagnostic_seconds=time.perf_counter()-tick,
        fde_payload_bytes=len(ids)*mapping.output_dim*4, serving_costs_measured=False))
    print(pd.DataFrame(fidelity).to_string(index=False), flush=True)
    print(pd.DataFrame(quality)[['method', 'candidate_budget', 'ndcg10']].to_string(index=False), flush=True)
    print('OFFICIAL_QUICK_REPORT_READY', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['reference-dir', 'cache-dir', 'output-dir', 'fde-cache-dir', 'library']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--config', choices=CONFIGS, default='r20_b5_p8')
    p.add_argument('--queries', type=int, default=10)
    p.add_argument('--query-seed', type=int, default=7)
    p.add_argument('--fde-seed', type=int, default=7)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--candidates', nargs='+', type=int, default=[100, 1000, 5000])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--block-docs', type=int, default=256)
    a = p.parse_args(argv)
    if not 1 <= a.queries <= 60 or min(a.candidates+[a.threads, a.block_docs, a.k]) < 1:
        p.error('Use 1..60 queries and positive budgets, threads, block size and k')
    return a


if __name__ == '__main__':
    args = parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)
