"""Reuse WANDS embeddings for FDE ablations and CPU/GPU LateOn serving baselines."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from experiments import wands_comparison as w
from experiments import wands_systems as s
from ras.late_interaction import MuveraFDE, maxsim
from ras.muvera_diagnostics import ProjectedFDE, TensorMaxSim, topk_numpy, topk_tensor

CONFIGS = ['original_4096', 'original_8192', 'raw_r8_b4', 'paper_r20_b5_p8', 'paper_r20_b5_p16']
ARMS = ['dense_cpu', 'dense_gpu', 'late_cpu', 'late_gpu', 'shared_late_cpu', 'shared_late_gpu']


def roots(args):
    return Path(args.output_dir), Path(args.cache_dir), Path(args.reference_dir)


def lock_args(args):
    return {k: v for k, v in vars(args).items() if k not in ['phase', 'arm', 'config', 'output_dir', 'cache_dir', 'reference_dir']}


def hardware(args):
    return s.hardware(args) | dict(search_device='per_arm_cpu_or_cuda', tf32=False)


def code_hashes():
    from ras import muvera_diagnostics
    return {name: w.sha_file(path) for name, path in
            [('runner', __file__), ('helpers', muvera_diagnostics.__file__), ('wands', w.__file__), ('systems', s.__file__)]}


def prepare(args):
    root, cache, reference = roots(args)
    root.mkdir(parents=True, exist_ok=True)
    data = s.read(reference / 'dataset.json')
    old = s.read(cache / 'manifest.json')
    if s.read(cache / 'parity.json')['status'] != 'passed':
        raise ValueError('The existing embeddings must have passed reference parity')
    if args.seed != old['args']['seed'] or old['args']['repetitions'] != 8 or old['args']['partition_bits'] != 4:
        raise ValueError('These controlled ablations expect the existing seed and R=8/bits=4 run')
    for filename in ['manifest.json', 'dataset.json', 'candidate_ids.json']:
        if w.sha_file(reference / filename) != old['reference_hashes'][filename]:
            raise ValueError(f'Wrong reference for cached embeddings: {filename}')
    original = s.read(reference / 'manifest.json')
    if args.max_length != original['max_length'] or original['models'] != {k: list(v) for k, v in w.MODELS.items()}:
        raise ValueError('Reference model/input settings differ')
    for package in ['pylate', 'sentence-transformers', 'transformers']:
        if importlib.metadata.version(package) != original['versions'][package]:
            raise ValueError(f'Match reference {package} version')
    if not 0 < args.k <= len(data['corpus']):
        raise ValueError('k outside corpus size')
    files = ['serving.json', 'dense/values.npy', 'dense/queries.npy', 'colbert/values.npy',
             'colbert/offsets.npy', 'colbert/queries.npz', 'dense/model.json', 'colbert/model.json']
    for dimension in [4096, 8192]:
        if f'original_{dimension}' in args.configs:
            files += [f'fde_{dimension}/{name}' for name in ['documents.npy', 'queries.npy', 'map.npz']]
    hashes = {}
    for name in files:
        print(f'CHECK_CACHE {name}', flush=True)
        hashes[name] = w.sha_file(cache / name)
    if s.read(cache / 'serving.json') != dict(corpus=data['corpus'], queries=data['queries']):
        raise ValueError('Cached embedding row order/text differs from reference')
    w.lock_manifest(root, dict(args=lock_args(args), sources=code_hashes(), cache_hashes=hashes,
        reference_manifest_sha=w.sha_file(reference / 'manifest.json'), hardware=hardware(args)))
    qids = list(data['queries'])
    indices = np.random.default_rng(args.seed).choice(len(qids), min(args.diagnostic_queries, len(qids)), replace=False)
    w.save(root / 'diagnostic_queries.json', [qids[i] for i in indices])
    w.save(root / 'scope.json', dict(
        reuse='Document/query embeddings and labels reused; no document encoding',
        diagnostics='Fixed seeded query subset, full original corpus; no ANN; candidate sets reranked with saved exact scores',
        construction='Original CountSketch vs its uncompressed R8/B16 precursor vs per-repetition random-sign inner projections R20/B32/P8 or P16; no final projection',
        paper='Section-2/3 construction and selected parameters; not Google bitwise parity or paper result reproduction on this dataset/model',
        device_scope='Dense/whole-corpus LateOn GPU arms keep vectors resident; shared GPU LateOn uploads only candidate tokens; BM25 and shared candidate generation stay on CPU',
        selection='Partial top-k with stable document-ID cutoff ties; GPU selection transfers only chosen IDs',
        validation='After timing, live query scores checked against saved reference for all timed queries; no successful report unless checks pass; shared pools checked every request',
        memory='Fresh process per arm; measured loading+warmup+requests; score-validation happens afterward and is excluded from memory report',
        timing='Synchronized FP32, TF32 disabled; warmed sequential concurrency=1; no disk/Drive backup during requests; 30 queries x3 repeats by default',
        failures='CUDA OOM is reported as skipped_cuda_oom, never silently replaced by CPU',
        interpretation='Diagnostic subset used for ablation, not held-out selection; new FDEs have no serving latency claim yet'))
    print('CACHED_EMBEDDINGS_LOCKED', flush=True)


def geometry(args):
    root, cache, _ = roots(args)
    tokens, queries = s.Tokens(cache / 'colbert'), s.query_tokens(cache)
    counts = np.diff(tokens.offsets)
    def stats(values):
        return {str(k): float(np.percentile(values, k)) for k in [0, 50, 90, 99, 100]} | {'mean': float(np.mean(values))}
    fde = MuveraFDE(tokens.values.shape[1], repetitions=8, partition_bits=4, final_dim=None, seed=args.seed)
    chosen = np.random.default_rng(args.seed).choice(len(tokens), min(256, len(tokens)), replace=False)
    occupied, per_bucket, norms = [], [], []
    for i in chosen:
        doc = tokens[int(i)]
        norms.extend(np.linalg.norm(doc, axis=1).tolist())
        for planes in fde.planes:
            labels = ((doc @ planes.T > 0) * (1 << np.arange(4))).sum(1)
            counts_here = np.bincount(labels, minlength=16)
            occupied.append(np.count_nonzero(counts_here))
            per_bucket.extend(counts_here[counts_here > 0].tolist())
    w.save(root / 'geometry.json', dict(documents=len(tokens), token_dimension=tokens.values.shape[1],
        total_tokens=len(tokens.values), document_tokens=stats(counts),
        query_tokens=stats([len(queries[i]) for i in range(len(queries))]),
        sampled_token_norms=stats(norms), sampled_occupied_buckets=stats(occupied),
        sampled_tokens_per_occupied_bucket=stats(per_bucket), sampled_documents=len(chosen)))
    print((root / 'geometry.json').read_text(), flush=True)


def diagnose(args):
    root, cache, reference = roots(args)
    data = s.read(reference / 'dataset.json')
    qids = s.read(root / 'diagnostic_queries.json')
    positions = {q: i for i, q in enumerate(data['queries'])}
    qidx = [positions[q] for q in qids]
    ids = list(data['corpus'])
    tie_order = np.argsort(np.argsort(np.asarray(ids)))
    small = dict(corpus=data['corpus'], queries={q: data['queries'][q] for q in qids}, qrels={q: data['qrels'][q] for q in qids})
    folder = root / args.config
    folder.mkdir(exist_ok=True)
    tokens, qt = s.Tokens(cache / 'colbert'), s.query_tokens(cache)
    if args.config.startswith('original_'):
        dimension = int(args.config.split('_')[1])
        matrix = np.load(cache / f'fde_{dimension}/documents.npy', mmap_mode='r')
        qfde = np.load(cache / f'fde_{dimension}/queries.npy')[qidx]
        mapping = None
    else:
        if args.config == 'raw_r8_b4':
            mapping = MuveraFDE(tokens.values.shape[1], repetitions=8, partition_bits=4, final_dim=None, seed=args.seed)
        else:
            projection = int(args.config.rsplit('_p', 1)[1])
            mapping = ProjectedFDE(tokens.values.shape[1], 20, 5, projection, args.seed)
        dimension = mapping.output_dim
        mapping.save(folder / 'map.npz')
        qfde = np.stack([mapping.encode(qt[i], query=True) for i in qidx])
    scores = np.empty((len(qids), len(ids)), dtype='float32')
    # Save small score shards, not multi-GB intermediate FDE corpora.
    for start in range(0, len(ids), args.shard_size):
        end = min(start+args.shard_size, len(ids))
        path = folder / f'scores_{start:06d}.npy'
        if path.exists():
            block = np.load(path)
        else:
            values = matrix[start:end] if mapping is None else np.stack([mapping.encode(tokens[i], query=False) for i in range(start, end)])
            block = qfde @ values.T
            w.array_save(path, block)
        if block.shape != (len(qids), end-start) or not np.isfinite(block).all():
            raise ValueError('Invalid diagnostic shard')
        scores[:, start:end] = block
        if start % (args.shard_size*10) == 0:
            print(f'DIAGNOSTIC {args.config} {end}/{len(ids)}', flush=True)
    exact = np.stack([s.reference_scores(reference, 'colbert', i) for i in qidx])
    # Shared-pool and dense-only candidate controls use exactly the same diagnostic queries.
    cached_pools = s.read(reference / 'candidate_ids.json')
    doc_pos = {d: i for i, d in enumerate(ids)}
    summaries, rows, quality, controls = [], [], [], []
    for budget in args.candidates:
        ranks = {}
        for qi, q in enumerate(qids):
            gold = topk_numpy(exact[qi], tie_order, args.k)
            candidates = topk_numpy(scores[qi], tie_order, budget)
            chosen = candidates[topk_numpy(exact[qi, candidates], tie_order[candidates], min(1000, len(candidates)))]
            ranks[q] = [ids[i] for i in chosen]
            rows.append(dict(config=args.config, dimension=dimension, query_id=q, candidate_budget=budget,
                topk_recall=len(set(gold) & set(candidates))/len(gold), top1_candidate_recall=float(gold[0] in candidates)))
        summary, _ = w.evaluate(small, ranks, 'full_wands_diagnostic_queries', args.config)
        summary.update(candidate_budget=budget, dimension=dimension)
        quality.append(summary)
        subset = [r for r in rows if r['candidate_budget'] == budget]
        mean, lo, hi = s.interval([r['topk_recall'] for r in subset], args.seed)
        summaries.append(dict(config=args.config, dimension=dimension, candidate_budget=budget, queries=len(qids),
            k=args.k, topk_recall=mean, lo=lo, hi=hi, top1_candidate_recall=float(np.mean([r['top1_candidate_recall'] for r in subset]))))
    for qi, q in enumerate(qids):
        gold = topk_numpy(exact[qi], tie_order, args.k)
        union = {doc_pos[d] for d in cached_pools[q]}
        controls.append(dict(query_id=q, method='shared_union_pool', candidates=len(union), topk_recall=len(set(gold)&union)/len(gold)))
    w.table(folder / 'fidelity.csv', summaries)
    w.table(folder / 'fidelity_queries.csv', rows)
    w.table(folder / 'quality.csv', quality)
    w.table(root / 'shared_pool_fidelity_queries.csv', controls)
    w.save(folder / 'complete.json', dict(status='complete', dimension=dimension,
        fde_float32_payload_if_materialized_B=int(len(ids)*dimension*4), serving_costs_measured=False))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def serving(args):
    import torch
    import re
    root, cache, reference = roots(args)
    folder = root / 'cost'
    folder.mkdir(exist_ok=True)
    arm = args.arm
    gpu = arm.endswith('_gpu')
    shared = arm.startswith('shared_')
    late = 'late' in arm
    mem = s.Memory()
    samples, ranks = [], []
    try:
        if gpu and not torch.cuda.is_available():
            w.save(folder / f'{arm}_status.json', dict(arm=arm, status='skipped_no_cuda'))
            mem.finish()
            return
        tick = time.perf_counter()
        data = s.read(cache / 'serving.json')
        ids = list(data['corpus'])
        id_array = np.asarray(ids)
        tie_order = np.argsort(np.argsort(id_array))
        doc_pos = {d: i for i, d in enumerate(ids)}
        models = {}
        dense = None
        tokens = None
        scorer = None
        if not late or shared:
            models['dense'], _ = w.load_model('dense', args.device, args.max_length)
            dense_np = np.load(cache / 'dense/values.npy', mmap_mode='r')
            dense = torch.tensor(np.asarray(dense_np), device='cuda') if gpu and not shared else dense_np
        if late:
            models['colbert'], _ = w.load_model('colbert', args.device, args.max_length)
            tokens = s.Tokens(cache / 'colbert')
            if gpu:
                scorer = TensorMaxSim(tokens, 'cuda', resident=not shared, batch_docs=args.score_batch_docs)
        if shared:
            bm25 = s.lexical_index(list(data['corpus'].values()))
            pool_size = s.read(reference / 'manifest.json')['pool_size']
            saved_pools = s.read(reference / 'candidate_ids.json')
        ties_gpu = torch.as_tensor(tie_order, device='cuda') if gpu else None
        s.sync()
        startup_s = time.perf_counter()-tick
        all_qids = list(data['queries'])
        qindices = np.random.default_rng(args.seed).choice(len(all_qids), min(args.timing_queries, len(all_qids)), replace=False)
        chosen = [all_qids[i] for i in qindices]

        def request(qid, validate=False):
            qtext = data['queries'][qid]
            stages = dict(encoding_ms=0., candidate_ms=0., scoring_ms=0., selection_ms=0.)
            pool = None
            s.sync()
            start = time.perf_counter()
            with torch.inference_mode():
                if not late or shared:
                    dq, elapsed = s.measured(lambda: models['dense'].encode_query([qtext], normalize_embeddings=True,
                        convert_to_tensor=(gpu and not shared), convert_to_numpy=not (gpu and not shared), show_progress_bar=False)[0])
                    stages['encoding_ms'] += elapsed
                    if shared:
                        tick = time.perf_counter()
                        dv = dense @ dq
                        lexical = bm25.get_scores(re.findall(r'\w+', qtext.lower()))
                        pool = sorted(set(topk_numpy(dv, tie_order, pool_size)) | set(topk_numpy(lexical, tie_order, pool_size)), key=lambda i: ids[i])
                        stages['candidate_ms'] = (time.perf_counter()-tick)*1000
                        if [ids[i] for i in pool] != saved_pools[qid]:
                            raise ValueError('Live shared pool differs from quality reference')
                if late:
                    q, elapsed = s.measured(lambda: models['colbert'].encode([qtext], is_query=True, normalize_embeddings=True,
                        convert_to_numpy=True, padding=False, show_progress_bar=False)[0])
                    stages['encoding_ms'] += elapsed
                    values, stages['scoring_ms'] = s.measured(lambda: scorer.score(q, pool) if gpu else maxsim(q, tokens, pool))
                else:
                    values, stages['scoring_ms'] = s.measured(lambda: dense @ dq)
                def select():
                    if gpu:
                        tie = ties_gpu if pool is None else ties_gpu[torch.as_tensor(pool, device='cuda')]
                        result = topk_tensor(values, tie, args.k).cpu().numpy()
                    else:
                        result = topk_numpy(values, tie_order if pool is None else tie_order[pool], args.k)
                    return result if pool is None else np.asarray(pool)[result]
                selected, stages['selection_ms'] = s.measured(select)
            s.sync()
            stages['total_ms'] = (time.perf_counter()-start)*1000
            stages['excluding_encoding_ms'] = stages['total_ms']-stages['encoding_ms']
            if validate:
                # Validation only after measurements/memory recording; never part of timed requests.
                original = s.reference_scores(reference, 'colbert' if late else 'dense', all_qids.index(qid))
                expected = original if pool is None else original[pool]
                actual = values.detach().cpu().numpy() if gpu else values
                if not np.allclose(actual, expected, atol=5e-4, rtol=5e-5):
                    raise ValueError(f'{arm} live scores differ from reference')
            return stages, [ids[i] for i in selected], len(ids) if pool is None else len(pool)

        for _ in range(args.warmup):
            request(chosen[0])
        schedule = [(qid, repeat) for qid in chosen for repeat in range(args.timing_repeats)]
        np.random.default_rng(args.seed).shuffle(schedule)
        for i, (qid, repeat) in enumerate(schedule):
            stages, selected, count = request(qid)
            samples.append(dict(method=arm, scope='shared_union_pool' if shared else 'full_wands',
                                query_id=qid, repeat=repeat, candidates=count, **stages))
            ranks.append(dict(query_id=qid, repeat=repeat, ids=selected))
            if i % 10 == 0:
                print(f'SERVING {arm} {i+1}/{len(schedule)}', flush=True)
        memory = mem.finish()
        memory.update(method=arm, startup_load_s=startup_s,
                      model_tensor_payload_B=sum(p.numel()*p.element_size() for m in models.values() for p in m.parameters()),
                      token_residency=('candidate_uploads_from_cpu_mmap' if shared and gpu else 'gpu_resident' if late and gpu else 'cpu_mmap' if late else 'not_applicable'))
        # All measured queries checked; no reference-score loading contaminates measured RAM/latency.
        for qid in chosen:
            request(qid, validate=True)
        w.save(folder / f'{arm}_memory.json', memory)
        w.save(folder / f'{arm}_rankings.json', ranks)
        w.table(folder / f'{arm}_samples.csv', samples)
        frame = pd.DataFrame(samples)
        summary = dict(method=arm, scope='shared_union_pool' if shared else 'full_wands', queries=len(chosen),
                       samples=len(samples), concurrency=1, parity='all_timed_queries_passed')
        for metric in ['encoding_ms', 'candidate_ms', 'scoring_ms', 'selection_ms', 'total_ms', 'excluding_encoding_ms']:
            for percentile in [50, 95, 99] if metric in ['total_ms', 'excluding_encoding_ms'] else [50]:
                summary[f'{metric}_p{percentile}'] = float(np.percentile(frame[metric], percentile))
        summary['sequential_qps'] = 1000/float(frame.total_ms.mean())
        w.table(folder / f'{arm}_latency.csv', [summary])
        w.save(folder / f'{arm}_status.json', dict(arm=arm, status='passed'))
    except torch.cuda.OutOfMemoryError:
        mem.finish()
        w.save(folder / f'{arm}_status.json', dict(arm=arm, status='skipped_cuda_oom',
            reason='GPU-resident allocation or workspace did not fit; no CPU fallback'))
        print(f'SKIPPED_CUDA_OOM {arm}', flush=True)
    except BaseException:
        mem.finish()
        w.save(folder / f'{arm}_status.json', dict(arm=arm, status='failed'))
        raise


def report(args):
    root, _, _ = roots(args)
    for pattern, filename in [('*/fidelity.csv', 'diagnostic_fidelity.csv'), ('*/quality.csv', 'diagnostic_quality.csv')]:
        paths = sorted(root.glob(pattern))
        if paths:
            pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).to_csv(root / filename, index=False)
    passed = [r['arm'] for p in (root / 'cost').glob('*_status.json') if (r := s.read(p))['status'] == 'passed']
    statuses = [s.read(p) for p in (root / 'cost').glob('*_status.json')]
    w.save(root / 'serving_status.json', statuses)
    if passed:
        pd.concat([pd.read_csv(root / 'cost' / f'{arm}_latency.csv') for arm in passed], ignore_index=True).to_csv(root / 'latency.csv', index=False)
        w.table(root / 'memory.csv', [s.read(root / 'cost' / f'{arm}_memory.json') for arm in passed])
    print('FOLLOWUP_REPORT_READY', flush=True)


def child(args, phase, **selectors):
    command = [sys.executable, '-u', '-m', 'experiments.wands_followup']
    for key, value in vars(args).items():
        if key not in ['phase', 'arm', 'config']:
            command += ['--'+key.replace('_', '-')]
            command += list(map(str, value)) if isinstance(value, list) else [str(value)]
    command += ['--phase', phase]
    for key, value in selectors.items():
        command += ['--'+key, str(value)]
    subprocess.run(command, check=True)


def run(args):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    with threadpool_limits(limits=args.threads):
        if args.phase in ['prepare', 'all']:
            prepare(args)
        else:
            lock = s.read(Path(args.output_dir) / 'manifest.json')
            if lock['args'] != lock_args(args) or lock['sources'] != code_hashes() or lock['hardware'] != hardware(args):
                raise ValueError('Configuration/source/hardware changed; use a new follow-up directory')
        if args.phase == 'all':
            child(args, 'geometry')
            for arm in args.arms:
                child(args, 'serving', arm=arm)
            for config in args.configs:
                child(args, 'diagnose', config=config)
            report(args)
        elif args.phase != 'prepare':
            if args.phase == 'diagnose' and args.config not in args.configs:
                raise ValueError('Configuration not in locked diagnostic sweep')
            if args.phase == 'serving' and args.arm not in args.arms:
                raise ValueError('Arm not in locked serving sweep')
            globals()[args.phase](args)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['reference-dir', 'cache-dir', 'output-dir']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--phase', choices=['all', 'prepare', 'geometry', 'diagnose', 'serving', 'report'], default='all')
    p.add_argument('--config', choices=CONFIGS, default='raw_r8_b4')
    p.add_argument('--configs', choices=CONFIGS, nargs='+', default=CONFIGS)
    p.add_argument('--arm', choices=ARMS, default='shared_late_cpu')
    p.add_argument('--arms', choices=ARMS, nargs='+', default=['dense_cpu', 'dense_gpu', 'late_gpu', 'shared_late_cpu', 'shared_late_gpu'])
    p.add_argument('--candidates', nargs='+', type=int, default=[100, 500, 1000, 5000, 10000])
    for name, default in [('k', 10), ('seed', 7), ('diagnostic-queries', 60), ('timing-queries', 30),
                          ('timing-repeats', 3), ('warmup', 2), ('threads', 2), ('max-length', 512),
                          ('shard-size', 128), ('score-batch-docs', 128)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--device', default='cuda')
    args = p.parse_args(argv)
    if min(args.candidates+[args.k, args.diagnostic_queries, args.timing_queries, args.timing_repeats,
                           args.warmup, args.threads, args.shard_size, args.score_batch_docs, args.max_length]) < 1:
        p.error('Counts must be positive')
    if min(args.candidates) < args.k:
        p.error('Candidate budgets must be at least k')
    return args


if __name__ == '__main__':
    run(parse_args())
