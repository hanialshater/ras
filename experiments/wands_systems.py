"""WANDS MUVERA fidelity and isolated CPU-search/GPU-encoder cost benchmark.

The quality reference is an immutable completed wands_comparison run. This uses
the repository's transparent NumPy FDE construction, not Google's production
C++/DiskANN implementation. No ground-truth score cache is used in timed search.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

from experiments import wands_comparison as w
from ras.late_interaction import MuveraFDE, maxsim


def read(path):
    return json.loads(Path(path).read_text())


class Tokens:
    """Read-only memory mapped FP32 tokens; do not eagerly copy the whole corpus."""
    def __init__(self, folder):
        self.values = np.load(Path(folder) / 'values.npy', mmap_mode='r')
        self.offsets = np.load(Path(folder) / 'offsets.npy', mmap_mode='r')

    def __len__(self):
        return len(self.offsets)-1

    def __getitem__(self, i):
        return self.values[self.offsets[i]:self.offsets[i+1]]


def combine_tokens(shards, folder):
    """Concatenate bounded shards on disk, without a corpus-sized RAM copy."""
    counts, dim = [], None
    for path in shards:
        with np.load(path) as f:
            counts.extend(np.diff(f['offsets']).tolist())
            dim = f['values'].shape[1]
    offsets = np.r_[0, np.cumsum(counts)]
    temp = folder / 'values.tmp.npy'
    dst = np.lib.format.open_memmap(temp, mode='w+', dtype='float32', shape=(int(offsets[-1]), dim))
    cursor = 0
    for path in shards:
        with np.load(path) as f:
            values = f['values']
            dst[cursor:cursor+len(values)] = values
            cursor += len(values)
    dst.flush()
    del dst
    temp.replace(folder / 'values.npy')
    w.array_save(folder / 'offsets.npy', offsets)


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measured(fn):
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    return result, (time.perf_counter()-start)*1000


def hardware(args):
    import torch
    return dict(python=sys.version, platform=platform.platform(), cpu=platform.processor(),
                cpu_threads=args.threads, encoding_device=args.device, search_device='cpu',
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                versions={p: importlib.metadata.version(p) for p in
                          ['torch', 'numpy', 'pylate', 'sentence-transformers', 'transformers', 'faiss-cpu']})


class Memory:
    """One worker per serving pipeline. Sample RSS, separately track CUDA allocator."""
    def __init__(self):
        import psutil
        import torch
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop = threading.Event()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        def sample():
            while not self.stop.wait(.02):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def finish(self):
        import torch
        self.stop.set()
        self.thread.join()
        rss = self.process.memory_info().rss
        return dict(rss_before_load_B=self.baseline, rss_after_requests_B=rss,
                    rss_sampled_peak_B=max(self.peak, rss),
                    process_lifetime_peak_rss_B=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                    cuda_peak_allocated_B=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    cuda_peak_reserved_B=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                    scope='Isolated pipeline worker; RSS includes resident mmap pages, Python, models and metadata; CUDA is PyTorch allocator, not total device use')


def prepare(args):
    root, reference = Path(args.output_dir), Path(args.reference_dir)
    root.mkdir(parents=True, exist_ok=True)
    if read(reference / 'complete.json').get('status') != 'complete':
        raise ValueError('A completed WANDS quality run is required')
    manifest, data = read(reference / 'manifest.json'), read(reference / 'dataset.json')
    if manifest['dataset_sha256'] != w.digest(data):
        raise ValueError('Reference dataset differs from its manifest')
    if manifest['models'] != {k: list(v) for k, v in w.MODELS.items()}:
        raise ValueError('Reference uses different model revisions')
    if manifest['max_length'] != args.max_length:
        raise ValueError('max-length must match the completed quality run')
    if args.k > len(data['corpus']):
        raise ValueError('k exceeds corpus size')
    for package in ['pylate', 'sentence-transformers', 'transformers']:
        if manifest['versions'][package] != importlib.metadata.version(package):
            raise ValueError(f'Match reference version of {package}')
    files = ['dataset.json', 'manifest.json', 'candidate_ids.json', 'quality.csv']
    files += [str(p.relative_to(reference)) for arm in ['dense', 'colbert']
              for p in sorted((reference / arm).glob('scores_*.npy'))]
    if not any(p.startswith('colbert/scores_') for p in files):
        raise ValueError('Missing reference score shards')
    cfg = dict(reference_hashes={p: w.sha_file(reference / p) for p in files},
               script=w.sha_file(__file__), scorer=w.sha_file(w.__file__),
               fde_source=w.sha_file(Path(__file__).parents[1] / 'src/ras/late_interaction.py'),
               args={k: v for k, v in vars(args).items() if k not in ['phase', 'arm', 'dimension', 'reference_dir', 'output_dir']},
               hardware=hardware(args))
    w.lock_manifest(root, cfg)
    # Serving workers never load qrels or saved reference scores.
    w.save(root / 'serving.json', dict(corpus=data['corpus'], queries=data['queries']))
    w.save(root / 'scope.json', dict(
        implementation='NumPy MUVERA reference: SimHash, asymmetric sums/means, Hamming fill, CountSketch; not Google C++/DiskANN/PQ',
        reference='Completed WANDS scores reused only for offline fidelity and relevance; embeddings re-encoded with pinned checkpoints',
        latency='Live query encoding + CPU search + live FP32 MaxSim reranking + top-k; CUDA synchronized; warmed sequential requests; concurrency=1',
        devices='Encoder on selected device; dense, BM25, FDE/FAISS and token MaxSim search on CPU with fixed threads',
        cache='Token and dense corpus arrays memory mapped; pages may be evicted under RAM pressure; timings include resulting reads',
        candidates='MUVERA searches full corpus without filters; CE regenerates the same dense/BM25 union as the reference',
        fidelity='Overlap with exact LateOn top-k; separately compare HNSW FDE candidates with flat FDE candidates',
        timing_queries='Fixed seeded subset; no quality labels used for selection; p99 exploratory with few samples',
        build='Measured encoding, FDE transform, index construction and serialization; excludes downloads, model loading, validation and query encoding',
        storage='Serving corpus files and model tensor payloads reported separately; excludes offline query/score caches, temporary shards, logs and reference outputs',
        tuning='Targets selected on these evaluation queries are exploratory, not held-out guarantees'))
    print(f'REFERENCE_LOCKED products={len(data["corpus"])} queries={len(data["queries"])}', flush=True)


def encode(args):
    import torch
    root = Path(args.output_dir)
    data = read(root / 'serving.json')
    folder = root / args.arm
    folder.mkdir(exist_ok=True)
    if (folder / 'complete.json').exists():
        return
    model, meta = w.load_model(args.arm, args.device, args.max_length)
    reference_meta = read(Path(args.reference_dir) / args.arm / 'model.json')
    for key in ['repo', 'revision', 'query_length', 'do_query_expansion', 'max_seq_length', 'prompts']:
        if meta.get(key) != reference_meta.get(key):
            raise ValueError(f'Re-encoded {args.arm} differs from reference: {key}')
    w.save(folder / 'model.json', meta)
    texts = list(data['corpus'].values())
    records, shards = [], []
    for start in range(0, len(texts), args.shard_size):
        name = folder / f'shard_{start:06d}.npz'
        receipt = name.with_suffix('.json')
        if not name.exists() or not receipt.exists():
            chunk = texts[start:start+args.shard_size]
            with torch.inference_mode():
                if args.arm == 'dense':
                    values, ms = measured(lambda: model.encode_document(chunk, batch_size=args.batch_size,
                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
                    payload = dict(values=values)
                else:
                    values, ms = measured(lambda: model.encode(chunk, is_query=False, batch_size=args.batch_size,
                        normalize_embeddings=True, convert_to_numpy=True, padding=False, show_progress_bar=False))
                    payload = dict(values=np.concatenate(values), offsets=np.r_[0, np.cumsum([len(v) for v in values])])
            tick = time.perf_counter()
            temp = name.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.savez(f, **payload)
            temp.replace(name)
            w.save(receipt, dict(encoding_s=ms/1000, shard_serialization_s=time.perf_counter()-tick))
            print(f'ENCODE {args.arm} {min(start+args.shard_size,len(texts))}/{len(texts)}', flush=True)
        records.append(read(receipt))
        shards.append(name)
    tick = time.perf_counter()
    if args.arm == 'colbert':
        combine_tokens(shards, folder)
    else:
        with np.load(shards[0]) as f:
            dim = f['values'].shape[1]
        dst = np.lib.format.open_memmap(folder / 'values.npy', mode='w+', dtype='float32', shape=(len(texts), dim))
        cursor = 0
        for shard in shards:
            with np.load(shard) as f:
                values = f['values']
                dst[cursor:cursor+len(values)] = values
                cursor += len(values)
        dst.flush()
        del dst
    packing_s = time.perf_counter()-tick
    queries = list(data['queries'].values())
    with torch.inference_mode():
        if args.arm == 'dense':
            q = model.encode_query(queries, batch_size=args.batch_size, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            w.array_save(folder / 'queries.npy', q)
        else:
            q = model.encode(queries, is_query=True, batch_size=args.batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, padding=False, show_progress_bar=False)
            from ras.late_interaction import RaggedEmbeddings
            RaggedEmbeddings.from_list(q).save(folder / 'queries.npz')
    w.save(folder / 'build.json', dict(document_encoding_s=sum(r['encoding_s'] for r in records),
        shard_serialization_s=sum(r['shard_serialization_s'] for r in records), packing_s=packing_s,
        provenance='Sum of measured completed shards; may span resumed sessions; model load/query encoding excluded'))
    w.save(folder / 'complete.json', dict(status='complete'))
    # Shards can be regenerated if final arrays are deliberately removed; no double storage in normal runs.
    for shard in shards:
        shard.unlink()


def query_tokens(root):
    from ras.late_interaction import RaggedEmbeddings
    return RaggedEmbeddings.load(root / 'colbert/queries.npz')


def reference_scores(reference, arm, query_index):
    return np.concatenate([np.load(p, mmap_mode='r')[query_index] for p in sorted((reference / arm).glob('scores_*.npy'))])


def parity(args):
    root, reference = Path(args.output_dir), Path(args.reference_dir)
    data = read(root / 'serving.json')
    ids = list(data['corpus'])
    chosen = np.random.default_rng(args.seed).choice(len(data['queries']), min(args.parity_queries, len(data['queries'])), replace=False)
    dense = np.load(root / 'dense/values.npy', mmap_mode='r')
    dq = np.load(root / 'dense/queries.npy')
    tokens, tq = Tokens(root / 'colbert'), query_tokens(root)
    rows = []
    for qi in chosen:
        for arm in ['dense', 'colbert']:
            scores = dense @ dq[qi] if arm == 'dense' else maxsim(tq[int(qi)], tokens)
            expected = reference_scores(reference, arm, int(qi))
            if scores.shape != expected.shape or not np.allclose(scores, expected, atol=5e-4, rtol=5e-5):
                raise ValueError(f'{arm} re-encoding/scoring differs from reference; do not benchmark mixed representations')
            overlap = len(set(w.ranked_ids(scores, ids, args.k)) & set(w.ranked_ids(expected, ids, args.k)))/args.k
            rows.append(dict(query_index=int(qi), method=arm, max_score_error=float(np.max(np.abs(scores-expected))), topk_overlap=overlap))
            print(f'PARITY {arm} query={qi} overlap={overlap:.4f}', flush=True)
    w.save(root / 'parity.json', dict(status='passed', checked_queries=len(chosen),
        tolerance='atol=5e-4, rtol=5e-5; top-k overlap reported separately for near ties', rows=rows))


def variant(root, dimension):
    return root / f'fde_{dimension}'


def transform(args):
    root = Path(args.output_dir)
    folder = variant(root, args.dimension)
    folder.mkdir(exist_ok=True)
    if (folder / 'transform.json').exists():
        return
    tokens = Tokens(root / 'colbert')
    tick = time.perf_counter()
    fde = MuveraFDE(tokens.values.shape[1], repetitions=args.repetitions,
                    partition_bits=args.partition_bits, final_dim=args.dimension, seed=args.seed)
    fde.save(folder / 'map.npz')
    map_s = time.perf_counter()-tick
    rows, paths = [], []
    for start in range(0, len(tokens), args.shard_size):
        path = folder / f'shard_{start:06d}.npy'
        receipt = path.with_suffix('.json')
        if not path.exists() or not receipt.exists():
            tick = time.perf_counter()
            values = np.stack([fde.encode(tokens[i], query=False) for i in range(start, min(start+args.shard_size, len(tokens)))])
            transform_s = time.perf_counter()-tick
            tick = time.perf_counter()
            w.array_save(path, values)
            w.save(receipt, dict(transform_s=transform_s, serialization_s=time.perf_counter()-tick))
            print(f'FDE {args.dimension} {min(start+args.shard_size,len(tokens))}/{len(tokens)}', flush=True)
        rows.append(read(receipt))
        paths.append(path)
    tick = time.perf_counter()
    dst = np.lib.format.open_memmap(folder / 'documents.npy', mode='w+', dtype='float32', shape=(len(tokens), args.dimension))
    cursor = 0
    for path in paths:
        values = np.load(path)
        dst[cursor:cursor+len(values)] = values
        cursor += len(values)
    dst.flush()
    del dst
    packing_s = time.perf_counter()-tick
    w.array_save(folder / 'queries.npy', fde.encode_many(query_tokens(root), query=True))
    w.save(folder / 'transform.json', dict(map_construction_s=map_s,
        document_transform_s=sum(r['transform_s'] for r in rows),
        serialization_s=sum(r['serialization_s'] for r in rows)+packing_s))
    for path in paths:
        path.unlink()


def build_index(args):
    import faiss
    root = Path(args.output_dir)
    folder = variant(root, args.dimension)
    path = folder / f'{args.arm}.faiss'
    if path.exists() and path.with_suffix('.json').exists():
        return
    values = np.load(folder / 'documents.npy', mmap_mode='r')
    tick = time.perf_counter()
    if args.arm == 'flat':
        index = faiss.IndexFlatIP(values.shape[1])
    else:
        index = faiss.IndexHNSWFlat(values.shape[1], args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.ef_construction
    for start in range(0, len(values), 1024):
        index.add(values[start:start+1024])
    construction_s = time.perf_counter()-tick
    tick = time.perf_counter()
    faiss.write_index(index, str(path)+'.tmp')
    Path(str(path)+'.tmp').replace(path)
    w.save(path.with_suffix('.json'), dict(index_construction_s=construction_s,
        index_serialization_s=time.perf_counter()-tick, size_B=path.stat().st_size))
    print(f'INDEX {args.arm} dim={args.dimension} seconds={construction_s:.1f}', flush=True)


def retrieve(index, query, budget, ef_search):
    if hasattr(index, 'hnsw'):
        index.hnsw.efSearch = max(ef_search, budget)
    _, rows = index.search(np.asarray(query, dtype=np.float32).reshape(1, -1), min(budget, index.ntotal))
    result = rows[0]
    if np.any(result < 0) or len(np.unique(result)) != len(result):
        raise ValueError('ANN returned padding or duplicate candidates')
    return result


def interval(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    lo, hi = np.quantile(rng.choice(values, (2000, len(values))).mean(axis=1), [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def fidelity(args):
    import faiss
    root, reference = Path(args.output_dir), Path(args.reference_dir)
    if read(root / 'parity.json')['status'] != 'passed':
        raise ValueError('Reference parity check is required')
    folder = variant(root, args.dimension)
    data = read(reference / 'dataset.json')
    ids = list(data['corpus'])
    qids = list(data['queries'])
    queries = np.load(folder / 'queries.npy')
    flat = faiss.read_index(str(folder / 'flat.faiss'))
    index = flat if args.arm == 'flat' else faiss.read_index(str(folder / 'hnsw.faiss'))
    per_query, quality, quality_rows = [], [], []
    all_ranks = {budget: {} for budget in args.candidates}
    exact_ranks = {}
    for qi, qid in enumerate(qids):
        scores = reference_scores(reference, 'colbert', qi)
        if scores.shape != (len(ids),):
            raise ValueError('Invalid reference score count')
        truth = w.ranked_ids(scores, ids, args.k)
        exact_ranks[qid] = w.ranked_ids(scores, ids, 1000)
        for budget in args.candidates:
            candidates = retrieve(index, queries[qi], budget, args.ef_search)
            flat_ids = candidates if args.arm == 'flat' else retrieve(flat, queries[qi], budget, args.ef_search)
            selected = w.ranked_ids(scores[candidates], [ids[i] for i in candidates])
            all_ranks[budget][qid] = selected
            per_query.append(dict(query_id=qid, method=f'muvera_{args.arm}', dimension=args.dimension,
                candidate_budget=budget, actual_candidates=len(candidates), k=args.k,
                colbert_topk_recall=len(set(truth) & set(selected[:args.k]))/args.k,
                colbert_candidate_recall=len(set(truth) & {ids[i] for i in candidates})/args.k,
                flat_fde_candidate_overlap=len(set(candidates) & set(flat_ids))/len(flat_ids),
                fill_rate=min(len(selected), args.k)/args.k))
        if qi % 20 == 0:
            print(f'FIDELITY {args.arm} dim={args.dimension} queries={qi+1}/{len(qids)}', flush=True)
    _, exact_rows = w.evaluate(data, exact_ranks, 'full_wands', 'colbert_reference')
    exact_ndcg = {r['query_id']: r['ndcg10'] for r in exact_rows}
    for budget, ranks in all_ranks.items():
        summary, rows = w.evaluate(data, ranks, 'full_wands', f'muvera_{args.arm}')
        summary.update(dimension=args.dimension, candidate_budget=budget)
        delta, lo, hi = interval([r['ndcg10']-exact_ndcg[r['query_id']] for r in rows], args.seed)
        summary.update(ndcg_minus_exact=delta, ndcg_delta_lo=lo, ndcg_delta_hi=hi)
        quality.append(summary)
        for row in rows:
            row.update(dimension=args.dimension, candidate_budget=budget)
        quality_rows.extend(rows)
        w.save(folder / f'{args.arm}_{budget}_rankings.json', ranks)
        print(f'FIDELITY {args.arm} dim={args.dimension} candidates={budget}', flush=True)
    w.table(folder / f'{args.arm}_fidelity_queries.csv', per_query)
    summaries = []
    for budget in args.candidates:
        rows = [r for r in per_query if r['candidate_budget'] == budget]
        mean, lo, hi = interval([r['colbert_topk_recall'] for r in rows], args.seed)
        summaries.append(dict(method=f'muvera_{args.arm}', dimension=args.dimension, candidate_budget=budget,
            k=args.k, queries=len(rows), colbert_topk_recall=mean, lo=lo, hi=hi,
            flat_fde_candidate_overlap=float(np.mean([r['flat_fde_candidate_overlap'] for r in rows]))))
    w.table(folder / f'{args.arm}_fidelity.csv', summaries)
    w.table(folder / f'{args.arm}_quality.csv', quality)
    w.table(folder / f'{args.arm}_quality_queries.csv', quality_rows)


def lexical_index(corpus):
    from rank_bm25 import BM25Okapi
    import re
    return BM25Okapi([re.findall(r'\w+', text.lower()) for text in corpus])


def latency(args):
    import faiss
    import torch
    import re
    root = Path(args.output_dir)
    mem = Memory()
    data = read(root / 'serving.json')
    ids, queries = list(data['corpus']), list(data['queries'])
    arm = args.arm
    model, dense_model, dense, tokens, fde, index, bm25 = (None,)*7
    tick = time.perf_counter()
    if arm in ['dense', 'ce']:
        dense_model, _ = w.load_model('dense', args.device, args.max_length)
        dense = np.load(root / 'dense/values.npy', mmap_mode='r')
    if arm in ['colbert', 'flat', 'hnsw']:
        model, _ = w.load_model('colbert', args.device, args.max_length)
        tokens = Tokens(root / 'colbert')
    if arm in ['bm25', 'ce']:
        bm25 = lexical_index(list(data['corpus'].values()))
    if arm == 'ce':
        model, _ = w.load_model('ce', args.device, args.max_length)
    if arm in ['flat', 'hnsw']:
        folder = variant(root, args.dimension)
        fde = MuveraFDE.load(folder / 'map.npz')
        index = faiss.read_index(str(folder / f'{arm}.faiss'))
    sync()
    load_s = time.perf_counter()-tick
    model_payload = sum(sum(p.numel()*p.element_size() for p in m.parameters()) for m in [model, dense_model] if m is not None)
    budgets = args.candidates if index is not None else [0]
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(queries, min(args.timing_queries, len(queries)), replace=False).tolist()
    reference_pools = read(Path(args.reference_dir) / 'candidate_ids.json') if arm == 'ce' else None
    pool_size = read(Path(args.reference_dir) / 'manifest.json')['pool_size'] if arm == 'ce' else None

    def request(qid, budget):
        qtext = data['queries'][qid]
        stages = {k: 0. for k in ['encoding_ms', 'fde_ms', 'search_ms', 'rerank_ms', 'selection_ms']}
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            if arm in ['dense', 'ce']:
                q, stages['encoding_ms'] = measured(lambda: dense_model.encode_query([qtext],
                    normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0])
                values, stages['search_ms'] = measured(lambda: dense @ q)
                if arm == 'ce':
                    tick = time.perf_counter()
                    lexical = bm25.get_scores(re.findall(r'\w+', qtext.lower()))
                    pool = sorted(set(w.ranked_ids(values, ids, pool_size)) | set(w.ranked_ids(lexical, ids, pool_size)))
                    stages['search_ms'] += (time.perf_counter()-tick)*1000
                    if pool != reference_pools[qid]:
                        raise ValueError('Live CE pool differs from saved quality pool; investigate near ties/re-encoding')
                    values, stages['rerank_ms'] = measured(lambda: np.asarray(model.predict(
                        [(qtext, data['corpus'][d]) for d in pool], batch_size=args.batch_size,
                        activation_fn=torch.nn.Identity(), show_progress_bar=False)).reshape(-1))
                else:
                    pool = ids
            elif arm == 'bm25':
                values, stages['search_ms'] = measured(lambda: bm25.get_scores(re.findall(r'\w+', qtext.lower())))
                pool = ids
            else:
                q, stages['encoding_ms'] = measured(lambda: model.encode([qtext], is_query=True,
                    normalize_embeddings=True, convert_to_numpy=True, padding=False, show_progress_bar=False)[0])
                if index is not None:
                    fq, stages['fde_ms'] = measured(lambda: fde.encode(q, query=True))
                    candidates, stages['search_ms'] = measured(lambda: retrieve(index, fq, budget, args.ef_search))
                    values, stages['rerank_ms'] = measured(lambda: maxsim(q, tokens, candidates))
                    pool = [ids[i] for i in candidates]
                else:
                    values, stages['search_ms'] = measured(lambda: maxsim(q, tokens))
                    pool = ids
            selected, stages['selection_ms'] = measured(lambda: w.ranked_ids(values, pool, args.k))
        sync()
        stages['total_ms'] = (time.perf_counter()-start)*1000
        stages['excluding_encoding_ms'] = stages['total_ms']-stages['encoding_ms']
        return stages, selected, len(pool)

    for _ in range(args.warmup):
        for budget in budgets:
            request(chosen[0], budget)
    samples = []
    schedule = [(qid, b, repeat) for qid in chosen for b in budgets for repeat in range(args.timing_repeats)]
    rng.shuffle(schedule)
    for i, (qid, budget, repeat) in enumerate(schedule):
        stages, selected, actual = request(qid, budget)
        samples.append(dict(method=arm, dimension=args.dimension if index is not None else 0,
            scope='shared_union_pool' if arm == 'ce' else 'full_wands', candidate_budget=budget,
            query_id=qid, repeat=repeat, actual_candidates=actual, **stages))
        if i % 10 == 0:
            print(f'LATENCY {arm} {i+1}/{len(schedule)}', flush=True)
    name = f'{arm}_{args.dimension}' if index is not None else arm
    folder = root / 'cost'
    folder.mkdir(exist_ok=True)
    memory = mem.finish()
    memory.update(method=arm, dimension=args.dimension if index is not None else 0,
                  model_tensor_payload_B=model_payload, startup_load_s=load_s, hardware=hardware(args))
    w.save(folder / f'{name}_memory.json', memory)
    w.table(folder / f'{name}_samples.csv', samples)
    summaries = []
    for budget in budgets:
        frame = pd.DataFrame([r for r in samples if r['candidate_budget'] == budget])
        summary = dict(method=arm, dimension=args.dimension if index is not None else 0, candidate_budget=budget,
                       queries=len(chosen), samples=len(frame), concurrency=1)
        for metric in ['encoding_ms', 'fde_ms', 'search_ms', 'rerank_ms', 'selection_ms', 'total_ms', 'excluding_encoding_ms']:
            for pct in [50, 95, 99] if metric in ['total_ms', 'excluding_encoding_ms'] else [50]:
                summary[f'{metric}_p{pct}'] = float(np.percentile(frame[metric], pct))
        summary['sequential_qps'] = 1000/float(frame.total_ms.mean())
        summaries.append(summary)
    w.table(folder / f'{name}_latency.csv', summaries)


def report(args):
    root = Path(args.output_dir)
    def merge(paths, target):
        frames = [pd.read_csv(p) for p in paths]
        if frames:
            result = pd.concat(frames, ignore_index=True)
            result.to_csv(root / target, index=False)
            return result
        return pd.DataFrame()
    fidelity_rows = merge(sorted(root.glob('fde_*/*_fidelity.csv')), 'fidelity.csv')
    merge(sorted(root.glob('fde_*/*_quality.csv')), 'muvera_quality.csv')
    times = merge(sorted(root.glob('cost/*_latency.csv')), 'latency.csv')
    memories = [read(p) for p in sorted(root.glob('cost/*_memory.json'))]
    w.table(root / 'memory.csv', [{k: v for k, v in m.items() if k != 'hardware'} for m in memories])
    targets = []
    for (method, dimension), rows in fidelity_rows.groupby(['method', 'dimension']):
        for target in [.9, .95, .99]:
            passed = rows[rows.colbert_topk_recall >= target]
            eligible = times[(times.method == method.removeprefix('muvera_')) & (times.dimension == dimension)]
            joined = passed.merge(eligible, on=['dimension', 'candidate_budget'])
            row = dict(method=method, dimension=int(dimension), target=target, status='not_reached', candidate_budget=None, fidelity=None, total_ms_p50=None)
            if len(joined):
                best = joined.sort_values('total_ms_p50').iloc[0]
                row.update(status='exploratory_reached_mean', candidate_budget=int(best.candidate_budget),
                           fidelity=float(best.colbert_topk_recall), total_ms_p50=float(best.total_ms_p50))
            targets.append(row)
    w.table(root / 'latency_at_fidelity.csv', targets)
    size = lambda p: (root / p).stat().st_size
    metadata = size('serving.json')
    dense_B = size('dense/values.npy')
    tokens_B = size('colbert/values.npy') + size('colbert/offsets.npy')
    dense_build, late_build = read(root / 'dense/build.json'), read(root / 'colbert/build.json')
    def encoding_total(b):
        return b['document_encoding_s']+b['shard_serialization_s']+b['packing_s']
    rows = [dict(method='dense', dimension=0, serving_corpus_disk_B=metadata+dense_B,
                 corpus_build_s=encoding_total(dense_build), includes='metadata + dense vectors'),
            dict(method='colbert', dimension=0, serving_corpus_disk_B=metadata+tokens_B,
                 corpus_build_s=encoding_total(late_build), includes='metadata + FP32 token vectors/offsets')]
    # BM25 state is rebuilt in the worker; serialized separately for an explicit disk footprint.
    import pickle
    tick = time.perf_counter()
    bm25 = lexical_index(list(read(root / 'serving.json')['corpus'].values()))
    with (root / 'bm25.pkl').open('wb') as f:
        pickle.dump(bm25, f, protocol=5)
    bm25_build_s = time.perf_counter()-tick
    w.save(root / 'bm25_build.json', dict(index_construction_and_serialization_s=bm25_build_s))
    rows.append(dict(method='bm25', dimension=0, serving_corpus_disk_B=metadata+size('bm25.pkl'),
                     corpus_build_s=bm25_build_s, includes='metadata + serialized BM25 state'))
    rows.append(dict(method='ce', dimension=0, serving_corpus_disk_B=metadata+dense_B+size('bm25.pkl'),
                     corpus_build_s=encoding_total(dense_build)+bm25_build_s,
                     includes='metadata/text + dense vectors + BM25 state; CE has no document embeddings'))
    for dimension in args.dimensions:
        folder = variant(root, dimension)
        transformation = read(folder / 'transform.json')
        for backend in ['flat', 'hnsw']:
            index_build = read(folder / f'{backend}.json')
            corpus_s = encoding_total(late_build)+sum(transformation.values())+index_build['index_construction_s']+index_build['index_serialization_s']
            rows.append(dict(method=f'muvera_{backend}', dimension=dimension,
                serving_corpus_disk_B=metadata+tokens_B+(folder / 'map.npz').stat().st_size+(folder / f'{backend}.faiss').stat().st_size,
                corpus_build_s=corpus_s, includes='metadata + tokens + map + FAISS index (already contains FDE vectors)'))
    w.table(root / 'storage_build.csv', rows)
    print(fidelity_rows.to_string(index=False), flush=True)
    print(times.to_string(index=False), flush=True)


def child(args, phase, arm=None, dimension=None):
    command = [sys.executable, '-u', '-m', 'experiments.wands_systems']
    for key, value in vars(args).items():
        if key not in ['phase', 'arm', 'dimension']:
            command += ['--'+key.replace('_', '-')]
            command += [str(x) for x in value] if isinstance(value, list) else [str(value)]
    command += ['--phase', phase]
    if arm is not None:
        command += ['--arm', arm]
    if dimension is not None:
        command += ['--dimension', str(dimension)]
    subprocess.run(command, check=True)


def run(args):
    import torch
    import faiss
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    os.environ['PYLATE_SCORES_BACKEND'] = 'torch'
    with threadpool_limits(limits=args.threads):
        if args.phase == 'all':
            prepare(args)
            for arm in ['dense', 'colbert']:
                child(args, 'encode', arm)
            child(args, 'parity')
            for dimension in args.dimensions:
                child(args, 'transform', dimension=dimension)
                for arm in ['flat', 'hnsw']:
                    child(args, 'index', arm, dimension)
                    child(args, 'fidelity', arm, dimension)
            for arm in ['dense', 'colbert', 'bm25', 'ce']:
                child(args, 'latency', arm)
            for dimension in args.dimensions:
                for arm in ['flat', 'hnsw']:
                    child(args, 'latency', arm, dimension)
            report(args)
            w.save(Path(args.output_dir) / 'complete.json', dict(status='complete'))
        else:
            if args.phase != 'prepare':
                lock = read(Path(args.output_dir) / 'manifest.json')
                current = {k: v for k, v in vars(args).items() if k not in ['phase', 'arm', 'dimension', 'reference_dir', 'output_dir']}
                if (current != lock['args'] or lock['script'] != w.sha_file(__file__)
                        or lock['scorer'] != w.sha_file(w.__file__)
                        or lock['fde_source'] != w.sha_file(Path(__file__).parents[1] / 'src/ras/late_interaction.py')):
                    raise ValueError('Worker configuration/source differs from locked run; use a new output directory')
                if args.phase in ['transform', 'index', 'fidelity', 'latency', 'report']:
                    if read(Path(args.output_dir) / 'parity.json')['status'] != 'passed':
                        raise ValueError('Reference parity must pass before benchmarking')
                if args.phase in ['transform', 'index', 'fidelity'] or (args.phase == 'latency' and args.arm in ['flat', 'hnsw']):
                    if args.dimension not in args.dimensions:
                        raise ValueError('Worker FDE dimension must belong to the locked sweep')
            globals()[args.phase if args.phase != 'index' else 'build_index'](args)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--phase', choices=['all', 'prepare', 'encode', 'parity', 'transform', 'index', 'fidelity', 'latency', 'report'], default='all')
    p.add_argument('--arm', choices=['dense', 'colbert', 'ce', 'bm25', 'flat', 'hnsw'], default='dense')
    p.add_argument('--dimension', type=int, default=4096)
    p.add_argument('--dimensions', nargs='+', type=int, default=[4096, 8192])
    p.add_argument('--candidates', nargs='+', type=int, default=[100, 500, 1000, 2000, 5000, 10000])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--repetitions', type=int, default=8)
    p.add_argument('--partition-bits', type=int, default=4)
    p.add_argument('--hnsw-m', type=int, default=32)
    p.add_argument('--ef-construction', type=int, default=200)
    p.add_argument('--ef-search', type=int, default=200)
    p.add_argument('--timing-queries', type=int, default=30)
    p.add_argument('--timing-repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--parity-queries', type=int, default=5)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--shard-size', type=int, default=256)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cuda')
    args = p.parse_args(argv)
    if min(args.dimensions+args.candidates+[args.dimension, args.k, args.repetitions, args.threads, args.timing_queries, args.timing_repeats, args.parity_queries, args.batch_size, args.shard_size, args.max_length, args.hnsw_m, args.ef_search, args.ef_construction]) <= 0:
        p.error('Counts must be positive')
    if not 0 <= args.partition_bits <= 12 or args.warmup < 1:
        p.error('partition-bits must be 0..12; warmed timing requires warmup >= 1')
    if min(args.candidates) < args.k:
        p.error('Candidate budgets must be at least k')
    return args


if __name__ == '__main__':
    run(parse_args())
