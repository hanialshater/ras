"""Controlled first ColBERT/MUVERA baselines. See docs/LATE_INTERACTION.md."""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from threadpoolctl import threadpool_limits

from ras.calibration import fit_scalar_calibrator
from ras.composition import compose_query
from ras.config import load_config
from ras.late_interaction import ColBERTEncoder, MuveraFDE, RaggedEmbeddings, maxsim, topk_ids
from ras.metrics import bootstrap_mean_ci
from ras.queries import QuerySpec, apply_exact, generate_query_benchmark
from ras.repro import environment_manifest, write_json
from ras.semantic_index import BinarySemanticIndex
from ras.semantic_program import ProgramStore, compile_linear_program
from ras.splits import make_protocol_split
from ras.teacher_specs import LATENT_SPECS

NAMES = [s['name'] for s in LATENT_SPECS]
MAPPING = {n: i for i, n in enumerate(NAMES)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(x):
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


def prepare(args, root):
    """Fit predicates on fit rows only; reuse upstream fashion data and query protocol."""
    cfg = load_config(args.config)
    if args.synthetic:
        rng = np.random.default_rng(900)
        tokens = [normalize(rng.normal(size=(int(rng.integers(3, 9)), 32))) for _ in range(480)]
        x = normalize(np.stack([t.mean(axis=0) for t in tokens]))
        teacher = x @ rng.normal(size=(32, len(NAMES)))
        df = pd.DataFrame({'id': np.arange(len(x)), 'productDisplayName': 'synthetic item',
                           'baseColour': 'Black', 'gender': 'Unisex', 'subCategory': 'Shoes',
                           'articleType': 'Casual Shoes', 'masterCategory': 'Footwear'})
        model = None
    else:
        from experiments.review_followup import data
        x, teacher, df, model = data(cfg, False)
    if not df.id.is_unique:
        raise ValueError('catalogue IDs must be unique')
    split = make_protocol_split(len(x), args.seed, strict=True)
    xf, xc, xt = x[split.fit_idx], x[split.cal_idx], x[split.test_idx]
    thresholds = np.quantile(teacher[split.fit_idx], 1 - cfg['teacher']['positive_prevalence'], axis=0)
    labels = teacher >= thresholds
    yf, yc, yt = labels[split.fit_idx], labels[split.cal_idx], labels[split.test_idx]
    index = BinarySemanticIndex.build(root / 'binary_index', xt, fit_embeddings=xf,
                                     item_ids=df.id.to_numpy()[split.test_idx], overwrite=True)
    bits_cal, corr_cal = index.encode_batch(xc)
    store = ProgramStore(root / 'programs')
    binary, linear, weights, intercepts, calibrations = [], [], [], [], []
    for i, name in enumerate(NAMES):
        head = SGDClassifier(loss='log_loss', class_weight='balanced', alpha=1e-4,
                             max_iter=1500, random_state=args.seed + i).fit(xf, yf[:, i])
        w, intercept = head.coef_[0].astype(np.float32), float(head.intercept_[0])
        weights.append(w); intercepts.append(intercept)
        p = compile_linear_program(index, name=name, weight=w, intercept=intercept)
        bc = fit_scalar_calibrator(p.raw_scores(bits_cal, corr_cal), yc[:, i])
        lc = fit_scalar_calibrator(head.decision_function(xc), yc[:, i])
        p.calibration_a, p.calibration_b = bc.a, bc.b
        store.save(p)
        binary.append(p.calibrated_logits(index.bits, index.corrections))
        linear.append(lc.transform(head.decision_function(xt)))
        calibrations.append({'concept': name, 'binary_a': bc.a, 'binary_b': bc.b,
                             'linear_a': lc.a, 'linear_b': lc.b})
    queries = generate_query_benchmark(
        df.iloc[split.fit_idx].reset_index(drop=True), yf,
        n_queries=args.queries, seed=args.seed + 404,
        min_fit_truth=3 if args.synthetic else int(cfg['benchmark']['min_fit_truth']),
        max_positive_latents=int(cfg['benchmark'].get('max_positive_latents', 3)),
        allow_negative=bool(cfg['benchmark'].get('allow_negative', True)))
    texts = [q.text for q in queries]
    df_test = df.iloc[split.test_idx].reset_index(drop=True)
    if args.synthetic:
        query_tokens = RaggedEmbeddings.from_list([normalize(rng.normal(size=(5, 32))) for _ in queries])
        document_tokens = RaggedEmbeddings.from_list([tokens[int(i)] for i in split.test_idx])
        dense_queries = normalize(np.stack([query_tokens[i].mean(axis=0) for i in range(len(queries))]))
        encoding = {'synthetic': True, 'note': 'random token matrices; no ColBERT model executed'}
    else:
        start = time.perf_counter()
        dense_queries = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        dense_seconds = time.perf_counter() - start
        del model
        encoder = ColBERTEncoder(args.checkpoint, revision=args.revision,
                                 query_maxlen=args.query_maxlen, doc_maxlen=args.doc_maxlen,
                                 batch_size=args.batch_size)
        start = time.perf_counter()
        document_tokens = encoder.encode(df_test.productDisplayName.tolist(), query=False)
        document_seconds = time.perf_counter() - start
        start = time.perf_counter()
        query_tokens = encoder.encode(texts, query=True)
        encoding = {'synthetic': False, 'checkpoint': args.checkpoint,
                    'resolved_checkpoint': encoder.resolved_checkpoint,
                    'document_encoding_seconds': document_seconds,
                    'query_encoding_seconds': time.perf_counter() - start,
                    'dense_query_encoding_seconds': dense_seconds,
                    'query_count': len(queries), 'timing': 'batched encoding, separately from search'}
    document_tokens.save(root / 'documents.npz')
    query_tokens.save(root / 'queries.npz')
    write_json(root / 'queries.json', [q.to_dict() for q in queries])
    df_test.to_json(root / 'metadata.json', orient='records', indent=2)
    np.savez(root / 'prepared.npz', dense_items=xt, dense_queries=dense_queries,
             teacher_labels=yt, binary_logits=np.column_stack(binary).astype(np.float32),
             linear_logits=np.column_stack(linear).astype(np.float32))
    np.savez(root / 'fit_replay.npz', dense_items=x, teacher_scores=teacher,
             fit_indices=split.fit_idx, calibration_indices=split.cal_idx, test_indices=split.test_idx,
             teacher_thresholds=thresholds, weights=np.stack(weights), intercepts=intercepts)
    write_json(root / 'calibrations.json', calibrations)
    write_json(root / 'encoding.json', encoding)
    write_json(root / 'prepared.complete.json', {'files': {
        p.name: sha256(p) for p in root.iterdir() if p.is_file() and p.name != 'prepared.complete.json'}})


def ranking_metrics(truth, selected, k):
    truth = np.asarray(truth, dtype=bool)
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) > k or len(np.unique(selected)) != len(selected):
        raise ValueError('ranking must contain at most k distinct items')
    relevance = truth[selected].astype(float)
    total = int(truth.sum())
    ideal = min(k, total)
    dcg = float((relevance / np.log2(np.arange(len(selected)) + 2)).sum())
    idcg = float((1 / np.log2(np.arange(ideal) + 2)).sum())
    return {'hits': int(relevance.sum()), 'total_relevant': total, 'returned': len(selected),
            'recall': float(relevance.sum() / total) if total else np.nan,
            'precision_at_k': float(relevance.sum() / k),
            'fill_rate': len(selected) / k,
            'ndcg': dcg / idcg if idcg else np.nan}


def overlap(a, target):
    return len(np.intersect1d(a, target)) / len(target) if len(target) else np.nan


def evaluate(root, args):
    with np.load(root / 'prepared.npz', allow_pickle=False) as f:
        arrays = dict(f)
    docs, queries = RaggedEmbeddings.load(root / 'documents.npz'), RaggedEmbeddings.load(root / 'queries.npz')
    query_specs = [QuerySpec(**q) for q in json.loads((root / 'queries.json').read_text())]
    metadata = pd.read_json(root / 'metadata.json')
    n = len(docs)
    if len(metadata) != n or len(query_specs) != len(queries):
        raise ValueError('metadata/embedding row alignment mismatch')
    fde = MuveraFDE(docs.values.shape[1], repetitions=args.repetitions,
                    partition_bits=args.partition_bits, final_dim=args.fde_dim or None, seed=args.fde_seed)
    start = time.perf_counter()
    dfde = fde.encode_many(docs, query=False)
    build_seconds = time.perf_counter() - start
    dfde_path = root / 'document_fde.npy'
    np.save(dfde_path, dfde)
    fde.save(root / 'fde_parameters.npz')
    ann = None
    index_seconds = 0.
    if args.backend == 'hnsw':
        import faiss
        faiss.omp_set_num_threads(args.threads)
        ann = faiss.IndexHNSWFlat(fde.output_dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        ann.hnsw.efConstruction = args.ef_construction
        start = time.perf_counter()
        ann.add(np.ascontiguousarray(dfde))
        index_seconds = time.perf_counter() - start
        faiss.write_index(ann, str(root / 'fde_hnsw.faiss'))
    all_ids = np.arange(n)
    rows, outputs = [], []
    rng = np.random.default_rng(args.seed)
    for qi, query in enumerate(query_specs):
        q = queries[qi]
        exact_mask = apply_exact(metadata, query.exact)
        eligible = all_ids[exact_mask]
        truth = exact_mask.copy()
        for name in query.positive:
            truth &= arrays['teacher_labels'][:, MAPPING[name]]
        for name in query.negative:
            truth &= ~arrays['teacher_labels'][:, MAPPING[name]]
        dense = arrays['dense_items'] @ arrays['dense_queries'][qi]
        common_pool = topk_ids(dense, all_ids, args.pool_size)
        common_pool = common_pool[exact_mask[common_pool]]
        # Ground truth is deliberately outside the timing region. Timed methods recompute scores.
        exact_scores = maxsim(q, docs)
        oracle = topk_ids(exact_scores[eligible], eligible, args.k)
        global_oracle = topk_ids(exact_scores, all_ids, args.k)
        query_methods = [('shared_minilm_pool', m, args.pool_size) for m in
                         ['dense', 'binary_predicates', 'linear_predicates', 'colbert_exact']]
        query_methods += [('full_corpus', m, 0) for m in ['dense', 'colbert_exact']]
        for budget in args.candidates:
            query_methods.append(('full_corpus', 'muvera_flat_maxsim', budget))
            if ann is not None:
                query_methods.append(('full_corpus', 'muvera_hnsw_maxsim', budget))
        for repeat in range(args.timing_repeats):
            for mi in rng.permutation(len(query_methods)):
                scope, method, budget = query_methods[mi]
                candidate_ids = None
                search_ms = fde_ms = rerank_ms = 0.
                start = time.perf_counter()
                if method.startswith('muvera'):
                    qfde = fde.encode(q, query=True)
                    fde_ms = 1000 * (time.perf_counter() - start)
                    search_start = time.perf_counter()
                    if method == 'muvera_flat_maxsim':
                        candidate_ids = topk_ids(dfde @ qfde, all_ids, min(n, budget))
                    else:
                        ann.hnsw.efSearch = max(args.ef_search, min(n, budget))
                        _, found = ann.search(qfde[None, :], min(n, budget))
                        candidate_ids = found[0][found[0] >= 0]
                    ids = candidate_ids[exact_mask[candidate_ids]]
                    search_ms = 1000 * (time.perf_counter() - search_start)
                    rerank_start = time.perf_counter()
                    scores = maxsim(q, docs, ids)
                    selected = topk_ids(scores, ids, args.k)
                    rerank_ms = 1000 * (time.perf_counter() - rerank_start)
                else:
                    ids = common_pool if scope == 'shared_minilm_pool' else eligible
                    if method == 'dense':
                        scores = arrays['dense_items'][ids] @ arrays['dense_queries'][qi]
                    elif method == 'colbert_exact':
                        scores = maxsim(q, docs, ids)
                    else:
                        key = 'binary_logits' if method == 'binary_predicates' else 'linear_logits'
                        scores = compose_query(arrays[key][ids], MAPPING, query.positive, query.negative)
                    selected = topk_ids(scores, ids, args.k)
                elapsed_ms = 1000 * (time.perf_counter() - start)
                metric_truth = truth.copy()
                if scope == 'shared_minilm_pool':
                    outside = np.ones(n, dtype=bool); outside[common_pool] = False
                    metric_truth[outside] = False
                row = {'query_id': query.query_id, 'scope': scope, 'method': method,
                       'candidate_budget': budget, 'k': args.k, 'repeat': repeat,
                       'synthetic': args.synthetic,
                       'timing_scope': 'materialized_head_composition' if method.endswith('predicates') else
                           ('shared_pool_scoring' if scope == 'shared_minilm_pool' else 'reference_search_after_token_encoding'),
                       'ef_search': max(args.ef_search, min(n, budget)) if method == 'muvera_hnsw_maxsim' else 0,
                       'has_negation': bool(query.negative), 'positive_count': len(query.positive),
                       'eligible_count': len(eligible), 'common_pool_size': len(common_pool),
                       'candidate_count': len(candidate_ids) if candidate_ids is not None else len(ids),
                       'scored_count': len(ids), 'elapsed_ms': elapsed_ms,
                       'fde_ms': fde_ms, 'search_ms': search_ms, 'rerank_ms': rerank_ms,
                       'colbert_topk_recall': overlap(selected, oracle) if scope == 'full_corpus' else np.nan,
                       'global_colbert_candidate_recall': overlap(candidate_ids, global_oracle) if candidate_ids is not None else np.nan,
                       'eligible_colbert_candidate_recall': overlap(candidate_ids, oracle) if candidate_ids is not None else np.nan,
                       **ranking_metrics(metric_truth, selected, args.k)}
                rows.append(row)
                if repeat == 0:
                    outputs.append({'query_id': query.query_id, 'scope': scope, 'method': method,
                                    'candidate_budget': budget, 'selected_rows': selected.tolist(),
                                    'selected_product_ids': metadata.id.iloc[selected].tolist(),
                                    'candidate_rows': None if candidate_ids is None else candidate_ids.tolist()})
        # Query-level checkpoint leaves useful evidence even if a later query fails.
        pd.DataFrame(rows).to_csv(root / 'per_query.csv', index=False)
        print(f'[late interaction] query {qi + 1}/{len(queries)} complete', flush=True)
    frame = pd.DataFrame(rows)
    summary = []
    keys = ['scope', 'method', 'candidate_budget', 'k']
    for key, group in frame.groupby(keys):
        by_query = group.groupby('query_id').mean(numeric_only=True)
        entry = dict(zip(keys, key))
        entry.update(queries=len(by_query), zero_truth_queries=int((by_query.total_relevant == 0).sum()))
        for metric in ['recall', 'precision_at_k', 'ndcg', 'fill_rate', 'colbert_topk_recall',
                       'global_colbert_candidate_recall', 'eligible_colbert_candidate_recall']:
            stats = bootstrap_mean_ci(by_query[metric], n_boot=500)
            entry.update({metric + '_' + name: value for name, value in stats.items()})
        for percentile in [50, 95, 99]:
            entry[f'elapsed_p{percentile}_ms'] = float(np.percentile(group.elapsed_ms, percentile))
        summary.append(entry)
    pd.DataFrame(summary).to_csv(root / 'summary.csv', index=False)
    write_json(root / 'rankings.json', outputs)
    # Array payloads and serialized index size, not a claim about process RSS.
    write_json(root / 'memory.json', {
        'n_items': n, 'colbert_token_values_B': docs.values.nbytes,
        'colbert_offsets_B': docs.offsets.nbytes, 'fde_values_B': dfde.nbytes,
        'fde_parameters_file_B': (root / 'fde_parameters.npz').stat().st_size,
        'dense_values_B': arrays['dense_items'].nbytes,
        'hnsw_serialized_B': (root / 'fde_hnsw.faiss').stat().st_size if ann is not None else 0,
        'note': 'HNSW serialization includes its FDE copy. Prototype also retains dfde for flat control. '
                'Total RSS, temporary workspaces, model weights and allocator overhead are not measured.'})
    write_json(root / 'scope.json', {
        'label_source': 'synthetic' if args.synthetic else 'CLIP image teacher; not human relevance',
        'colbert': 'pretrained title-only ColBERTv2; no fashion fine-tuning',
        'muvera': 'NumPy reference FDE, optional Faiss HNSW; no PQ or Google DiskANN engine',
        'rsa': 'supervised Binary1-LS2-int4 and FP32 head scores, materialized offline for quality comparison',
        'filters': 'exact filters apply identically to all final results; MUVERA post-filters global candidate budget',
        'timings': 'CPU reference search/scoring after token encoding; batched encoding reported separately; '
                   'shared-pool timings exclude MiniLM pool generation and predicate execution. No production latency claims.',
        'fde_document_build_seconds': build_seconds, 'hnsw_build_seconds': index_seconds,
        'limitations': ['Generated compound queries and eight fixed semantic concepts.',
                       'Supervised predicates and pretrained ColBERT have different supervision.',
                       'Empty-truth queries retained in precision/fill; recall and nDCG undefined.',
                       'Approximation recall targets ColBERT ordering, not relevance.',
                       'Single seed per run. Repeat --seed and --fde-seed in separate output directories.']})


def bind_request(root, request):
    """Restart failed preparation after a code fix; preserve completed-run provenance."""
    request_file = root / 'request.json'
    if request_file.exists():
        previous = json.loads(request_file.read_text())
        if previous != request:
            unchanged_config = ({k: v for k, v in previous.items() if k != 'source_hash'} ==
                                {k: v for k, v in request.items() if k != 'source_hash'})
            completed_or_evaluated = any((root / name).exists() for name in [
                'prepared.complete.json', 'per_query.csv', 'summary.csv',
                'rankings.json', 'late_interaction_results.zip'])
            if not unchanged_config or completed_or_evaluated:
                raise ValueError('output directory belongs to another configuration/code version; choose a new directory')
            history = root / 'attempts'
            history.mkdir(exist_ok=True)
            (history / ('request-' + sha256(request_file) + '.json')).write_bytes(request_file.read_bytes())
            print('[restart] prior preparation failed; archiving its request and rebuilding with the code fix', flush=True)
    write_json(request_file, request)


def run(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    request = {**vars(args), 'config_contents': config,
               'source_hash': hashlib.sha256((Path(__file__).read_bytes() +
                   (Path(__file__).parents[1] / 'src/ras/late_interaction.py').read_bytes())).hexdigest()}
    bind_request(root, request)
    marker = root / 'prepared.complete.json'
    if marker.exists():
        for name, checksum in json.loads(marker.read_text())['files'].items():
            if not (root / name).exists() or sha256(root / name) != checksum:
                raise ValueError(f'prepared input checksum mismatch: {name}')
        print('[resume] reusing verified prepared embeddings and labels', flush=True)
    else:
        prepare(args, root)
    environment = environment_manifest()
    for package in ['colbert-ai', 'faiss-cpu', 'huggingface-hub', 'threadpoolctl']:
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = None
    write_json(root / 'environment.json', environment)
    with threadpool_limits(limits=args.threads):
        evaluate(root, args)
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix != '.zip' and p.name != 'checksums.json')
    write_json(root / 'checksums.json', {str(p.relative_to(root)): sha256(p) for p in files})
    with zipfile.ZipFile(root / 'late_interaction_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files + [root / 'checksums.json']:
            z.write(p, p.relative_to(root))
    print(f'Results: {root}', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/binary_bbq.yaml')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--checkpoint', default='colbert-ir/colbertv2.0')
    p.add_argument('--revision', default=None)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--fde-seed', type=int, default=7)
    p.add_argument('--queries', type=int, default=30)
    p.add_argument('--k', type=int, default=50)
    p.add_argument('--pool-size', type=int, default=5000)
    p.add_argument('--candidates', type=int, nargs='+', default=[100, 500, 1000])
    p.add_argument('--repetitions', type=int, default=8)
    p.add_argument('--partition-bits', type=int, default=4)
    p.add_argument('--fde-dim', type=int, default=4096, help='0 disables final CountSketch')
    p.add_argument('--backend', choices=['flat', 'hnsw'], default='flat')
    p.add_argument('--hnsw-m', type=int, default=32)
    p.add_argument('--ef-construction', type=int, default=200)
    p.add_argument('--ef-search', type=int, default=128)
    p.add_argument('--query-maxlen', type=int, default=32)
    p.add_argument('--doc-maxlen', type=int, default=180)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--timing-repeats', type=int, default=1)
    args = p.parse_args(argv)
    for name in ['queries', 'k', 'pool_size', 'repetitions', 'hnsw_m', 'ef_construction',
                 'ef_search', 'query_maxlen', 'doc_maxlen', 'batch_size', 'threads', 'timing_repeats']:
        if getattr(args, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if args.fde_dim < 0 or not 0 <= args.partition_bits <= 12 or min(args.candidates) < args.k:
        p.error('invalid FDE dimensions or candidate budget smaller than k')
    if len(set(args.candidates)) != len(args.candidates):
        p.error('candidate budgets must be distinct')
    return args


if __name__ == '__main__':
    run(parse_args())
