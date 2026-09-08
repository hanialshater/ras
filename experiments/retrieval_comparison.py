"""Separate scoring quality, MUVERA fidelity and measured reference pipeline costs."""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import late_interaction_baselines as legacy
from ras.composition import compose_query
from ras.config import load_config
from ras.late_interaction import ColBERTEncoder, CrossEncoderScorer, MuveraFDE, RaggedEmbeddings, maxsim, topk_ids
from ras.metrics import bootstrap_mean_ci
from ras.queries import QuerySpec, apply_exact
from ras.repro import environment_manifest, write_json

KEYS = ['scope', 'method', 'candidate_budget', 'k']
PREPARED = ['prepared.npz', 'documents.npz', 'queries.npz', 'queries.json',
            'metadata.json', 'encoding.json', 'fit_replay.npz', 'calibrations.json']


def sync_device():
    # The synthetic path must work without torch installed.
    import sys
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def measured(function):
    sync_device()
    start = time.perf_counter()
    result = function()
    sync_device()
    return result, 1000 * (time.perf_counter() - start)


def prepare_inputs(args, root):
    """Import immutable prepared representations; never reuse prior search results."""
    target_marker = root / 'prepared.complete.json'
    if target_marker.exists():
        verify_prepared(root)
        return
    if not args.prepared_from:
        legacy.prepare(args, root)
        return
    source = Path(args.prepared_from).resolve()
    if source == root.resolve():
        raise ValueError('--prepared-from must be different from --output-dir')
    verify_prepared(source)
    previous = json.loads((source / 'request.json').read_text())
    # Search parameters may change, but every preparation-defining parameter must match.
    for name in ['synthetic', 'seed', 'queries', 'checkpoint', 'revision',
                 'query_maxlen', 'doc_maxlen']:
        if previous.get(name) != getattr(args, name):
            raise ValueError(f'prepared source has a different {name}; run fresh preparation')
    if previous.get('config_contents') != load_config(args.config):
        raise ValueError('prepared source has a different data/teacher configuration')
    for name in PREPARED:
        shutil.copy2(source / name, root / name)
    for name in ['binary_index', 'programs']:
        shutil.copytree(source / name, root / name, dirs_exist_ok=True)
    write_json(root / 'prepared_import.json', {
        'source': str(source), 'source_request_sha256': legacy.sha256(source / 'request.json'),
        'source_manifest_sha256': legacy.sha256(source / 'prepared.complete.json'),
        'note': 'Encoding build times retain their original provenance; search/indexes are rebuilt.'})
    write_json(target_marker, {'files': {name: legacy.sha256(root / name) for name in PREPARED}})
    print('[prepare] imported verified embeddings; previous result files remain untouched', flush=True)


def verify_prepared(root):
    manifest = json.loads((root / 'prepared.complete.json').read_text())
    for name, expected in manifest['files'].items():
        path = root / name
        if path.parent.resolve() != root.resolve() or not path.is_file() or legacy.sha256(path) != expected:
            raise ValueError(f'prepared checksum/path mismatch: {name}')
    if not all((root / name).is_file() for name in PREPARED):
        raise ValueError('prepared source is incomplete')


def resolve_checkpoint(name, revision=None):
    from huggingface_hub import snapshot_download
    path = Path(name)
    return str(path.resolve()) if path.is_dir() else snapshot_download(name, revision=revision,
        allow_patterns=['*.json', '*.txt', '*.model', '*.safetensors', 'pytorch_model*.bin'])


def model_payload(model):
    # Tensor payload only: not process RSS or peak GPU allocation.
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


def snapshot_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


class Models:
    def __init__(self, args, root, arrays, queries):
        self.args, self.arrays, self.queries = args, arrays, queries
        self.dense = self.colbert = self.cross = None
        self.info = {'synthetic': args.synthetic}
        self.build_rows = []
        if args.synthetic:
            self.info['note'] = 'Random token embeddings and deterministic stub pair scores; no models executed.'
            return
        from sentence_transformers import SentenceTransformer
        cfg = load_config(args.config)
        dense_path = resolve_checkpoint(cfg['retrieval']['model'])
        self.dense, ms = measured(lambda: SentenceTransformer(dense_path))
        self.build_rows.append(build_row('dense_dot_product', 'model_load', ms / 1000, 'current_run'))
        # Reuse the actual ColBERT snapshot, not a moving model alias, when importing.
        encoding = json.loads((root / 'encoding.json').read_text())
        saved = encoding.get('resolved_checkpoint')
        if saved and Path(saved).is_dir():
            checkpoint, revision = saved, None
        else:
            checkpoint, revision = args.checkpoint, args.revision
            if saved and '/snapshots/' in saved:
                revision = Path(saved).name
        self.colbert, ms = measured(lambda: ColBERTEncoder(
            checkpoint, revision=revision, query_maxlen=args.query_maxlen,
            doc_maxlen=args.doc_maxlen, batch_size=args.batch_size))
        self.build_rows.append(build_row('colbert_exact', 'model_load', ms / 1000, 'current_run'))
        self.info.update(dense_checkpoint=dense_path, colbert_checkpoint=self.colbert.resolved_checkpoint,
                         dense_device=str(self.dense.device),
                         colbert_encoding_device=str(next(self.colbert.model.parameters()).device),
                         colbert_scoring_device='cpu', dense_scoring_device='cpu',
                         dense_model_tensor_B=model_payload(self.dense),
                         colbert_model_tensor_B=model_payload(self.colbert.model),
                         dense_snapshot_disk_B=snapshot_bytes(dense_path),
                         colbert_snapshot_disk_B=snapshot_bytes(self.colbert.resolved_checkpoint))
        if not args.skip_cross_encoder:
            self.cross, ms = measured(lambda: CrossEncoderScorer(
                args.cross_encoder_checkpoint, revision=args.cross_encoder_revision,
                max_length=args.cross_encoder_maxlen, batch_size=args.cross_encoder_batch_size))
            self.build_rows.append(build_row('cross_encoder', 'model_load', ms / 1000, 'current_run'))
            self.info.update(cross_encoder_checkpoint=self.cross.resolved_checkpoint,
                             cross_encoder_device=str(next(self.cross.model.model.parameters()).device),
                             cross_encoder_model_tensor_B=model_payload(self.cross.model.model),
                             cross_encoder_snapshot_disk_B=snapshot_bytes(self.cross.resolved_checkpoint))

    def dense_query(self, text, qi):
        if self.args.synthetic:
            return self.arrays['dense_queries'][qi]
        return self.dense.encode([text], convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)[0].astype(np.float32)

    def token_query(self, text, qi):
        if self.args.synthetic:
            return self.queries[qi]
        return self.colbert.encode([text], query=True)[0]

    def cross_scores(self, text, titles, ids, qi):
        if self.args.synthetic:
            # Exercise the pair-scoring branch, never label this as cross-encoder evidence.
            return (self.arrays['dense_items'][ids] @ self.arrays['dense_queries'][qi]) ** 3
        return self.cross.score(text, titles)


def build_row(method, component, seconds, provenance, **extra):
    return dict(method=method, component=component, seconds=seconds,
                provenance=provenance, **extra)


def write_summaries(frame, root, args):
    summary = []
    metrics = ['recall', 'precision_at_k', 'ndcg', 'fill_rate', 'colbert_topk_recall',
               'eligible_colbert_candidate_recall', 'pool_relevant_coverage',
               'candidate_survival_rate']
    for key, group in frame.groupby(KEYS):
        by_query = group.groupby('query_id').mean(numeric_only=True)
        entry = dict(zip(KEYS, key))
        entry.update(queries=len(by_query), synthetic=args.synthetic,
                     zero_truth_queries=int((by_query.total_relevant == 0).sum()))
        for metric in metrics:
            entry.update({metric + '_' + stat: value for stat, value in
                          bootstrap_mean_ci(by_query[metric], n_boot=1000).items()})
        summary.append(entry)
    summary = pd.DataFrame(summary)
    summary.to_csv(root / 'summary.csv', index=False)
    quality = ['scope', 'method', 'candidate_budget', 'k', 'queries', 'synthetic',
               'recall_mean', 'precision_at_k_mean', 'ndcg_mean', 'fill_rate_mean',
               'pool_relevant_coverage_mean', 'ndcg_lo', 'ndcg_hi']
    # Keep relevance and approximation in separate, explicitly named tables.
    summary.loc[summary.scope == 'shared_minilm_pool', quality].to_csv(root / 'ranking_quality.csv', index=False)
    summary.loc[summary.scope == 'full_corpus', quality].to_csv(root / 'retrieval_quality.csv', index=False)
    approximation = ['method', 'candidate_budget', 'k', 'queries', 'synthetic',
                     'colbert_topk_recall_mean', 'colbert_topk_recall_lo',
                     'colbert_topk_recall_hi', 'eligible_colbert_candidate_recall_mean',
                     'candidate_survival_rate_mean', 'fill_rate_mean']
    summary.loc[summary.scope == 'full_corpus', approximation].to_csv(root / 'approximation.csv', index=False)
    latency = []
    components = ['query_encoding_ms', 'pool_search_ms', 'fde_ms', 'search_ms',
                  'scoring_ms', 'total_ms']
    for key, group in frame.groupby(KEYS):
        entry = dict(zip(KEYS, key))
        entry.update(samples=len(group), queries=group.query_id.nunique(),
                     warmup_per_query=args.warmup, concurrency=1, synthetic=args.synthetic,
                     timing_scope=group.timing_scope.iloc[0])
        for component in components:
            for pct in [50, 95, 99]:
                entry[f'{component}_p{pct}'] = float(np.percentile(group[component], pct))
        latency.append(entry)
    latency = pd.DataFrame(latency)
    latency.to_csv(root / 'latency.csv', index=False)
    write_paired_deltas(frame, root)
    # Exploratory operating points, not a held-out tuning result.
    matched = []
    full = summary[(summary.scope == 'full_corpus') & summary.method.str.startswith('muvera')]
    for method, group in full.groupby('method'):
        for target in args.target_recalls:
            qualifying = group[group.colbert_topk_recall_mean >= target]
            row = dict(method=method, target_colbert_topk_recall=target,
                       status='not_reached', candidate_budget=np.nan,
                       observed_recall=np.nan, total_ms_p50=np.nan, synthetic=args.synthetic)
            if len(qualifying):
                choice = qualifying.merge(latency, on=KEYS).sort_values('total_ms_p50').iloc[0]
                row.update(status='reached_on_this_run', candidate_budget=int(choice.candidate_budget),
                           observed_recall=float(choice.colbert_topk_recall_mean),
                           total_ms_p50=float(choice.total_ms_p50))
            matched.append(row)
    pd.DataFrame(matched).to_csv(root / 'matched_fidelity.csv', index=False)


def write_paired_deltas(frame, root):
    shared = frame[frame.scope == 'shared_minilm_pool']
    rows = []
    for metric in ['ndcg', 'precision_at_k', 'recall']:
        values = shared.groupby(['query_id', 'method'])[metric].mean().unstack('method')
        for left, right in combinations(sorted(values.columns), 2):
            delta = (values[left] - values[right]).dropna()
            stats = bootstrap_mean_ci(delta, n_boot=2000)
            rows.append(dict(metric=metric, method_a=left, method_b=right,
                             direction='a_minus_b', paired_queries=len(delta), **stats))
    pd.DataFrame(rows).to_csv(root / 'paired_quality_deltas.csv', index=False)


def write_costs(root, arrays, docs, dfde, models, args, build):
    np.save(root / 'dense_documents.npy', arrays['dense_items'])
    encoding = json.loads((root / 'encoding.json').read_text())
    build += models.build_rows
    build.append(build_row('colbert_exact', 'document_encoding',
                           encoding.get('document_encoding_seconds', np.nan),
                           'imported_preparation' if args.prepared_from else 'current_preparation'))
    # Cross-encoders have no reusable document vectors, but use the dense candidate index.
    build.append(build_row('cross_encoder', 'standalone_document_encoding', 0., 'not_applicable'))
    pd.DataFrame(build).assign(synthetic=args.synthetic).to_csv(root / 'build_time.csv', index=False)
    times = {(row['method'], row['component']): row['seconds'] for row in build}
    dense_time = times[('dense_dot_product', 'document_encoding')]
    colbert_time = times[('colbert_exact', 'document_encoding')]
    muvera_time = (colbert_time + times[('muvera', 'encoding_map_construction')] +
                   times[('muvera', 'document_fde_transformation')])
    totals = [('dense_dot_product', dense_time), ('colbert_exact', colbert_time),
              ('dense_pool_colbert', dense_time + colbert_time), ('muvera_flat_maxsim', muvera_time)]
    if not args.skip_cross_encoder:
        totals.append(('dense_pool_cross_encoder', dense_time))
    if args.backend == 'hnsw':
        totals.append(('muvera_hnsw_maxsim', muvera_time + times[('muvera_hnsw_maxsim', 'ann_index_construction')]))
    pd.DataFrame([dict(pipeline=name, corpus_build_seconds=seconds, synthetic=args.synthetic,
                       provenance='mixed_preparation_and_current_run' if args.prepared_from else 'current_run',
                       excludes='model download/load, dataset/teacher creation, serialization, training')
                  for name, seconds in totals]).to_csv(root / 'build_totals.csv', index=False)
    components = [('dense_documents.npy', 'dense_document_vectors'),
                  ('documents.npz', 'colbert_token_vectors_and_offsets'),
                  ('document_fde.npy', 'muvera_fde_vectors'),
                  ('fde_parameters.npz', 'muvera_encoding_parameters'),
                  ('fde_hnsw.faiss', 'hnsw_including_fde_vectors'),
                  ('metadata.json', 'shared_titles_and_metadata')]
    storage = [dict(artifact=name, component=component, disk_bytes=(root / name).stat().st_size)
               for name, component in components if (root / name).exists()]
    pd.DataFrame(storage).to_csv(root / 'storage_components.csv', index=False)
    sizes = {row['artifact']: row['disk_bytes'] for row in storage}
    common = sizes['metadata.json']
    tokens = sizes['documents.npz']
    dense = sizes['dense_documents.npy']
    plans = [('dense_dot_product', common + dense, 'metadata + dense vectors'),
             ('colbert_exact', common + tokens, 'metadata + token vectors/offsets'),
             ('dense_pool_colbert', common + dense + tokens, 'metadata + dense vectors + tokens'),
             ('muvera_flat_maxsim', common + tokens + sizes['document_fde.npy'] + sizes['fde_parameters.npz'],
              'metadata + tokens + FDE vectors + FDE maps')]
    if not args.skip_cross_encoder:
        plans.append(('dense_pool_cross_encoder', common + dense, 'metadata/titles + dense vectors; no CE document vectors'))
    if args.backend == 'hnsw':
        plans.append(('muvera_hnsw_maxsim', common + tokens + sizes['fde_hnsw.faiss'] + sizes['fde_parameters.npz'],
                      'metadata + tokens + HNSW (already includes FDE vectors) + FDE maps'))
    pd.DataFrame([dict(pipeline=name, corpus_disk_bytes=size, includes=description,
                       excludes='model checkpoints, query artifacts, training/replay artifacts',
                       synthetic=args.synthetic) for name, size, description in plans]).to_csv(root / 'storage.csv', index=False)
    import resource
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    write_json(root / 'memory.json', {
        'harness_process_peak_rss_B': peak,
        'measurement_scope': 'whole process high-water mark, includes all arms and preparation; NOT per-method RAM',
        'dense_document_array_B': arrays['dense_items'].nbytes,
        'colbert_document_array_B': docs.values.nbytes + docs.offsets.nbytes,
        'fde_document_array_B': dfde.nbytes,
        'model_tensor_payloads': {k: v for k, v in models.info.items() if k.endswith('_tensor_B')},
        'note': 'Array/model payloads exclude allocator/workspace overhead. RSS excludes CUDA memory. '
                'Per-pipeline resident/peak memory needs isolated serving processes.'})


def evaluate(root, args):
    with np.load(root / 'prepared.npz', allow_pickle=False) as f:
        arrays = dict(f)
    docs = RaggedEmbeddings.load(root / 'documents.npz')
    queries = RaggedEmbeddings.load(root / 'queries.npz')
    specs = [QuerySpec(**q) for q in json.loads((root / 'queries.json').read_text())]
    metadata = pd.read_json(root / 'metadata.json')
    n = len(docs)
    if len(metadata) != n or len(queries) != len(specs) or len(arrays['dense_items']) != n:
        raise ValueError('metadata/embedding row alignment mismatch')
    models = Models(args, root, arrays, queries)
    build = []
    if not args.synthetic:
        # A measured fresh encoding pass is necessary: upstream dense vectors may be cached.
        fresh, ms = measured(lambda: models.dense.encode(
            metadata.productDisplayName.tolist(), batch_size=args.batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True))
        if not np.allclose(fresh, arrays['dense_items'], atol=1e-3, rtol=1e-3):
            raise ValueError('dense checkpoint differs from prepared vectors; use fresh preparation')
        build.append(build_row('dense_dot_product', 'document_encoding', ms / 1000, 'fresh_verification_pass'))
        del fresh
        # Prevent mixed model versions from silently contaminating retrieval/approximation.
        for qi, spec in enumerate(specs):
            fresh_q = models.token_query(spec.text, qi)
            if fresh_q.shape != queries[qi].shape or not np.allclose(fresh_q, queries[qi], atol=2e-3, rtol=2e-3):
                raise ValueError('ColBERT checkpoint differs from prepared queries; use fresh preparation')
    else:
        build.append(build_row('dense_dot_product', 'document_encoding', np.nan, 'synthetic_no_model'))
    write_json(root / 'models.json', models.info)
    fde, ms = measured(lambda: MuveraFDE(docs.values.shape[1], repetitions=args.repetitions,
                        partition_bits=args.partition_bits, final_dim=args.fde_dim or None, seed=args.fde_seed))
    build.append(build_row('muvera', 'encoding_map_construction', ms / 1000, 'current_run'))
    dfde, ms = measured(lambda: fde.encode_many(docs, query=False))
    build.append(build_row('muvera', 'document_fde_transformation', ms / 1000, 'current_run'))
    np.save(root / 'document_fde.npy', dfde)
    fde.save(root / 'fde_parameters.npz')
    # Flat controls are direct matrix scans: no graph build beyond the representation.
    build += [build_row(name, 'ann_index_construction', 0., 'flat_scan_no_graph')
              for name in ['dense_dot_product', 'colbert_exact', 'muvera_flat_maxsim']]
    ann = None
    if args.backend == 'hnsw':
        import faiss
        faiss.omp_set_num_threads(args.threads)
        def make_ann():
            index = faiss.IndexHNSWFlat(fde.output_dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = args.ef_construction
            index.add(np.ascontiguousarray(dfde))
            return index
        ann, ms = measured(make_ann)
        build.append(build_row('muvera_hnsw_maxsim', 'ann_index_construction', ms / 1000, 'current_run'))
        _, ms = measured(lambda: faiss.write_index(ann, str(root / 'fde_hnsw.faiss')))
        build.append(build_row('muvera_hnsw_maxsim', 'index_serialization', ms / 1000, 'current_run'))
    all_ids = np.arange(n)
    rows, rankings, coverage, judgment_pool = [], [], [], []
    rng = np.random.default_rng(args.seed)
    methods = [('shared_minilm_pool', m, args.pool_size) for m in
               ['dense_dot_product', 'colbert_exact', 'binary_predicates', 'linear_predicates']]
    if not args.skip_cross_encoder:
        methods.append(('shared_minilm_pool', 'cross_encoder', args.pool_size))
    methods += [('full_corpus', m, 0) for m in ['dense_dot_product', 'colbert_exact']]
    for budget in args.candidates:
        methods.append(('full_corpus', 'muvera_flat_maxsim', budget))
        if ann is not None:
            methods.append(('full_corpus', 'muvera_hnsw_maxsim', budget))
    for qi, spec in enumerate(specs):
        exact_mask = apply_exact(metadata, spec.exact)
        eligible = all_ids[exact_mask]
        truth = exact_mask.copy()
        for name in spec.positive:
            truth &= arrays['teacher_labels'][:, legacy.MAPPING[name]]
        for name in spec.negative:
            truth &= ~arrays['teacher_labels'][:, legacy.MAPPING[name]]
        # Freeze quality candidates once; every scorer sees exactly these row IDs.
        common = topk_ids(arrays['dense_items'] @ arrays['dense_queries'][qi], all_ids, args.pool_size)
        common = common[exact_mask[common]]
        pool_truth = np.zeros(n, dtype=bool)
        pool_truth[common] = truth[common]
        pool_coverage = float(pool_truth.sum() / truth.sum()) if truth.sum() else np.nan
        coverage.append(dict(query_id=spec.query_id, global_budget=args.pool_size, eligible_count=len(eligible),
                             pool_count=len(common), full_relevant=int(truth.sum()),
                             pool_relevant=int(pool_truth.sum()), pool_relevant_coverage=pool_coverage))
        judgment_pool.extend(dict(query_id=spec.query_id, query=spec.text, product_id=metadata.id.iloc[i],
                                  title=metadata.productDisplayName.iloc[i], relevance='') for i in common)
        oracle = topk_ids(maxsim(queries[qi], docs, eligible), eligible, args.k)
        # Unmeasured warmups run for every query/method, followed by randomized measured rounds.
        for repeat in range(-args.warmup, args.timing_repeats):
            for mi in rng.permutation(len(methods)):
                scope, method, budget = methods[mi]
                stage = dict(query_encoding_ms=0., pool_search_ms=0., fde_ms=0., search_ms=0., scoring_ms=0.)
                sync_device()
                start = time.perf_counter()
                # Metadata filtering is part of each complete request's measured work.
                request_mask = apply_exact(metadata, spec.exact)
                request_eligible = all_ids[request_mask]
                ids = request_eligible
                candidate_ids = None
                if scope == 'shared_minilm_pool' or method == 'dense_dot_product':
                    dense_q, ms = measured(lambda: models.dense_query(spec.text, qi))
                    stage['query_encoding_ms'] += ms
                    if scope == 'shared_minilm_pool':
                        def get_pool():
                            found = topk_ids(arrays['dense_items'] @ dense_q, all_ids, args.pool_size)
                            return found[request_mask[found]]
                        runtime_pool, stage['pool_search_ms'] = measured(get_pool)
                        # Quality uses one immutable pool, even if online encoding has tiny numerical drift.
                        ids = common
                    else:
                        runtime_pool = None
                    # Measure live encoding, but fix the representation for model/fidelity
                    # comparisons. Batched vs single-query FP16 drift must not change truth.
                    dense_q = arrays['dense_queries'][qi]
                if method in ['colbert_exact'] or method.startswith('muvera'):
                    q, ms = measured(lambda: models.token_query(spec.text, qi))
                    stage['query_encoding_ms'] += ms
                    q = queries[qi]
                if method.startswith('muvera'):
                    qfde, stage['fde_ms'] = measured(lambda: fde.encode(q, query=True))
                    def retrieve():
                        if method == 'muvera_flat_maxsim':
                            return topk_ids(dfde @ qfde, all_ids, min(n, budget))
                        ann.hnsw.efSearch = max(args.ef_search, min(n, budget))
                        _, found = ann.search(qfde[None, :], min(n, budget))
                        return found[0][found[0] >= 0]
                    candidate_ids, stage['search_ms'] = measured(retrieve)
                    ids = candidate_ids[request_mask[candidate_ids]]
                    selected, stage['scoring_ms'] = measured(lambda: topk_ids(maxsim(q, docs, ids), ids, args.k))
                else:
                    def score():
                        if method == 'dense_dot_product':
                            scores = arrays['dense_items'][ids] @ dense_q
                        elif method == 'colbert_exact':
                            scores = maxsim(q, docs, ids)
                        elif method == 'cross_encoder':
                            scores = models.cross_scores(spec.text, metadata.productDisplayName.iloc[ids].tolist(), ids, qi)
                        else:
                            key = 'binary_logits' if method == 'binary_predicates' else 'linear_logits'
                            scores = compose_query(arrays[key][ids], legacy.MAPPING, spec.positive, spec.negative)
                        return topk_ids(scores, ids, args.k)
                    selected, stage['scoring_ms'] = measured(score)
                sync_device()
                total_ms = 1000 * (time.perf_counter() - start)
                if repeat < 0:
                    continue
                timing_scope = ('pool_generation_plus_materialized_predicate_composition' if method.endswith('predicates') else
                                'sequential_request_including_query_encoding')
                row = dict(query_id=spec.query_id, scope=scope, method=method, candidate_budget=budget,
                           k=args.k, repeat=repeat, synthetic=args.synthetic, timing_scope=timing_scope,
                           total_ms=total_ms, **stage,
                           eligible_count=len(eligible), scored_count=len(ids),
                           candidate_count=len(candidate_ids) if candidate_ids is not None else len(ids),
                           candidate_survival_rate=(len(ids) / len(candidate_ids) if candidate_ids is not None and len(candidate_ids) else np.nan),
                           pool_relevant_coverage=pool_coverage if scope == 'shared_minilm_pool' else np.nan,
                           runtime_pool_overlap=(legacy.overlap(runtime_pool, common) if scope == 'shared_minilm_pool' else np.nan),
                           colbert_topk_recall=legacy.overlap(selected, oracle) if scope == 'full_corpus' else np.nan,
                           eligible_colbert_candidate_recall=legacy.overlap(candidate_ids, oracle) if candidate_ids is not None else np.nan,
                           **legacy.ranking_metrics(pool_truth if scope == 'shared_minilm_pool' else truth, selected, args.k))
                rows.append(row)
                if repeat == 0:
                    rankings.append(dict(query_id=spec.query_id, scope=scope, method=method,
                                         candidate_budget=budget, selected_rows=selected.tolist(),
                                         selected_product_ids=metadata.id.iloc[selected].tolist(),
                                         scored_rows=ids.tolist(), candidate_rows=None if candidate_ids is None else candidate_ids.tolist()))
        pd.DataFrame(rows).to_csv(root / 'per_query.csv', index=False)
        write_json(root / 'rankings.json', rankings)
        print(f'[comparison] query {qi + 1}/{len(specs)} complete', flush=True)
    pd.DataFrame(coverage).to_csv(root / 'pool_coverage.csv', index=False)
    pd.DataFrame(judgment_pool).to_csv(root / 'judgment_pool.csv', index=False)
    write_summaries(pd.DataFrame(rows), root, args)
    write_costs(root, arrays, docs, dfde, models, args, build)
    write_json(root / 'scope.json', {
        'schema_version': 2,
        'label_source': 'synthetic' if args.synthetic else 'CLIP image teacher; not independent human judgments',
        'ranking_quality': 'Same frozen MiniLM candidate rows and title input for dot product, ColBERT, cross-encoder and predicate controls. Recall denominator is pool truth.',
        'retrieval_quality': 'Full held-out catalogue truth, not shared-pool truth.',
        'approximation': 'Exact ColBERT top-K is a scoring reference, not relevance ground truth.',
        'dot_product': 'Normalized MiniLM title/query vectors; exact inner product (equivalent to cosine here). No learned predicates.',
        'cross_encoder': 'Pretrained MS MARCO default; not an oracle. Joint query/title inference per request; no per-document CE index.',
        'filters': 'Full exact scans prefilter; common pool and MUVERA postfilter a global candidate budget. Underfill and candidate survival reported, no hidden refill.',
        'timing': 'Sequential warmed reference requests, concurrency=1, live query encoding and pool generation included. Scoring uses fixed prepared query vectors to isolate approximation from numerical encoding drift. CE pair tokenization/inference is scoring_ms. Model load/build/network excluded. Shared scorers use frozen quality pool; runtime pool overlap recorded.',
        'predicates': 'Quality controls with offline materialized logits. Latency excludes predicate execution; not a complete serving pipeline.',
        'storage': 'Actual corpus artifacts per deployment plan; HNSW includes its FDE copy. Model tensor payloads separate, model checkpoint disk excluded.',
        'build_time': 'Component seconds; ColBERT uses saved preparation timing, dense a fresh verification pass. Imported times may originate on different hardware; provenance column is mandatory.',
        'limitations': ['Teacher supervision favors task-specific predicates; human/behavioral relevance still needed.',
                       'ColBERT MaxSim is CPU NumPy, not PLAID; CE/encoding may run on GPU. See models.json.',
                       'Single-request p99 with few samples is descriptive, not a production SLA or throughput test.',
                       'Matched-fidelity choices use this run mean recall; validate selected settings on separate queries.',
                       'Whole-process peak RSS is not isolated per-method serving memory.',
                       'Synthetic runs execute no real neural models.']})


def run(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), Path(legacy.__file__), Path(__file__).parents[1] / 'src/ras/late_interaction.py']
    request = {**vars(args), 'config_contents': load_config(args.config),
               'source_hash': hashlib.sha256(b''.join(p.read_bytes() for p in source_files)).hexdigest()}
    legacy.bind_request(root, request)
    with threadpool_limits(limits=args.threads):
        prepare_inputs(args, root)
        if not args.synthetic:
            import torch
            torch.set_num_threads(args.threads)
        evaluate(root, args)
    write_json(root / 'environment.json', environment_manifest())
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix != '.zip' and p.name != 'checksums.json')
    write_json(root / 'checksums.json', {str(p.relative_to(root)): legacy.sha256(p) for p in files})
    with zipfile.ZipFile(root / 'late_interaction_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in files + [root / 'checksums.json']:
            z.write(path, path.relative_to(root))
    print(f'Results: {root}', flush=True)


def parse_args(argv=None):
    parser = legacy.build_parser()
    parser.description = __doc__
    parser.set_defaults(timing_repeats=3)
    parser.add_argument('--prepared-from', help='Completed v1/v2 run with matching preparation; results go to a new directory')
    parser.add_argument('--cross-encoder-checkpoint', default='cross-encoder/ms-marco-MiniLM-L6-v2')
    parser.add_argument('--cross-encoder-revision', default=None)
    parser.add_argument('--cross-encoder-maxlen', type=int, default=256)
    parser.add_argument('--cross-encoder-batch-size', type=int, default=32)
    parser.add_argument('--skip-cross-encoder', action='store_true')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--target-recalls', type=float, nargs='+', default=[.90, .95, .99])
    args = legacy.validate_args(parser.parse_args(argv), parser)
    if args.pool_size < args.k:
        parser.error('--pool-size must be at least k')
    if args.warmup < 0 or min(args.cross_encoder_maxlen, args.cross_encoder_batch_size) < 1:
        parser.error('warmup must be nonnegative; cross-encoder lengths/batch size positive')
    if not all(0 < value <= 1 for value in args.target_recalls):
        parser.error('--target-recalls must be in (0, 1]')
    return args


if __name__ == '__main__':
    run(parse_args())
