"""WANDS product search: native neural scoring and graded ir_measures metrics.

Run with --phase data, smoke, or compare. No training or label-derived candidates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import re
import time
import urllib.request

import numpy as np
import pandas as pd

from experiments.upstream_nanobeir import save, digest, table

WANDS_REVISION = '3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5'
MODELS = {
    'dense': ('lightonai/DenseOn', 'cb9947ebccb33862d24e3c7ca2edb25e51acd887'),
    'colbert': ('lightonai/LateOn', '62911e105059585d244384c7d17826e35f669c17'),
    'ce': ('Alibaba-NLP/gte-reranker-modernbert-base', 'f7481e6055501a30fb19d090657df9ec1f79ab2c'),
}
GRADES = {'Exact': 2, 'Partial': 1, 'Irrelevant': 0}
FIELDS = ['product_name', 'product_class', 'product_features', 'product_description']


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_data(products, queries, labels, *, query_limit=0, seed=7, text_mode='full'):
    for frame, key in [(products, 'product_id'), (queries, 'query_id')]:
        if frame[key].duplicated().any():
            raise ValueError(f'Duplicate {key}')
    if not set(labels.label) <= GRADES.keys():
        raise ValueError('Unknown WANDS label')
    if not set(labels.product_id) <= set(products.product_id) or not set(labels.query_id) <= set(queries.query_id):
        raise ValueError('Judgment IDs missing from products/queries')
    corpus = {}
    for row in products.to_dict('records'):
        fields = FIELDS if text_mode == 'full' else FIELDS[:1]
        text = '\n'.join(f'{f}: {row[f].strip()}' for f in fields if row[f].strip())
        if not text:
            raise ValueError(f'Empty product text: {row["product_id"]}')
        corpus[row['product_id']] = text
    qs = dict(zip(queries.query_id, queries['query']))
    if any(not q.strip() for q in qs.values()):
        raise ValueError('Empty query')
    # Explicit conservative resolution, never a last-row-wins dictionary overwrite.
    grouped = labels.groupby(['query_id', 'product_id'], sort=False).label.agg(list)
    judgments = {q: {} for q in qs}
    conflicts = []
    for (q, d), values in grouped.items():
        grade = min(GRADES[v] for v in values)
        judgments[q][d] = grade
        if len(set(values)) > 1:
            conflicts.append(dict(query_id=q, product_id=d, labels=values, retained_grade=grade))
    if any(not judgments[q] for q in qs):
        raise ValueError('Query has no judgments')
    if query_limit:
        selected = set(np.random.default_rng(seed).choice(sorted(qs), min(query_limit, len(qs)), replace=False))
        qs = {q: text for q, text in qs.items() if q in selected}
    audit = dict(products=len(corpus), source_queries=len(queries), evaluated_queries=len(qs),
                 raw_judgments=len(labels), unique_pairs=len(grouped),
                 duplicate_rows=len(labels)-len(grouped), conflicting_pairs=conflicts,
                 conflict_policy='minimum grade; identical duplicates collapsed',
                 label_grades=GRADES, text_fields=FIELDS if text_mode == 'full' else FIELDS[:1],
                 no_positive_queries=[q for q in qs if not any(judgments[q].values())])
    return dict(corpus=corpus, queries=qs, qrels={q: judgments[q] for q in qs}), audit


def load_data(root, args):
    raw = Path(args.data_dir) if args.data_dir else root / 'source'
    raw.mkdir(parents=True, exist_ok=True)
    frames, hashes = {}, {}
    for name in ['product', 'query', 'label']:
        path = raw / f'{name}.csv'
        if not path.exists():
            url = f'https://raw.githubusercontent.com/wayfair/WANDS/{WANDS_REVISION}/dataset/{name}.csv'
            print(f'DOWNLOAD {name}', flush=True)
            temp = path.with_suffix('.tmp')
            with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as f:
                __import__('shutil').copyfileobj(response, f)
            temp.replace(path)
        hashes[name] = sha_file(path)
        frames[name] = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    data, audit = prepare_data(frames['product'], frames['query'], frames['label'],
                              query_limit=args.queries, seed=args.seed, text_mode=args.text_mode)
    audit.update(source='https://github.com/wayfair/WANDS',
                 revision=WANDS_REVISION if not args.data_dir else 'user_supplied_files',
                 source_sha256=hashes, dataset_sha256=digest(data))
    return data, audit


def lock_manifest(root, config):
    path = root / 'manifest.json'
    config = json.loads(json.dumps(config))
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Configuration/data/source changed: use a new output directory')
    save(path, config)


def array_save(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.save(f, values, allow_pickle=False)
    temp.replace(path)


def load_model(method, device, max_length):
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from pylate import models
    repo, revision = MODELS[method]
    path = snapshot_download(repo, revision=revision)
    kw = dict(device=device, model_kwargs={'torch_dtype': torch.float32, 'attn_implementation': 'sdpa'},
              config_kwargs={'reference_compile': False})
    if method == 'dense':
        model = SentenceTransformer(path, **kw)
        model.max_seq_length = max_length
    elif method == 'colbert':
        model = models.ColBERT(path, **kw)
        model.document_length = max_length
        model.max_seq_length = max_length
    else:
        model = CrossEncoder(path, max_length=max_length, **kw)
    model.eval()
    meta = dict(repo=repo, revision=revision, parameters=sum(p.numel() for p in model.parameters()),
                dtype=str(next(model.parameters()).dtype), device=str(model.device))
    for attr in ['prompts', 'default_prompt_name', 'max_seq_length', 'max_length',
                 'query_length', 'document_length', 'do_query_expansion']:
        if hasattr(model, attr):
            meta[attr] = getattr(model, attr)
    return model, meta


def colbert_block(queries, documents, device, query_batch=8, document_batch=32):
    """Use PyLate MaxSim; group documents by length to avoid artificial pad matches.

    PyLate's torch mask multiplies padded similarities by zero. A zero may beat a
    negative real match. Equal-length document groups make scores batch invariant.
    Query padding contributes exactly zero to the sum.
    """
    import torch
    from pylate.scores import colbert_scores
    q = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x) for x in queries], batch_first=True).to(device)
    out = np.empty((len(queries), len(documents)), dtype=np.float32)
    groups = {}
    for i, doc in enumerate(documents):
        if len(doc) == 0:
            raise ValueError('Empty document embedding')
        groups.setdefault(len(doc), []).append(i)
    with torch.inference_mode():
        for ids in groups.values():
            for start in range(0, len(ids), document_batch):
                idx = ids[start:start+document_batch]
                d = torch.stack([torch.as_tensor(documents[i]) for i in idx]).to(device)
                for qi in range(0, len(q), query_batch):
                    out[qi:qi+query_batch, idx] = colbert_scores(q[qi:qi+query_batch], d,
                        backend='torch').cpu().numpy()
    return out


def neural_scores(method, data, root, args):
    import torch
    from sentence_transformers.util import dot_score
    folder = root / method
    folder.mkdir(exist_ok=True)
    model, meta = load_model(method, args.device, args.max_length)
    save(folder / 'model.json', meta)
    queries, documents = list(data['queries'].values()), list(data['corpus'].values())
    with torch.inference_mode():
        if method == 'dense':
            q = model.encode_query(queries, batch_size=args.batch_size, normalize_embeddings=True,
                                   convert_to_tensor=True, show_progress_bar=True)
        else:
            q = model.encode(queries, is_query=True, batch_size=args.batch_size,
                             normalize_embeddings=True, convert_to_numpy=True, padding=False,
                             show_progress_bar=True)
        scores = np.empty((len(queries), len(documents)), dtype=np.float32)
        for start in range(0, len(documents), args.shard_size):
            end = min(start+args.shard_size, len(documents))
            path = folder / f'scores_{start:06d}.npy'
            if path.exists():
                block = np.load(path, allow_pickle=False)
            else:
                tick = time.perf_counter()
                text = documents[start:end]
                if method == 'dense':
                    d = model.encode_document(text, batch_size=args.batch_size, normalize_embeddings=True,
                                               convert_to_tensor=True, show_progress_bar=False)
                    block = dot_score(q, d).cpu().numpy()
                else:
                    d = model.encode(text, is_query=False, batch_size=args.batch_size,
                                     normalize_embeddings=True, convert_to_numpy=True, padding=False,
                                     show_progress_bar=False)
                    block = colbert_block(q, d, args.device)
                array_save(path, block)
                print(f'{method.upper()} {end}/{len(documents)} products; shard {time.perf_counter()-tick:.1f}s', flush=True)
            if block.shape != (len(queries), end-start) or not np.isfinite(block).all():
                raise ValueError(f'Invalid score shard: {path}')
            scores[:, start:end] = block
    del model, q
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def lexical_scores(data):
    from rank_bm25 import BM25Okapi
    tokenize = lambda text: re.findall(r'\w+', text.lower())
    index = BM25Okapi([tokenize(t) for t in data['corpus'].values()])
    return np.asarray([index.get_scores(tokenize(q)) for q in data['queries'].values()], dtype=np.float32)


def ranked_ids(values, ids, limit=None):
    # Stable ties by document ID across full corpus and every candidate scope.
    order = np.lexsort((np.asarray(ids), -np.asarray(values)))
    return [ids[i] for i in order[:limit]]


def make_pools(data, dense, lexical, budget):
    ids = list(data['corpus'])
    # Symmetric union broadens coverage beyond a dense-only pool; no labels used.
    return {q: sorted(set(ranked_ids(dense[i], ids, budget)) |
                      set(ranked_ids(lexical[i], ids, budget)))
            for i, q in enumerate(data['queries'])}


def cross_scores(data, pools, root, args):
    import torch
    folder = root / 'ce'
    folder.mkdir(exist_ok=True)
    model, meta = load_model('ce', args.device, args.max_length)
    save(folder / 'model.json', meta)
    results = {}
    for i, (q, ids) in enumerate(pools.items()):
        path = folder / f'query_{q}.json'
        if path.exists():
            row = json.loads(path.read_text())
            if row['ids'] != ids:
                raise ValueError('CE cached candidates differ')
            values = np.asarray(row['scores'])
        else:
            with torch.inference_mode():
                values = np.asarray(model.predict([(data['queries'][q], data['corpus'][d]) for d in ids],
                    batch_size=args.batch_size, show_progress_bar=False, activation_fn=torch.nn.Identity())).reshape(-1)
            save(path, dict(ids=ids, scores=values.tolist()))
        if values.shape != (len(ids),) or not np.isfinite(values).all():
            raise ValueError('CE needs one finite score per candidate')
        results[q] = ranked_ids(values, ids)
        print(f'CE {i+1}/{len(pools)} queries', flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def evaluate(data, rankings, scope, method):
    import ir_measures as irm
    metrics = {irm.nDCG@10: 'ndcg10', irm.RR(rel=2)@10: 'exact_mrr10',
               irm.P(rel=2)@10: 'exact_precision10', irm.R(rel=2)@100: 'exact_recall100'}
    # Synthetic monotonic scores preserve the saved deterministic ranking, including ties.
    run = {q: {d: float(len(ids)-i) for i, d in enumerate(ids)} for q, ids in rankings.items()}
    per_query = {q: dict(dataset='wands', query_id=q, scope=scope, method=method) for q in data['queries']}
    for m in irm.iter_calc(list(metrics), data['qrels'], run):
        per_query[m.query_id][metrics[m.measure]] = float(m.value)
    for q, row in per_query.items():
        row['judged_at10'] = sum(d in data['qrels'][q] for d in rankings[q][:10])/10
    rows = list(per_query.values())
    summary = dict(dataset='wands', scope=scope, method=method, queries=len(rows))
    for key in [*metrics.values(), 'judged_at10']:
        summary[key] = float(np.mean([r[key] for r in rows]))
    return summary, rows


def paired_deltas(rows, seed):
    rng = np.random.default_rng(seed)
    out = []
    for scope in sorted({r['scope'] for r in rows}):
        methods = sorted({r['method'] for r in rows if r['scope'] == scope})
        values = {m: {r['query_id']: r['ndcg10'] for r in rows if r['scope'] == scope and r['method'] == m} for m in methods}
        for a, b in itertools.combinations(methods, 2):
            if values[a].keys() != values[b].keys():
                raise ValueError('Unpaired query sets')
            delta = np.array([values[a][q]-values[b][q] for q in sorted(values[a])])
            lo, hi = np.quantile(rng.choice(delta, (5000, len(delta))).mean(axis=1), [.025, .975])
            out.append(dict(scope=scope, method_a=a, method_b=b, paired_queries=len(delta),
                            mean_a_minus_b=float(delta.mean()), lo=float(lo), hi=float(hi)))
    return out


def run(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    data, audit = load_data(root, args)
    if args.phase == 'data':
        save(root / 'data_audit.json', audit)
        print(json.dumps({k: v for k, v in audit.items() if k != 'conflicting_pairs'}), flush=True)
        return
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime or explicitly use --device cpu')
    torch.manual_seed(args.seed)
    os.environ['PYLATE_SCORES_BACKEND'] = 'torch'
    if args.phase == 'smoke':
        # Exercise real model downloads/encoding/scoring before a long comparison.
        root = root / 'smoke'
        root.mkdir(exist_ok=True)
        qids, ids = list(data['queries'])[:2], list(data['corpus'])[:4]
        data = dict(corpus={d: data['corpus'][d] for d in ids},
                    queries={q: data['queries'][q] for q in qids},
                    qrels={q: data['qrels'][q] for q in qids})
    versions = {p: importlib.metadata.version(p) for p in
                ['torch', 'pylate', 'sentence-transformers', 'transformers', 'rank-bm25', 'ir-measures']}
    lock_manifest(root, dict(models=MODELS, versions=versions, dataset_sha256=digest(data),
        script_sha256=sha_file(__file__), shared_helpers_sha256=sha_file(Path(__file__).with_name('upstream_nanobeir.py')),
        seed=args.seed, max_length=args.max_length, batch_size=args.batch_size, shard_size=args.shard_size,
        pool_size=args.pool_size, text_mode=args.text_mode, device=args.device, phase=args.phase))
    save(root / 'data_audit.json', audit)
    save(root / 'dataset.json', data)
    save(root / 'scope.json', dict(
        labels='Human WANDS; Exact=2, Partial=1, Irrelevant=0; linear nDCG gains',
        unjudged='Scored as zero by standard metrics, separately reported with judged_at10; never removed',
        denominators='Full-corpus judgments for every scope; exact-only RR/P/Recall use grade>=2',
        corpus='Every product retained; no category filter or label-based candidate injection',
        shared_pool='Union of dense top-N and BM25 top-N; up to 2N candidates; IDs saved',
        scoring='ST normalized dot; PyLate native MaxSim with equal-length document groups; FP32',
        text='Identical source text, max_length cap; CE cap applies to entire query-document pair',
        training='No WANDS training; DenseOn/LateOn paired published recipe; CE separately trained',
        timing='Offline job only; no serving latency, build-time or memory comparison',
        retention='Full neural scores cached in corpus shards; top-1000 full rankings and all pool rankings exported'))
    lexical = lexical_scores(data)
    dense = neural_scores('dense', data, root, args)
    pools = make_pools(data, dense, lexical, args.pool_size)
    save(root / 'candidate_ids.json', pools)
    colbert = neural_scores('colbert', data, root, args)
    ce = cross_scores(data, pools, root, args)
    if args.phase == 'smoke':
        save(root / 'complete.json', dict(status='passed', quality_metrics=False))
        print('REAL_MODEL_SMOKE_PASSED', flush=True)
        return
    ids = list(data['corpus'])
    positions = {d: i for i, d in enumerate(ids)}
    summaries, per_query = [], []
    for method, scores in [('dense_normalized_dot', dense), ('colbert_native', colbert), ('bm25', lexical)]:
        for scope in ['full_wands', 'shared_union_pool']:
            rankings = {}
            for i, q in enumerate(data['queries']):
                candidates = ids if scope == 'full_wands' else pools[q]
                values = scores[i] if scope == 'full_wands' else scores[i, [positions[d] for d in candidates]]
                rankings[q] = ranked_ids(values, candidates, 1000 if scope == 'full_wands' else None)
            save(root / f'{method}_{scope}_rankings.json', rankings)
            summary, rows = evaluate(data, rankings, scope, method)
            summaries.append(summary)
            per_query.extend(rows)
    save(root / 'cross_encoder_shared_union_pool_rankings.json', ce)
    summary, rows = evaluate(data, ce, 'shared_union_pool', 'cross_encoder')
    summaries.append(summary)
    per_query.extend(rows)
    table(root / 'quality.csv', summaries)
    table(root / 'per_query.csv', per_query)
    table(root / 'paired_ndcg.csv', paired_deltas(per_query, args.seed))
    coverage = []
    for q, pool in pools.items():
        exact = {d for d, grade in data['qrels'][q].items() if grade == 2}
        coverage.append(dict(query_id=q, candidates=len(pool), exact_relevant=len(exact),
            exact_candidate_coverage=len(exact & set(pool))/len(exact) if exact else None,
            judged_candidate_fraction=sum(d in data['qrels'][q] for d in pool)/len(pool)))
    table(root / 'pool_coverage.csv', coverage)
    save(root / 'complete.json', dict(status='complete', evaluated_queries=len(data['queries'])))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--data-dir', help='Optional directory of original WANDS TSV .csv files')
    p.add_argument('--phase', choices=['data', 'smoke', 'compare'], default='compare')
    p.add_argument('--queries', type=int, default=0, help='0=all 480; otherwise seeded subset, full corpus retained')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--pool-size', type=int, default=100, help='Top N from EACH of dense and BM25')
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--shard-size', type=int, default=256)
    p.add_argument('--text-mode', choices=['full', 'title'], default='full')
    p.add_argument('--device', default='cuda')
    a = p.parse_args(argv)
    if a.queries < 0 or min(a.pool_size, a.max_length, a.batch_size, a.shard_size) <= 0:
        p.error('Counts must be positive; queries=0 means all queries')
    return a


if __name__ == '__main__':
    run(parse_args())
