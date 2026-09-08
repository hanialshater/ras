"""Upstream-first NanoBEIR correctness baseline; no RAS scoring or teacher labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import re
import time

import numpy as np

CARD = dict(zip(
    ['climatefever', 'dbpedia', 'fever', 'fiqa2018', 'hotpotqa', 'msmarco',
     'nfcorpus', 'nq', 'quoraretrieval', 'scidocs', 'arguana', 'scifact', 'touche2020'],
    [.4148, .7296, .9452, .5670, .9012, .7089, .3957, .7645, .9691, .3987,
     .5609, .8372, .5927]))
MODEL_IDS = {
    'colbert': ('lightonai/GTE-ModernColBERT-v1', '25f6f7bb8237b7ae25ae1d9b805ce17c0d1cc639'),
    'dense': ('Alibaba-NLP/gte-modernbert-base', 'e7f32e3c00f91d699e8c43b53106206bcc72bb22'),
    'ce': ('Alibaba-NLP/gte-reranker-modernbert-base', 'f7481e6055501a30fb19d090657df9ec1f79ab2c'),
}
CARD_URL = 'https://huggingface.co/lightonai/GTE-ModernColBERT-v1#nano-beir'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def table(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_data(name, root):
    """Same text and binary qrels preparation as PyLate's NanoBEIR loader, pinned."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from pylate.evaluation.nano_beir_evaluator import MAPPING_DATASET_NAME_TO_ID
    path = root / name / 'dataset.json'
    if path.exists():
        data = json.loads(path.read_text())
        fingerprint = data.pop('sha256')
        if digest(data) != fingerprint:
            raise ValueError(f'Dataset checksum mismatch: {path}')
        return data
    repo = MAPPING_DATASET_NAME_TO_ID[name]
    revision = HfApi().dataset_info(repo).sha
    parts = {part: load_dataset(repo, part, split='train', revision=revision)
             for part in ['corpus', 'queries', 'qrels']}
    corpus = {str(x['_id']): x['text'] for x in parts['corpus'] if x['text']}
    queries = {str(x['_id']): x['text'] for x in parts['queries'] if x['text']}
    qrels = {}
    for x in parts['qrels']:
        if float(x.get('score', 1)) <= 0:
            raise ValueError('Nonpositive qrel: native NanoBEIR binary semantics need review')
        qrels.setdefault(str(x['query-id']), set()).add(str(x['corpus-id']))
    # Preserve official corpus/text order, and the evaluator's query selection.
    queries = {qid: text for qid, text in queries.items() if qrels.get(qid)}
    if not queries or not corpus:
        raise ValueError('Empty dataset')
    if any(not qrels[qid] <= corpus.keys() for qid in queries):
        raise ValueError('Relevant documents missing from corpus')
    data = dict(repo=repo, revision=revision, corpus=corpus, queries=queries,
                qrels={q: sorted(qrels[q]) for q in queries})
    save(path, dict(data, sha256=digest(data)))
    return data


def evaluator(data, *, colbert=False, batch_size=16, chunk_size=128, all_scores=False):
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.util import cos_sim
    from pylate.evaluation import PyLateInformationRetrievalEvaluator
    from pylate.scores import colbert_scores
    cls = PyLateInformationRetrievalEvaluator if colbert else InformationRetrievalEvaluator
    kwargs = dict(queries=data['queries'], corpus=data['corpus'],
                  relevant_docs={q: set(d) for q, d in data['qrels'].items()},
                  batch_size=batch_size, corpus_chunk_size=chunk_size,
                  ndcg_at_k=[10], mrr_at_k=[10], map_at_k=[100],
                  accuracy_at_k=[1, 3, 5, 10], precision_recall_at_k=[1, 3, 5, 10],
                  write_csv=False, write_predictions=True,
                  score_functions={'MaxSim': colbert_scores} if colbert else {'cosine': cos_sim})
    if all_scores:
        # Retain every native score for subsequent identical-pool evaluation.
        # Extra recall cutoff does not alter nDCG@10/MRR@10/MAP@100.
        kwargs['precision_recall_at_k'].append(len(data['corpus']))
    return cls(**kwargs)


def native_retrieval(model, data, path, *, colbert, batch_size, chunk_size):
    path.mkdir(parents=True, exist_ok=True)
    ev = evaluator(data, colbert=colbert, batch_size=batch_size,
                   chunk_size=chunk_size, all_scores=True)
    with __import__('torch').inference_mode():
        metrics = ev(model, output_path=str(path))
    files = list(path.glob('*.jsonl'))
    if len(files) != 1:
        raise ValueError(f'Expected one upstream predictions file: {files}')
    predictions = [json.loads(line) for line in files[0].read_text().splitlines()]
    ranks = {str(x['query_id']): x['results'] for x in predictions}
    expected = set(data['corpus'])
    if set(ranks) != set(data['queries']):
        raise ValueError('Upstream query IDs differ')
    for rows in ranks.values():
        if len(rows) != len(expected) or {str(r['corpus_id']) for r in rows} != expected:
            raise ValueError('Upstream predictions do not contain the full corpus')
        if not all(np.isfinite(r['score']) for r in rows):
            raise ValueError('Nonfinite upstream scores')
    return metrics, ranks


def restrict(ranks, pools):
    out = {}
    if set(ranks) != set(pools):
        raise ValueError('Query sets differ')
    for q, rows in ranks.items():
        wanted = set(pools[q])
        if len(wanted) != len(pools[q]):
            raise ValueError('Duplicate pool IDs')
        out[q] = [r for r in rows if str(r['corpus_id']) in wanted]
        if {str(r['corpus_id']) for r in out[q]} != wanted:
            raise ValueError('Missing candidate scores')
    return out


def native_metrics(data, ranks):
    ev = evaluator(data)
    return ev.compute_metrics([ranks[q] for q in ev.queries_ids])


def evaluate_rows(name, method, scope, budget, data, ranks):
    metrics = native_metrics(data, ranks)
    summary = dict(dataset=name, method=method, scope=scope, candidate_budget=budget,
                   queries=len(data['queries']), ndcg10=metrics['ndcg@k'][10],
                   mrr10=metrics['mrr@k'][10], recall10=metrics['recall@k'][10])
    per_query = []
    for q in data['queries']:
        one = dict(data, queries={q: data['queries'][q]}, qrels={q: data['qrels'][q]})
        m = native_metrics(one, {q: ranks[q]})
        per_query.append(dict(dataset=name, query_id=q, method=method, scope=scope,
                              ndcg10=m['ndcg@k'][10]))
    return summary, per_query


def paired(rows, seed=7):
    output = []
    rng = np.random.default_rng(seed)
    for ds in sorted({r['dataset'] for r in rows}):
        values = {}
        for r in rows:
            if r['dataset'] == ds and r['scope'] == 'shared_dense_pool':
                values.setdefault(r['method'], {})[r['query_id']] = r['ndcg10']
        for a, b in itertools.combinations(sorted(values), 2):
            if values[a].keys() != values[b].keys():
                raise ValueError('Unpaired query sets')
            qids = sorted(values[a])
            delta = np.array([values[a][q] - values[b][q] for q in qids])
            boot = rng.choice(delta, (5000, len(delta))).mean(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            output.append(dict(dataset=ds, method_a=a, method_b=b, paired_queries=len(delta),
                               mean_a_minus_b=float(delta.mean()), lo=float(lo), hi=float(hi)))
    return output


def bm25(data):
    from rank_bm25 import BM25Okapi
    tokenize = lambda text: re.findall(r'\w+', text.lower())
    ids = list(data['corpus'])
    index = BM25Okapi([tokenize(data['corpus'][d]) for d in ids])
    return {q: sorted([dict(corpus_id=d, score=float(s)) for d, s in
                       zip(ids, index.get_scores(tokenize(text)))],
                      key=lambda x: (-x['score'], x['corpus_id']))
            for q, text in data['queries'].items()}


def model_load(method, device):
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from pylate import models
    repo, revision = MODEL_IDS[method]
    path = snapshot_download(repo, revision=revision)
    kwargs = dict(device=device, model_kwargs={'torch_dtype': torch.float32,
                  'attn_implementation': 'sdpa'}, config_kwargs={'reference_compile': False})
    if method == 'colbert':
        model = models.ColBERT(path, **kwargs)  # Native checkpoint lengths, markers, expansion.
    elif method == 'dense':
        model = SentenceTransformer(path, **kwargs)
    else:
        model = CrossEncoder(path, max_length=512, **kwargs)
    model.eval()
    meta = dict(repo=repo, revision=revision, parameters=sum(p.numel() for p in model.parameters()),
                device=str(model.device), dtype=str(next(model.parameters()).dtype))
    for key in ['query_length', 'document_length', 'do_query_expansion', 'max_seq_length', 'max_length']:
        if hasattr(model, key):
            meta[key] = getattr(model, key)
    return model, meta


def run(args):
    import torch
    os.environ['PYLATE_SCORES_BACKEND'] = 'torch'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime, or explicitly use --device cpu')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    versions = {p: importlib.metadata.version(p) for p in
                ['torch', 'pylate', 'sentence-transformers', 'transformers', 'datasets', 'rank-bm25']}
    config = dict(models=MODEL_IDS, versions=versions, seed=args.seed, device=args.device,
                  chunk_size=args.chunk_size, batch_size=args.batch_size,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest = root / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Configuration changed: use a new output directory')
    save(manifest, config)
    save(root / 'scope.json', dict(
        labels='Official NanoBEIR binary qrels; same semantics as PyLate NanoBEIREvaluator',
        metric_denominator='All relevant documents in each NanoBEIR corpus, also for shared pools',
        text='Official NanoBEIR text fields; no metadata filters, CLIP labels or RAS code',
        precision='FP32; SDPA; PyLate torch scoring backend',
        lengths='Checkpoint defaults; CE pair limit 512. Same source text, differing token limits.',
        timings='Offline batch job durations only; NOT serving latency or QPS',
        supervision_matching=False, card_source=CARD_URL,
        reference='Per-dataset upstream PyLate evaluator, as used by NanoBEIREvaluator; binary qrels',
        caveat='Model card lacks a complete historical environment lock. A discrepancy is reported, not hidden.',
        padding='Upstream PyLate padded scoring semantics retained; not the old ragged NumPy scorer.'))
    data_by_name = {name: load_data(name, root) for name in args.datasets}
    if args.phase == 'reference':
        model, meta = model_load('colbert', args.device)
        save(root / 'colbert_model.json', meta)
        checks = []
        for name, data in data_by_name.items():
            out = root / name / 'reference'
            cache = out / 'complete.json'
            if cache.exists():
                result = json.loads(cache.read_text())
            else:
                print(f'REFERENCE_START {name}', flush=True)
                start = time.perf_counter()
                metrics, _ = native_retrieval(model, data, out, colbert=True,
                    batch_size=args.batch_size, chunk_size=args.chunk_size)
                observed = float(metrics['MaxSim_ndcg@10'])
                result = dict(dataset=name, observed=observed, published=CARD[name],
                    difference=observed-CARD[name], queries=len(data['queries']),
                    within_0_01=abs(observed-CARD[name]) <= .01,
                    offline_job_seconds=time.perf_counter()-start, dataset_sha256=digest(data))
                save(out / 'metrics.json', metrics)
                save(cache, result)
            if result['dataset_sha256'] != digest(data):
                raise ValueError('Reference dataset checksum mismatch')
            checks.append(result)
            table(root / 'reference_check.csv', checks)
            print(json.dumps(result), flush=True)
        save(root / 'reference_summary.json', dict(
            datasets=args.datasets, all_13=set(args.datasets) == set(CARD),
            observed_mean=float(np.mean([r['observed'] for r in checks])),
            published_subset_mean=float(np.mean([CARD[n] for n in args.datasets])),
            published_all_13_mean=.6758,
            within_0_01_all_datasets=all(r['within_0_01'] for r in checks),
            tolerance_note='0.01 is our investigation trigger, not a statistical equivalence test'))
        return

    missing = [n for n in args.datasets if not (root / n / 'reference/complete.json').exists()]
    if missing:
        raise ValueError(f'Run --phase reference first: {missing}')
    # Fail before expensive comparison if the reference is far from the card.
    failures = [n for n in args.datasets if not json.loads(
        (root / n / 'reference/complete.json').read_text())['within_0_01']]
    if failures and not args.allow_reference_mismatch:
        raise ValueError(f'Reference differs by >0.01 on {failures}. Inspect reference_check.csv. '
                         'Use --allow-reference-mismatch only for a labeled diagnostic comparison.')
    dense, dense_meta = model_load('dense', args.device)
    ce, ce_meta = model_load('ce', args.device)
    dest = root / f'comparison_pool{args.pool_size}'
    dest.mkdir(exist_ok=True)
    save(dest / 'models.json', dict(dense=dense_meta, ce=ce_meta, reference_mismatch=failures))
    summaries, per_query, coverage = [], [], []
    for name, data in data_by_name.items():
        out = dest / name
        cache = out / 'complete.json'
        if cache.exists():
            result = json.loads(cache.read_text())
        else:
            print(f'COMPARE_START {name}', flush=True)
            start = time.perf_counter()
            _, dense_ranks = native_retrieval(dense, data, out / 'dense', colbert=False,
                batch_size=args.batch_size, chunk_size=args.chunk_size)
            # Upstream cosine is exactly dot product of L2-normalized vectors.
            ref_file, = (root / name / 'reference').glob('*.jsonl')
            colbert_ranks = {str(x['query_id']): x['results'] for x in
                map(json.loads, ref_file.read_text().splitlines())}
            pools = {q: [str(r['corpus_id']) for r in rows[:args.pool_size]]
                     for q, rows in dense_ranks.items()}
            save(out / 'candidate_ids.json', pools)
            all_ranks = {'dense_normalized_dot': dense_ranks,
                         'colbert_native': colbert_ranks, 'bm25': bm25(data)}
            ds_summary, ds_queries, ds_coverage = [], [], []
            for method, ranks in all_ranks.items():
                row, pq = evaluate_rows(name, method, 'full_nanocorpus', 0, data, ranks)
                ds_summary.append(row)
                ds_queries.extend(pq)
            pool_ranks = {m: restrict(r, pools) for m, r in all_ranks.items()}
            ce_ranks = {}
            for q, ids in pools.items():
                pairs = [(data['queries'][q], data['corpus'][d]) for d in ids]
                values = np.asarray(ce.predict(pairs, batch_size=args.batch_size,
                    show_progress_bar=False, activation_fn=torch.nn.Identity())).reshape(-1)
                if len(values) != len(ids) or not np.isfinite(values).all():
                    raise ValueError('CE must emit one finite score per pair')
                ce_ranks[q] = [dict(corpus_id=d, score=float(s)) for d, s in zip(ids, values)]
                relevant = set(data['qrels'][q])
                ds_coverage.append(dict(dataset=name, query_id=q, pool_count=len(ids),
                    full_relevant=len(relevant), pool_relevant=len(relevant & set(ids)),
                    relevant_coverage=len(relevant & set(ids))/len(relevant)))
            pool_ranks['cross_encoder'] = ce_ranks
            for method, ranks in pool_ranks.items():
                if any({r['corpus_id'] for r in ranks[q]} != set(pools[q]) for q in pools):
                    raise ValueError('Methods did not score identical candidates')
                save(out / f'{method}_shared_scores.json', ranks)
                row, pq = evaluate_rows(name, method, 'shared_dense_pool', args.pool_size, data, ranks)
                ds_summary.append(row)
                ds_queries.extend(pq)
            result = dict(summary=ds_summary, per_query=ds_queries, coverage=ds_coverage,
                          dataset_sha256=digest(data), offline_job_seconds=time.perf_counter()-start)
            save(cache, result)
        if result['dataset_sha256'] != digest(data):
            raise ValueError('Comparison dataset checksum mismatch')
        summaries.extend(result['summary'])
        per_query.extend(result['per_query'])
        coverage.extend(result['coverage'])
        table(dest / 'quality.csv', summaries)
        table(dest / 'per_query.csv', per_query)
        table(dest / 'pool_coverage.csv', coverage)
        table(dest / 'paired_ndcg.csv', paired(per_query, args.seed))
        print(f'COMPARE_DONE {name}', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['reference', 'compare'], required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--datasets', nargs='+', choices=list(CARD), default=list(CARD))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--pool-size', type=int, default=100)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--allow-reference-mismatch', action='store_true')
    args = p.parse_args()
    if min(args.batch_size, args.chunk_size, args.pool_size) < 1 or len(set(args.datasets)) != len(args.datasets):
        p.error('Sizes must be positive and datasets unique')
    return args


if __name__ == '__main__':
    run(parse_args())
