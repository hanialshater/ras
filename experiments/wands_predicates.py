"""WANDS attribute representation/locality/compilation study, with resumable stages."""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import wands_comparison as w, wands_systems as s
from ras import predicate_study as p

MINILM = ('sentence-transformers/all-MiniLM-L6-v2', '1110a243fdf4706b3f48f1d95db1a4f5529b4d41')
MODELS = dict(minilm=MINILM, dense=w.MODELS['dense'], colbert=w.MODELS['colbert'])


def prepare(args):
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(args.products) if args.products else root/'product.csv'
    if not source.exists():
        if args.products: raise FileNotFoundError(source)
        print('DOWNLOAD WANDS product metadata', flush=True)
        url = f'https://raw.githubusercontent.com/wayfair/WANDS/{w.WANDS_REVISION}/dataset/product.csv'
        with urllib.request.urlopen(url, timeout=120) as response, source.with_suffix('.tmp').open('wb') as out:
            __import__('shutil').copyfileobj(response, out)
        source.with_suffix('.tmp').replace(source)
    frame = pd.read_csv(source, sep='\t', dtype=str, keep_default_na=False)
    if frame.product_id.duplicated().any(): raise ValueError('Duplicate product IDs')
    rows = frame.to_dict('records')
    # Fixed item subset independent of labels, held-out performance or model scores.
    ordered = sorted(rows, key=lambda r: w.digest(['predicate-pilot-v1', r['product_id']]))
    chosen = ordered[:args.max_products] if args.max_products else ordered
    texts = [p.text_for_product(r, args.view) for r in chosen]
    y, audit = p.labels_from_products(chosen)
    manifest = dict(source_sha256=w.sha_file(source), source_revision=w.WANDS_REVISION if not args.products else 'supplied',
        max_products=args.max_products, view=args.view, concepts=p.CONCEPTS,
        ids=[r['product_id'] for r in chosen], text_sha256=w.digest(texts), label_sha256=w.digest(y.tolist()))
    w.lock_manifest(root, manifest)
    w.save(root/'dataset.json', dict(ids=manifest['ids'], texts=texts,
        titles=[r['product_name'] for r in chosen], classes=[r['product_class'] for r in chosen]))
    w.array_save(root/'labels.npy', y)
    w.table(root/'label_audit.csv', audit)
    full_y, full_audit = p.labels_from_products(rows)
    w.table(root/'full_source_label_audit.csv', full_audit)
    examples = []
    for j, spec in enumerate(p.CONCEPTS):
        for value in [0, 1]:
            for i in np.flatnonzero(y[:, j] == value)[:3]:
                examples.append(dict(concept=spec[0], label=value, product_id=chosen[i]['product_id'],
                    title=chosen[i]['product_name'], source_field=spec[1], source_values=sorted(p.attributes(chosen[i]['product_features'])[spec[1]])))
    w.save(root/'label_examples.json', examples)
    w.save(root/'scope.json', dict(
        labels='Explicit WANDS product metadata; not CLIP, not independent human aesthetic labels',
        missing='Missing, conflicting or invalid binary values remain unknown; never negative',
        text='Titles and descriptions only; class and all structured feature fields omitted' if args.view == 'redacted' else
             'Full original text contains source label fields: metadata decoding control only',
        residual='Descriptions may naturally state the property; this is property recognition, not visual inference or held-out-concept generalization',
        sample='Fixed hash-selected products, independent of target values',
        scope='Full known-label held-out product universe per plan; missing-label products are outside the measured universe',
        task='Attribute predicates and synthetic compositions, separate from human WANDS query relevance'))
    print(pd.DataFrame(audit).to_string(index=False), flush=True)
    print(f'PREDICATE_DATA_READY items={len(chosen)} view={args.view}', flush=True)


def embedding_folder(args, arm):
    data = s.read(Path(args.data_dir)/'dataset.json')
    key = dict(schema=1, ids=data['ids'], text_sha256=w.digest(data['texts']), model=MODELS[arm],
        max_length=args.max_length, fp32=True, normalize=True,
        versions={name: importlib.metadata.version(name) for name in ['pylate', 'sentence-transformers', 'transformers']})
    folder = Path(args.embedding_dir)/w.digest(key)[:20]/arm
    folder.mkdir(parents=True, exist_ok=True)
    w.lock_manifest(folder, key)
    return folder


def load_encoder(arm, args):
    if arm != 'minilm': return w.load_model(arm, args.device, args.max_length)
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(MINILM[0], revision=MINILM[1], device=args.device,
        model_kwargs={'torch_dtype': torch.float32})
    model.max_seq_length = args.max_length
    model.eval()
    return model, dict(repo=MINILM[0], revision=MINILM[1], max_seq_length=model.max_seq_length,
                       parameters=sum(x.numel() for x in model.parameters()), dtype='torch.float32')


def encode(args):
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise ValueError('Select GPU runtime or --device cpu')
    data = s.read(Path(args.data_dir)/'dataset.json')
    for arm in args.models:
        folder = embedding_folder(args, arm)
        ready = ['values.npy'] if arm != 'colbert' else ['values.npy', 'offsets.npy', 'mean_raw.npy', 'pooled.npy']
        if (folder/'complete.json').exists() and all((folder/f).exists() for f in ready):
            print(f'REUSE_EMBEDDINGS {arm}', flush=True); continue
        if arm == 'colbert' and (folder/'values.npy').exists() and (folder/'offsets.npy').exists():
            # combine_tokens publishes offsets last, before the pooled controls.
            # Complete those controls after an interrupted run without re-encoding.
            finish_token_means(folder)
            w.save(folder/'complete.json', dict(status='complete', items=len(data['ids'])))
            for shard in folder.glob('shard_*.npz'): shard.unlink()
            print('RESUME_POOLED_CONTROL colbert', flush=True); continue
        model, meta = load_encoder(arm, args)
        w.save(folder/'model.json', meta)
        shards = []
        for start in range(0, len(data['texts']), 512):
            shard = folder/f'shard_{start:06d}.npz'
            if not shard.exists():
                chunk = data['texts'][start:start+512]
                with torch.inference_mode():
                    if arm == 'colbert':
                        encoded = model.encode(chunk, is_query=False, padding=False, normalize_embeddings=True,
                            convert_to_numpy=True, batch_size=args.encode_batch, show_progress_bar=False)
                        values = np.concatenate(encoded).astype('float32')
                        payload = dict(values=values, offsets=np.r_[0, np.cumsum([len(x) for x in encoded])])
                    else:
                        values = model.encode_document(chunk, normalize_embeddings=True, convert_to_numpy=True,
                            batch_size=args.encode_batch, show_progress_bar=False)
                        payload = dict(values=np.asarray(values, dtype='float32'))
                with shard.with_suffix('.tmp').open('wb') as out: np.savez(out, **payload)
                shard.with_suffix('.tmp').replace(shard)
                print(f'ENCODE {arm} {min(start+512,len(data["texts"]))}/{len(data["texts"])}', flush=True)
            shards.append(shard)
        if arm == 'colbert':
            s.combine_tokens(shards, folder)
            finish_token_means(folder)
        else:
            arrays = [np.load(shard)['values'] for shard in shards]
            w.array_save(folder/'values.npy', np.concatenate(arrays))
        w.save(folder/'complete.json', dict(status='complete', items=len(data['ids'])))
        # Kept only until the final arrays are safely published.
        for shard in shards: shard.unlink(missing_ok=True)
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def finish_token_means(folder):
    tokens = s.Tokens(folder)
    means = np.stack([tokens[i].mean(0) for i in range(len(tokens))])
    w.array_save(folder/'mean_raw.npy', means)
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    w.array_save(folder/'pooled.npy', means)


def save_head(folder, head):
    folder.mkdir(parents=True, exist_ok=True)
    if len(head['weights']) > 1:
        w.array_save(folder/'item_routes.npy', head['routes'].astype('uint8'))
    with (folder/'head.tmp').open('wb') as out:
        np.savez(out, **head)
    (folder/'head.tmp').replace(folder/'head.npz')


def load_head(folder):
    with np.load(folder/'head.npz') as f: return {k: f[k] for k in f.files}


def compact_items(folder, values, fit, tokens=None, means=None):
    folder.mkdir(parents=True, exist_ok=True)
    if tokens is None:
        centroid = values[fit].mean(0)
    else:
        lengths = np.diff(tokens.offsets)[fit]
        centroid = np.average(means[fit], axis=0, weights=lengths).astype('float32')
    if not (folder/'complete.json').exists():
        packed = np.lib.format.open_memmap(folder/'bits.npy', mode='w+', dtype='uint8',
            shape=(len(values), (values.shape[1]+7)//8))
        corrections = np.lib.format.open_memmap(folder/'corrections.npy', mode='w+', dtype='float32', shape=(len(values),2))
        for start in range(0,len(values),65536):
            packed[start:start+65536], corrections[start:start+65536] = p.binary_arrays(values[start:start+65536],centroid)
        packed.flush(); corrections.flush()
        w.array_save(folder/'centroid.npy', centroid)
        w.save(folder/'complete.json',dict(status='complete'))
    return np.load(folder/'bits.npy',mmap_mode='r'), np.load(folder/'corrections.npy',mmap_mode='r'), np.load(folder/'centroid.npy')


def study(args):
    import torch
    torch.set_num_threads(args.threads)
    if 'colbert' not in args.models or 'dense' not in args.models: raise ValueError('DenseOn and LateOn are required')
    data_root, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    data, labels = s.read(data_root/'dataset.json'), np.load(data_root/'labels.npy')
    folders = {arm:embedding_folder(args,arm) for arm in args.models}
    for arm, folder in folders.items():
        if not (folder/'complete.json').exists(): raise ValueError(f'Run encode first: {arm}')
    manifests = {arm:s.read(folder/'manifest.json') for arm,folder in folders.items()}
    w.lock_manifest(out, dict(data=s.read(data_root/'manifest.json'), embeddings=manifests,
        settings={k:v for k,v in vars(args).items() if k not in ['phase','products','data_dir','output_dir','embedding_dir']},
        sources={'runner':w.sha_file(__file__),'helpers':w.sha_file(p.__file__)}))
    tokens = s.Tokens(folders['colbert'])
    representations = {arm:np.load(folder/'values.npy',mmap_mode='r') for arm,folder in folders.items() if arm!='colbert'}
    representations['late_mean'] = np.load(folders['colbert']/'pooled.npy',mmap_mode='r')
    all_pred, all_rank, all_gap, costs = [],[],[],[]
    products = [dict(product_id=i,product_name=t) for i,t in zip(data['ids'],data['titles'])]
    for seed in args.seeds:
        root = out/f'seed_{seed}'; root.mkdir(exist_ok=True)
        split, groups = p.group_split(products,seed)
        fit,cal,test = [np.flatnonzero(split==i) for i in range(3)]
        keep = p.eligible_concepts(labels,split,args.min_fit,args.min_cal)
        if len(keep)<2: raise ValueError('Too few supported concepts; use more products')
        concepts = [p.CONCEPTS[c] for c in keep]; y=labels[:,keep]
        w.array_save(root/'split.npy',split)
        w.save(root/'split_audit.json',dict(seed=seed,fit=len(fit),calibration=len(cal),test=len(test),
            grouping='exact normalized product title; product-family IDs unavailable',
            concepts=[c[0] for c in concepts], excluded=[p.CONCEPTS[c][0] for c in range(len(p.CONCEPTS)) if c not in keep]))
        plans = p.make_plans(y[fit],concepts,seed,args.pairs,args.min_fit)
        w.save(root/'plans.json',plans)
        qpath=root/'dense_plan_queries.npy'
        if not qpath.exists():
            model,_=load_encoder('dense',args)
            queries=model.encode_query([x['text'] for x in plans],normalize_embeddings=True,convert_to_numpy=True,
                batch_size=args.encode_batch,show_progress_bar=False)
            w.array_save(qpath,queries); del model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        dense_scores=np.load(qpath) @ representations['dense'].T
        controls=p.pools_and_truth(y,test,plans,dense_scores,args.pool_size,args.k,data['ids'])
        w.save(root/'candidate_ids.json',{c['plan']['name']:[data['ids'][i] for i in c['pool']] for c in controls})
        seed_rank=p.ranking_metrics(dense_scores,controls,'dense_query',args.k,data['ids'],dense=True)
        seed_pred=[]; raw_heads={}; model_groups=[]
        for name,x in representations.items():
            compact=compact_items(root/f'items_{name}',x,fit)
            decoded=p.binary_reconstruct(*compact)
            for clusters in [0,*args.clusters]:
                label=f'{name}_'+('global' if not clusters else f'local{clusters}')
                folder=root/label
                if not (folder/'head.npz').exists():
                    print(f'FIT {label}',flush=True)
                    save_head(folder,p.fit_linear(x,y,fit,clusters,seed))
                head=load_head(folder); raw_heads[label]=head
                quant,planes,scalars=p.quantized_head(head)
                for suffix,values,hh in [('fp32',x,head),('binary1',decoded,head),('binary1_int4',decoded,quant)]:
                    method=label+'_'+suffix
                    rows=record_method(root,method,lambda values=values,hh=hh:p.score_linear(values,hh),
                        y,cal,test,concepts,controls,args,data['ids'])
                    seed_pred+=rows[0]; seed_rank+=rows[1]
                    if suffix=='binary1_int4':
                        w.array_save(folder/'int4_planes.npy',planes); w.array_save(folder/'int4_scalars.npy',scalars)
                    costs.append(cost_record(seed,method,x.shape[1]*4 if suffix=='fp32' else compact[0].shape[1]+8,
                        head,x.shape[1],suffix,len(data['ids']),tokens_per_item=1))
                model_groups.append(label)
            del decoded
        token_compact=compact_items(root/'items_late_tokens',tokens.values,fit,tokens=tokens,
            means=np.load(folders['colbert']/'mean_raw.npy'))
        for clusters in [0,*args.clusters]:
            label='late_tokens_'+('global' if not clusters else f'local{clusters}')
            folder=root/label
            routing=raw_heads['late_mean_'+('global' if not clusters else f'local{clusters}')]
            if not (folder/'head.npz').exists():
                print(f'FIT {label}',flush=True)
                init=None if not clusters else load_head(root/'late_tokens_global')
                head=p.fit_token_head(tokens,y,fit,routing['routes'],init=init,epochs=args.epochs,
                    seed=seed,device=args.device,batch_size=args.head_batch)
                head['centers']=routing['centers']
                save_head(folder,head)
            head=load_head(folder); quant,planes,scalars=p.quantized_head(head)
            for suffix,hh,compact in [('fp32',head,None),('binary1',head,token_compact),('binary1_int4',quant,token_compact)]:
                method=label+'_'+suffix
                rows=record_method(root,method,lambda hh=hh,compact=compact:p.score_tokens(tokens,hh,args.device,args.head_batch,compact),
                    y,cal,test,concepts,controls,args,data['ids'])
                seed_pred+=rows[0]; seed_rank+=rows[1]
                if suffix=='binary1_int4':
                    w.array_save(folder/'int4_planes.npy',planes); w.array_save(folder/'int4_scalars.npy',scalars)
                per_token=tokens.values.shape[1]*4 if suffix=='fp32' else token_compact[0].shape[1]+8
                costs.append(cost_record(seed,method,per_token*len(tokens.values)/len(tokens)+8,
                    head,tokens.values.shape[1],suffix,len(data['ids']),tokens_per_item=len(tokens.values)/len(tokens)))
            model_groups.append(label)
        for row in seed_rank: row['seed']=seed
        for row in seed_pred: row['seed']=seed
        all_pred+=seed_pred; all_rank+=seed_rank
        lookup={(r['method'],r['plan']):r for r in seed_rank}
        for label in model_groups:
            for plan in plans:
                fp=lookup[label+'_fp32',plan['name']]; bit=lookup[label+'_binary1',plan['name']]; q4=lookup[label+'_binary1_int4',plan['name']]
                all_gap.append(dict(seed=seed,method=label,plan=plan['name'],full_relevant=fp['full_relevant'],
                    coverage_loss=fp['coverage_loss'],budget_loss=fp['budget_loss'],
                    oracle_recall=fp['oracle_recall'],fp32_recall=fp['recall'],binary1_recall=bit['recall'],
                    binary1_int4_recall=q4['recall'],oracle_minus_fp32=fp['oracle_recall']-fp['recall'],
                    fp32_minus_binary1=fp['recall']-bit['recall'],binary1_minus_int4=bit['recall']-q4['recall']))
        w.table(root/'predicates.csv',seed_pred); w.table(root/'rankings.csv',seed_rank)
        w.save(root/'complete.json',dict(status='complete'))
        report(out,all_pred,all_rank,all_gap,costs)
    w.save(out/'complete.json',dict(status='complete',seeds=args.seeds))
    print('PREDICATE_STUDY_READY',flush=True)


def cost_record(seed,method,item_bytes,head,dim,suffix,n,tokens_per_item):
    experts,concepts,_=head['weights'].shape
    weight_bytes=dim*4 if suffix!='binary1_int4' else ((dim+7)//8)*4+8
    program=experts*concepts*(weight_bytes+4)+concepts*8 # biases and scalar calibration
    return dict(seed=seed,method=method,item_payload_bytes_mean=item_bytes+(1 if experts>1 else 0),
        items=n,token_vectors_per_item=tokens_per_item,program_payload_bytes=program,
        shared_encoder_bytes=dim*4 if suffix!='fp32' else 0,routing_centers_bytes=head['centers'].nbytes,
        fallback_heads=int(head['fallback'].sum()),head_count=experts*concepts,
        scope='Representation/program payload only; excludes ANN navigation, model, allocator and file headers; no serving timing')


def record_method(root,method,score_fn,y,cal,test,concepts,controls,args,ids):
    path=root/(method+'_raw.npy')
    if path.exists(): raw=np.load(path)
    else:
        tick=time.perf_counter(); raw=score_fn()
        if raw.shape!=y.shape or not np.isfinite(raw).all(): raise ValueError('Invalid predicate scores')
        w.array_save(path,raw)
        print(f'SCORE {method} {time.perf_counter()-tick:.1f}s',flush=True)
    scores,parameters,thresholds=p.calibrate(raw,y,cal)
    w.array_save(root/(method+'_calibration.npy'),parameters)
    w.array_save(root/(method+'_thresholds.npy'),thresholds)
    preds=p.predicate_metrics(scores,y,test,thresholds,concepts,method)
    ranks=p.ranking_metrics(scores,controls,method,args.k,ids)
    return preds,ranks


def report(out,predicates,rankings,gaps,costs):
    w.table(out/'predicate_metrics.csv',predicates); w.table(out/'composition_queries.csv',rankings)
    w.table(out/'gap_queries.csv',gaps); w.table(out/'payloads.csv',costs)
    pr=pd.DataFrame(predicates); rr=pd.DataFrame(rankings); gg=pd.DataFrame(gaps)
    summary=pr.groupby('method')[['ap','f1']].mean().join(rr.groupby('method')[['recall','oracle_recall','precision','candidate_coverage']].mean(), how='outer')
    summary['undefined_recall_plans']=rr.groupby('method').recall.apply(lambda v:v.isna().sum())
    summary.to_csv(out/'summary.csv')
    gg.groupby('method')[['coverage_loss','budget_loss','oracle_recall','fp32_recall','binary1_recall','binary1_int4_recall',
        'oracle_minus_fp32','fp32_minus_binary1','binary1_minus_int4']].mean().to_csv(out/'gap_summary.csv')
    print(summary.round(4).to_string(),flush=True)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['data-dir','embedding-dir','output-dir']: parser.add_argument('--'+name,required=True)
    parser.add_argument('--products')
    parser.add_argument('--phase',choices=['audit','encode','study','all'],default='all')
    parser.add_argument('--view',choices=['redacted','full'],default='redacted')
    parser.add_argument('--max-products',type=int,default=12000)
    parser.add_argument('--models',nargs='+',choices=list(MODELS),default=list(MODELS))
    parser.add_argument('--max-length',type=int,default=512)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--encode-batch',type=int,default=16)
    parser.add_argument('--head-batch',type=int,default=64)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--clusters',nargs='*',type=int,default=[8,16])
    parser.add_argument('--seeds',nargs='+',type=int,default=[7])
    parser.add_argument('--pairs',type=int,default=12)
    parser.add_argument('--min-fit',type=int,default=30)
    parser.add_argument('--min-cal',type=int,default=10)
    parser.add_argument('--pool-size',type=int,default=500)
    parser.add_argument('--k',type=int,default=50)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args(argv)
    if min([args.max_length,args.encode_batch,args.head_batch,args.epochs,args.min_fit,args.min_cal,args.pool_size,args.k,args.threads,*args.clusters])<1 or args.max_products<0 or args.pairs<0:
        parser.error('Invalid sizes')
    if args.k>args.pool_size or any(k>255 for k in args.clusters): parser.error('Require k<=pool size and <=255 clusters')
    return args


if __name__=='__main__':
    args=parse_args()
    with threadpool_limits(limits=args.threads):
        if args.phase in ['audit','all']: prepare(args)
        if args.phase in ['encode','all']: encode(args)
        if args.phase in ['study','all']: study(args)
