import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ras import predicate_study as p
from ras.binary import build_centered_binary_code, score_compiled_linear
from ras.late_interaction import RaggedEmbeddings
from experiments import wands_predicates as study
from experiments import wands_comparison as w


def test_metadata_unknowns_conflicts_and_redaction():
    rows=[dict(product_id=str(i),product_name='chair',product_description='comfortable seating',
        product_class='Chairs',product_features=text) for i,text in enumerate([
        'upholstered : yes|dsprimaryproductstyle : modern',
        'upholstered:no|outdooruse:unavailable|dsprimaryproductstyle:unknown',
        'upholstered:yes|upholstered:no', '',
        'upholstered:yes|upholstered:yes'])]
    y,audit=p.labels_from_products(rows)
    c=[v[0] for v in p.CONCEPTS].index('upholstered')
    assert y[:,c].tolist()==[1,0,-1,-1,1]
    assert y[1,5]==-1
    assert y[1,0]==-1
    assert audit[c]['conflicts']==1
    text=p.text_for_product(rows[0],'redacted')
    assert 'upholstered' not in text and 'product_class' not in text and 'comfortable' in text


def test_group_splits_and_plan_generation_do_not_use_test_labels():
    rows=[dict(product_id=str(i),product_name=f'chair {i//2}') for i in range(100)]
    split,groups=p.group_split(rows,7)
    assert all(len(set(split[np.array(groups)==g]))==1 for g in set(groups))
    y=np.array([[i%2,(i//2)%2,(i//3)%2] for i in range(100)],dtype=np.int8)
    specs=[('a','a','yes','A'),('b','b','yes','B'),('c','c','yes','C')]
    before=p.make_plans(y[split==0],specs,7,min_positive=2)
    eligible=p.eligible_concepts(y,split,min_fit=2,min_cal=1)
    y[split==2]=-1
    assert before==p.make_plans(y[split==0],specs,7,min_positive=2)
    assert eligible==p.eligible_concepts(y,split,min_fit=2,min_cal=1)


def test_packed_reference_matches_existing_binary_compiler():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(20,13)).astype('float32')
    code=build_centered_binary_code(x[:10],x[10:15],x[15:])
    packed,correction=p.binary_arrays(x,code.centroid)
    decoded=p.binary_reconstruct(packed,correction,code.centroid)
    head=dict(weights=rng.normal(size=(1,2,13)).astype('float32'),bias=np.array([[.3,-.4]],dtype='float32'),routes=np.zeros(20,dtype=int))
    quant,planes,scalars=p.quantized_head(head)
    for int4,hh in [(False,head),(True,quant)]:
        for j in range(2):
            expected=score_compiled_linear(np.unpackbits(packed,axis=1,count=13,bitorder='little'),correction,
                code,head['weights'][0,j],head['bias'][0,j],int4_query=int4)
            np.testing.assert_allclose(p.score_linear(decoded,hh)[:,j],expected,atol=1e-5)
    assert planes.dtype==np.uint8 and planes.shape==(1,2,4,2)


def test_oracle_retrieval_and_budget_gaps_with_unknowns():
    y=np.array([[1],[1],[1],[0],[-1]],dtype=np.int8)
    plans=[dict(name='a',columns=[0],signs=[1],text='a')]
    dense=np.array([[5.,4.,1.,3.,100.]])
    controls=p.pools_and_truth(y,np.arange(5),plans,dense,3,1,list('abcde'))
    assert controls[0]['known_universe']==4
    assert controls[0]['candidate_coverage']==pytest.approx(2/3)
    assert controls[0]['oracle_recall']==pytest.approx(1/3)
    # Ranking misses all positives in the same pool; gaps must sum to 1-recall.
    rows=p.ranking_metrics(np.array([[0.],[0.],[0.],[2.],[9.]]),controls,'probe',1,list('abcde'))
    row=rows[0]
    assert row['recall']==0
    assert row['coverage_loss']+row['budget_loss']+row['oracle_minus_method']==pytest.approx(1-row['recall'])


def test_negative_token_max_has_no_padding_matches():
    torch=pytest.importorskip('torch')
    tokens=RaggedEmbeddings.from_list([np.array([[-2.,-3.]],dtype='float32'),np.array([[-4.,-1.],[-2.,-5.]],dtype='float32')])
    head=dict(weights=np.ones((1,1,2),dtype='float32'),bias=np.zeros((1,1),dtype='float32'),routes=np.zeros(2,dtype=int))
    np.testing.assert_allclose(p.score_tokens(tokens,head),[[-5.],[-5.]])


def test_local_fallback_is_global_for_unsupported_cluster_labels():
    rng=np.random.default_rng(2)
    x=np.r_[rng.normal(-5,.1,(20,4)),rng.normal(5,.1,(20,4))].astype('float32')
    y=np.r_[np.zeros(20),np.ones(20)].astype('int8')[:,None]
    global_head=p.fit_linear(x,y,np.arange(40))
    local=p.fit_linear(x,y,np.arange(40),clusters=2)
    assert local['fallback'].all()
    np.testing.assert_allclose(p.score_linear(x,local),p.score_linear(x,global_head),atol=1e-5)


def test_cached_end_to_end(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    torch.set_num_threads(1)
    n=100
    rows=[]
    for i in range(n):
        style=['modern','traditional','industrial','glam'][i%4]
        features=f'dsprimaryproductstyle:{style}|'+ '|'.join(f'{c[1]}:'+('yes' if (i//(j+1))%2 else 'no') for j,c in enumerate(p.CONCEPTS[4:]))
        rows.append(dict(product_id=str(i),product_name=f'item {i}',product_class='Furniture',
                         product_description=f'usable object {i}',product_features=features))
    source=tmp_path/'product.csv'; pd.DataFrame(rows).to_csv(source,sep='\t',index=False)
    args=study.parse_args(['--data-dir',str(tmp_path/'data'),'--embedding-dir',str(tmp_path/'emb'),
        '--output-dir',str(tmp_path/'run'),'--products',str(source),'--max-products','100',
        '--device','cpu','--epochs','1','--clusters','2','--min-fit','2','--min-cal','1',
        '--pairs','2','--k','3','--pool-size','8'])
    study.prepare(args)
    data=json.loads((tmp_path/'data/dataset.json').read_text())
    rng=np.random.default_rng(22)
    for arm in args.models:
        folder=study.embedding_folder(args,arm)
        if arm=='colbert':
            tokens=RaggedEmbeddings.from_list([rng.normal(size=(i%4+1,8)).astype('float32') for i in range(n)])
            w.array_save(folder/'values.npy',tokens.values); w.array_save(folder/'offsets.npy',tokens.offsets)
            means=np.stack([t.mean(0) for t in tokens])
            w.array_save(folder/'mean_raw.npy',means); w.array_save(folder/'pooled.npy',means/np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-12))
        else:
            x=rng.normal(size=(n,8)).astype('float32'); x/=np.linalg.norm(x,axis=1,keepdims=True)
            w.array_save(folder/'values.npy',x)
        w.save(folder/'complete.json',dict(status='complete'))
    class Toy:
        def encode_query(self,texts,**kw):
            return np.ones((len(texts),8),dtype='float32')/np.sqrt(8)
    monkeypatch.setattr(study,'load_encoder',lambda *a:(Toy(),{}))
    study.study(args)
    report=tmp_path/'run'
    assert json.loads((report/'complete.json').read_text())['status']=='complete'
    gaps=pd.read_csv(report/'gap_queries.csv').dropna()
    sums=gaps.coverage_loss+gaps.budget_loss+gaps.oracle_minus_fp32+gaps.fp32_minus_binary1+gaps.binary1_minus_int4
    np.testing.assert_allclose(sums,1-gaps.binary1_int4_recall,atol=1e-6)
    assert len(pd.read_csv(report/'summary.csv'))==25  # 24 fitted controls + dense query baseline
    def forbidden(*a,**kw): raise AssertionError('Should reuse fitted heads')
    monkeypatch.setattr(p,'fit_linear',forbidden);monkeypatch.setattr(p,'fit_token_head',forbidden)
    study.study(args)


def test_resume_after_token_arrays_before_pooled_control(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    data=tmp_path/'data'; data.mkdir()
    w.save(data/'dataset.json',dict(ids=['a','b'],texts=['chair','table']))
    args=study.parse_args(['--data-dir',str(data),'--embedding-dir',str(tmp_path/'emb'),
        '--output-dir',str(tmp_path/'out'),'--models','colbert','--device','cpu'])
    folder=study.embedding_folder(args,'colbert')
    w.array_save(folder/'values.npy',np.array([[1.,0.],[0.,1.],[1.,0.]],dtype='float32'))
    w.array_save(folder/'offsets.npy',np.array([0,1,3]))
    def forbidden(*a,**kw): raise AssertionError('Should finish pooling without loading an encoder')
    monkeypatch.setattr(study,'load_encoder',forbidden)
    study.encode(args)
    np.testing.assert_allclose(np.load(folder/'mean_raw.npy'),[[1.,0.],[.5,.5]])
    assert (folder/'complete.json').exists()
    study.encode(args)
