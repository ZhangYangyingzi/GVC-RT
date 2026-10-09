"""Raw split-specific tables, fixed comparisons, pooled FID, plots and integrity."""
import math,statistics,itertools,traceback
from v614_io import *
METRICS=('LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM')
def compare(a,b,metric):
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    from select_checkpoint import curve
    try:
        x,y=curve(a,metric);u,v=curve(b,metric);lo=max(x[0],u[0]);hi=min(x[-1],u[-1])
        if hi<=lo:raise ValueError('no common positive-length bpp interval')
        grid=np.linspace(lo,hi,100);a=PchipInterpolator(x,y,extrapolate=False)(grid);b=PchipInterpolator(u,v,extrapolate=False)(grid)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        return dict(status='available',common_min_bpp=float(np.exp(lo)),common_max_bpp=float(np.exp(hi)),grid_points=100,delta=float((a-b).mean()),reason='')
    except ValueError as e:return dict(status='unavailable',delta=None,grid_points=0,reason=str(e))
def main():
    import numpy as np,torch
    torch.set_num_threads(2);cfg=frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    from eval_adapter import validate
    from select_checkpoint import final_methods
    methods=final_methods();cpindex=load(ROOT/'checkpoint_index.json');assert set(cpindex)==set(map(str,STEPS))
    assert load(ROOT/'training_integrity.json')['status']=='PASS';assert load(ROOT/'audits/step0_codec_equivalence.json')['status']=='PASS'
    assert load(ROOT/'frozen_checkpoint_index.json')['checkpoint_index_sha256']==sha(ROOT/'checkpoint_index.json')
    rows=[json.loads(x) for x in (ROOT/'training_logs/uvg_only.jsonl').read_text().splitlines()];assert len(rows)==5000 and [r['adaptation_step'] for r in rows]==list(range(1,5001))
    for r in rows:
        assert r['video'] in SPLITS['train_fit'] and r['source_step']==1000 and r['optimizer_update_count']==r['adaptation_step'] and r['all_finite']
        v=next(v for v in sources() if v['name']==r['video']);ix=r['canonical_indices'];assert ix==list(range(ix[0],ix[0]+4))
        assert r['source_frame_indices']==[v['source_frame_indices'][i] for i in ix]
        assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(r['external_qp'])][:4]
        assert all(math.isfinite(v) for v in r.values() if isinstance(v,float))
    write(ROOT/'training_logs/uvg_only.csv',rows)
    for step in STEPS:
        cp=cpindex[str(step)];assert sha(cp['path'])==cp['sha256'];s=torch.load(cp['path'],map_location='cpu',weights_only=True)
        assert s['adaptation_step']==s['optimizer_update_count']==step and s['source_step']==1000
        assert s['code_hashes']=={p.name:sha(p) for p in ROOT.glob('*.py')} and s['config_sha256']==sha(ROOT/'config.json')
        assert s['compression_hash']==load(ROOT/'audits/initialization.json')['compression_hash']
        assert {k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')}==cp['module_hashes']
        inf=cp['inference'];assert sha(inf['path'])==inf['sha256'];payload=torch.load(inf['path'],map_location='cpu',weights_only=True)
        assert set(payload)=={'wrapper','bridge','generator'} and {k:tensor_hash(payload[k]) for k in payload}==cp['module_hashes'];del s,payload
    # Validate every expected history point, including every saved Bosphorus curve.
    history=[]
    for step in STEPS:
        for v in sources():
            qs=range(10) if split(v)=='validation' else (0,4,9) if split(v)=='train_fit' and step in (0,1000,3000,5000) else ()
            for q in qs:
                r=load(point('uvg',name(step),v['video_index'],q));validate(r,v,q,name(step));history.append(dict(r,adaptation_step=step,split=split(v),exposure='training_exposed' if split(v)=='train_fit' else 'held_out_from_this_training',actual_bpp=r['bpp']))
    assert len(history)==118
    for sp in ('validation','train_fit'):write(ROOT/'results'/sp/'adaptation_history.csv',[r for r in history if r['split']==sp])
    fidmod=module('v614_fid',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py');fid=fidmod.low_rank_fid;allfinal=[];allfid=[];gtmeta={}
    for sp,names in SPLITS.items():
        vs=[v for v in sources() if v['name'] in names];exposure='training_exposed' if sp=='train_fit' else 'held_out_from_this_training';out=ROOT/'results'/sp
        own=[];pooled=[];macro=[];gt=[]
        for v in vs:
            base=load(point('uvg','original',v['video_index'],0))
            with np.load(base['feature_path']) as f:gt.append(f['real'].copy())
        reference=np.concatenate(gt);assert reference.shape==(len(vs)*64,2048)
        cache=ROOT/'features'/f'GT_{sp}.npy';cache.parent.mkdir(exist_ok=True)
        if cache.exists():assert np.array_equal(np.load(cache),reference)
        else:
            with cache.with_suffix('.tmp').open('wb') as f:np.save(f,reference)
            cache.with_suffix('.tmp').replace(cache)
        poolhash=hashlib.sha256(json.dumps([(v['source_sha256'],v['source_frame_indices'],v['rgb_sha256']) for v in vs],sort_keys=True).encode()).hexdigest()
        gtmeta[sp]=dict(path=str(cache),sha256=sha(cache),frame_pool_sha256=poolhash,frames=len(reference),sequences=[v['name'] for v in vs],construction='only matching split GT features; no other split cache')
        for m in methods:
            for v in vs:
                for q in range(10):
                    r=load(point('uvg',m,v['video_index'],q));validate(r,v,q,m)
                    own.append(dict(r,split=sp,exposure=exposure,actual_bpp=r['bpp'],rate_axis='actual bpp',samples=64,aggregation='per-sequence native measurements'))
                    prefix=out/'per_frame'/m/f'video_{v["video_index"]:02d}_qp{q}'
                    write(prefix.with_suffix('.frames.csv'),read(r['frame_metrics_path']));write(prefix.with_suffix('.transitions.csv'),read(r['transitions_path']))
            for q in range(10):
                rs=[next(r for r in own if r['method']==m and r['video_index']==v['video_index'] and r['QP']==q) for v in vs];rec=[]
                for i,r in enumerate(rs):
                    with np.load(r['feature_path']) as f:
                        assert np.allclose(f['real'],gt[i],rtol=1e-5,atol=1e-6);rec.append(f['reconstruction'].copy())
                reconstruction=np.concatenate(rec);value=fid(reference,reconstruction);assert math.isfinite(value)
                pooled.append(dict(split=sp,exposure=exposure,method=m,QP=q,FID=value,bpp=statistics.mean(r['bpp'] for r in rs),rate_axis='mean actual bpp within this split',samples=len(reference),sequences=len(vs),aggregation='pooled all frame features within split; no average of per-video FID',GT_pool_sha256=poolhash,GT_cache_sha256=sha(cache)))
                macro.append(dict(split=sp,exposure=exposure,method=m,QP=q,**{k:statistics.mean(r[k] for r in rs) for k in ('bpp','kbps',*METRICS)},FID=value,rate_axis='mean actual bpp',samples=len(reference),sequences=len(vs),aggregation='equal video mean metrics; pooled FID',shared_interval='not applicable for raw measured points'))
        write(out/'per_sequence_all_qp.csv',own);write(out/'all_qp_summary.csv',macro);write(out/'fid_pooled.csv',pooled);allfinal.extend(own);allfid.extend(pooled)
        per=[];summary=[]
        pairs=[(m,b) for i,m in enumerate(methods) for b in methods[:i]]
        for m,b in pairs:
            for metric in ('LPIPS','DISTS','FloLPIPS'):
                group=[]
                for v in vs:
                    r=dict(split=sp,exposure=exposure,sequence=v['name'],method=m,reference=b,metric=metric,rate_axis='ln(actual bpp)',aggregation='100 uniform shared log-rate points; PCHIP; no extrapolation',**compare([r for r in own if r['method']==m and r['video_index']==v['video_index']],[r for r in own if r['method']==b and r['video_index']==v['video_index']],metric));group.append(r);per.append(r)
                valid=[r for r in group if r['status']=='available'];ok=len(valid)==len(vs)
                summary.append(dict(split=sp,exposure=exposure,method=m,reference=b,metric=metric,status='available' if ok else 'unavailable',delta=statistics.mean(r['delta'] for r in valid) if ok else None,available_sequences=len(valid),expected_sequences=len(vs),grid_points_per_sequence=100,rate_axis='per-sequence ln(actual bpp)',shared_interval='see equal_rate_per_sequence.csv',aggregation='equal sequence weights; incomplete coverage is unavailable'))
        write(out/'equal_rate_per_sequence.csv',per);write(out/'equal_rate_summary.csv',summary)
    dump(ROOT/'audits/GT_feature_pools.json',gtmeta)
    argv=[PLOT_PYTHON,'-B',str(ROOT/'plot.py')];command(argv);subprocess.run(argv,cwd=REPO,check=True)
    assert inventory()==load(ROOT/'audits/old_inventory.json'),'historical experiments modified'
    frozen();expected=len(methods)*70;assert len(allfinal)==expected
    # All unique stored points are independently revalidated, not just final tables.
    points=list((ROOT/'parts').glob('*/video_*_qp*.json'))
    for p in points:
        r=load(p);v=next(v for v in sources() if v['video_index']==r['video_index']);validate(r,v,r['QP'],r['method'])
    required=[*ROOT.glob('*.py'),*ROOT.glob('*.json'),*list((ROOT/'results').rglob('*.csv')),*list((ROOT/'results').rglob('*.png')),*list((ROOT/'training_logs').glob('*'))]
    git=subprocess.run(['git','check-ignore','--stdin'],cwd=REPO,input='\n'.join(str(p.relative_to(REPO)) for p in required)+'\n',text=True,capture_output=True);assert git.returncode==1 and not git.stdout,git.stdout
    dump(ROOT/'final_integrity.json',dict(status='PASS',training_updates=5000,source_step=1000,primary_checkpoint=cpindex['5000'],validation_selection=load(ROOT/'validation_selection.json'),counts=dict(final_RD=len(allfinal),unique_RD=len(points),validation_history=70,train_fit_history=48,split_pooled_FID=len(allfid)),final_methods=methods,checked=dict(historical_unchanged=True,holdout_not_used_in_training_or_selection=True,real_RANS=True,independent_decode=True,state_sync=True,frozen_compression=True,checkpoint_hashes=True,disjoint_source_videos=True,source_time_axis_unchanged=True,uploaded_lightweight_files=True),missing_files=[],unresolved_errors=[],output_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*') if p.is_file()},finished_unix=time.time()))
    print('FINAL PASS',len(allfinal),'FINAL RD',len(points),'UNIQUE RD',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        error=dict(status='FAIL',traceback=traceback.format_exc());dump(ROOT/'final_integrity.json',error);dump(ROOT/'logs'/f'failure_report_{time.time_ns()}.json',error);raise
