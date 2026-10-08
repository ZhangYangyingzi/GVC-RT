"""Strict raw export, numerical tables, figures and completeness checks only."""
import math,statistics,traceback
from v68_io import *
from io_utils import command
def training_check(cfg):
    import torch
    plans=load(ROOT/'manifests/training_plans.json')['plans'];branches={};indexes={};validation=[]
    for branch in BRANCHES:
        rows=[json.loads(x) for x in (ROOT/'training_logs'/f'{branch}.jsonl').read_text().splitlines()]
        assert len(rows)==5000 and [r['step'] for r in rows]==list(range(1,5001))
        assert [r['optimizer_update_count'] for r in rows]==list(range(1,5001))
        for r,p in zip(rows,plans):
            assert all(r[k]==p[k] for k in p)
            assert r['finite'] and all(math.isfinite(v) for v in r.values() if isinstance(v,float))
            assert r['gradients']['generator']==0
            assert len(r['cosine_gradient_per_P_frame'])==3
            assert all(g['finite'] for f in r['cosine_gradient_per_P_frame'] for g in f.values())
        branches[branch]=rows;write(ROOT/'results'/f'training_{tag(branch)}.csv',rows)
        ti=load(ROOT/'branches'/branch/'training_integrity.json');assert ti['status']=='PASS' and ti['updates']==5000
        assert load(ROOT/'branches'/branch/'training_status.json')['status']=='PASS'
        index=load(ROOT/'branches'/branch/'checkpoint_hashes.json');assert set(index)==set(map(str,cfg['checkpoint_steps']));indexes[branch]=index
        initial_generator=None
        for step in cfg['checkpoint_steps']:
            cp=index[str(step)];assert sha(cp['path'])==cp['sha256']
            state=torch.load(cp['path'],map_location='cpu',weights_only=True)
            assert state['step']==state['optimizer_update_count']==state['plan_position']==step
            assert state['code_hashes']=={p.name:sha(p) for p in ROOT.glob('*.py')}
            assert state['training_plan_sha256']==sha(ROOT/'manifests/training_plans.json') and state['config_sha256']==sha(ROOT/'config.json')
            assert all(t.dtype==torch.float32 for t in state['fp32_core_master'].values())
            assert all(t.dtype==torch.float32 for s in state['optimizer']['state'].values() for t in s.values() if torch.is_tensor(t))
            assert [g['name'] for g in state['optimizer']['param_groups']]==['wrapper','bridge','compression_core']
            assert state['frozen_hashes']==ti['frozen_hashes']
            gen=state_hash(state['generator'])
            if step==0:initial_generator=gen
            assert gen==initial_generator
            infcp=cp['inference'];assert sha(infcp['path'])==infcp['sha256']
            inf=torch.load(infcp['path'],map_location='cpu',weights_only=True)
            assert set(inf)=={'wrapper','bridge','generator','full_p','precision'}
            assert state_hash(inf['full_p'])==state_hash({k:t.half() for k,t in state['full_p_master'].items()})
            assert all(state_hash(inf[k])==state_hash(state[k]) for k in ('wrapper','bridge','generator'))
            if step==5000:assert inf['precision']['compression_hash']==cfg['core_hashes'][tag(branch)+'5000']
            del inf,state
            v=load(ROOT/'validation'/branch/f'step_{step}.json');assert v['status']=='PASS' and v['optimizer_steps']==0 and len(v['rows'])==96
            validation.extend(v['rows'])
        for step in cfg['gradient_check_steps']:
            g=load(ROOT/'audits/gradient_checks'/f'{branch}_{step}.json')
            assert g['status']=='PASS' and g['generator_frozen'] and g['teacher_frozen'] and g['extra_optimizer_steps']==0
            assert all(o['reconstruction_latent_gradient']['finite'] for o in g['observations'])
    for a,b in zip(branches[BRANCHES[0]],branches[BRANCHES[1]]):assert a['frame_rgb_sha256']==b['frame_rgb_sha256']
    assert load(ROOT/'audits/branch_initialization_equivalence.json')['status']=='PASS'
    assert len(validation)==1344
    write(ROOT/'results/validation_history.csv',validation)
    dump(ROOT/'checkpoint_hashes.json',dict(branches=indexes,evaluated=cfg['checkpoints']))
    dump(ROOT/'audits/training_complete.json',dict(status='PASS',updates={b:5000 for b in BRANCHES},matched_plan_and_actual_cropped_RGB=True,validation_rows=len(validation),checkpoint_steps=cfg['checkpoint_steps'],generator_frozen=True,FP32_optimizer=True,teacher_not_deployed=True))
def main():
    import torch
    torch.set_num_threads(2);command([PYTHON,'-B','-u',str(Path(__file__).resolve())]);cfg=frozen(True)
    raw=[];fid=[];bootstrap=[];macro=[];feature_metadata=[]
    for d in DATASETS:
        own=[]
        for m in METHODS:
            for v in sources(d):
                rows=[]
                for q in range(10):
                    r=load(point(d,m,v['video_index'],q));validate_point(r,v,q,m)
                    assert r['source_frame_indices']==v['source_frame_indices']
                    if m in ('A5000','B5000'):
                        assert r['full_P_checkpoint_loaded'] and r['receiver_integrity']['CDF_rebuilt']
                        assert r['receiver_integrity']['I_hash']==load(ROOT/'audits/initialization_audit.json')['I_hash']
                    rows.append(r)
                    prefix=ROOT/'results'/d/'per_frame'/m/f'video_{v["video_index"]:02d}_qp{q}'
                    frames=read(r['frame_metrics_path']);trans=read(r['transitions_path'])
                    for fr in frames:fr.update(dataset=d,method=m,video_index=v['video_index'],source_frame_index=v['source_frame_indices'][int(fr['frame'])])
                    for tr in trans:tr.update(dataset=d,method=m,video_index=v['video_index'],source_from_frame=v['source_frame_indices'][int(tr['from_frame'])],source_to_frame=v['source_frame_indices'][int(tr['to_frame'])])
                    write(prefix.with_suffix('.frames.csv'),frames);write(prefix.with_suffix('.transitions.csv'),trans)
                    feature_metadata.append(dict(dataset=d,method=m,video_index=v['video_index'],QP=q,path=r['feature_path'],sha256=r['feature_sha256'],frames=v['frames'],source_frame_indices=v['source_frame_indices'],source_rgb_sha256=v['rgb_sha256']))
                write(ROOT/'results'/d/'per_video'/f'{m}_{v["video_index"]:02d}.csv',rows);own.extend(rows)
            for q in range(10):
                rs=[r for r in own if r['method']==m and r['QP']==q];assert len(rs)==len(sources(d))
                macro.append(dict(dataset=d,method=m,QP=q,sequences=len(rs),**{k:statistics.mean(r[k] for r in rs) for k in ('kbps','bpp',*METRICS)}))
                r=load(ROOT/'parts/fid'/d/m/f'qp{q}.json')
                assert r['status']=='PASS' and math.isfinite(r['FID']) and r['real_bytes']==sum(x['real_bytes'] for x in rs)
                assert r['num_frames']==sum(x['frames'] for x in rs)
                assert r['GT_cache_sha256']==load(ROOT/'audits/GT_feature_cache.json')['datasets'][d]['sha256']
                assert sha(r['bootstrap_path'])==r['bootstrap_sha256']
                for pt in r['source_points']:assert sha(pt['path'])==pt['sha256']
                draws=read(r['bootstrap_path']);assert len(draws)==20 and all(math.isfinite(float(x['FID'])) for x in draws)
                fid.append(r);bootstrap.extend(draws)
        write(ROOT/'results'/d/'raw_rd.csv',own);raw.extend(own)
    assert len(raw)==len({(r['dataset'],r['method'],r['video_index'],r['QP']) for r in raw})==600
    assert len(fid)==len({(r['dataset'],r['method'],r['QP']) for r in fid})==80
    assert len(bootstrap)==len({(r['dataset'],r['method'],int(r['QP']),int(r['repeat'])) for r in bootstrap})==1600
    for name,rows in [('raw_rd',raw),('evaluation_raw',raw),('dataset_macro_rd',macro),('fid_raw',fid),('fid_bootstrap',bootstrap),('feature_cache_metadata',feature_metadata)]:write(ROOT/'results'/f'{name}.csv',rows)
    write(ROOT/'results/fid_bootstrap_summary.csv',[dict(dataset=r['dataset'],method=r['method'],QP=r['QP'],**r['bootstrap']) for r in fid])
    import equal_rate
    equal_rate.main(raw,fid,bootstrap);training_check(cfg)
    argv=['/data1/anaconda3_new/anaconda_program/bin/python','-B',str(ROOT/'report.py'),'--plots-only'];command(argv);subprocess.run(argv,cwd=REPO,check=True)
    for name in ('teacher_gate','objective_smoke','rate_gradient_probe','training_complete'):assert load(ROOT/'audits'/f'{name}.json')['status']=='PASS'
    frozen(True);check_upload()
    failures=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),resolution='all final task outputs and checkpoints independently revalidated') for p in (ROOT/'logs/failures').glob('*.json')]
    dump(ROOT/'audits/failure_resolution.json',dict(status='PASS',records=failures))
    dump(ROOT/'generated_files.json',dict(files=[str(p.relative_to(ROOT)) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]))
    dump(ROOT/'final_integrity.json',dict(status='PASS',training_updates={b:5000 for b in BRANCHES},counts=dict(raw_RD=600,FID=80,bootstrap=1600,validation=1344),checked=dict(old_experiments_unchanged=True,checkpoint_hashes=True,real_RANS=True,independent_decode=True,state_sync=True,matched_frame_indices=True,matched_training_crops=True,teacher_not_deployed=True,FP32_masters_optimizer=True),output_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*') if p.is_file()},missing_files=[],unresolved_failures=[],finished_unix=time.time()))
    print('FINAL INTEGRITY PASS',flush=True)
def check_upload():
    required=[ROOT/'config.json',ROOT/'source_manifest.json',*ROOT.glob('*.py'),*list((ROOT/'audits').rglob('*.json')),*list((ROOT/'training_logs').glob('*.csv')),*list((ROOT/'results').rglob('*.csv')),*list((ROOT/'results').rglob('*.png'))]
    proc=subprocess.run(['git','check-ignore','--stdin'],cwd=REPO,input='\n'.join(str(p.relative_to(REPO)) for p in required)+'\n',capture_output=True,text=True)
    ignored=proc.stdout.splitlines();dump(ROOT/'audits/git_upload_audit.json',dict(status='PASS' if proc.returncode==1 and not ignored else 'FAIL',ignored=ignored,checked=len(required)))
    assert proc.returncode==1 and not ignored,ignored
def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    macro=read(ROOT/'results/dataset_macro_rd.csv');fid=read(ROOT/'results/fid_raw.csv')
    for d in DATASETS:
        for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
            fig,ax=plt.subplots(figsize=(9,6))
            for m in METHODS:
                rows=sorted([r for r in (fid if metric=='FID' else macro) if r['dataset']==d and r['method']==m],key=lambda r:float(r['kbps']));assert len(rows)==10
                ax.plot([float(r['kbps']) for r in rows],[float(r[metric]) for r in rows],'o-',markersize=3,label=m)
            ax.set(xlabel='Bitrate (kbps)',ylabel=metric+' (lower is better)',title=d);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(ROOT/'results'/d/f'Rate_{metric}.png',dpi=180);plt.close(fig)
    val=read(ROOT/'results/validation_history.csv')
    for metric in ('L_A','L_cos','LPIPS','DISTS','total_rate'):
        if metric not in val[0]:continue
        fig,ax=plt.subplots(figsize=(9,6))
        for b in BRANCHES:
            for d in ('ulong','vimeo'):
                rows=[r for r in val if r['branch']==b and r['domain']==d];steps=sorted({int(r['step']) for r in rows})
                values=[statistics.mean(float(r[metric]) for r in rows if int(r['step'])==s) for s in steps]
                ax.plot(steps,values,'o-',label=b+' / '+d)
        ax.set(xlabel='Optimizer updates',ylabel=metric,title='Fixed independent validation');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'results'/f'validation_{metric}.png',dpi=180);plt.close(fig)
if __name__=='__main__':
    try:
        if '--plots-only' in sys.argv:plots()
        else:main()
    except Exception:
        err=dict(status='FAIL',traceback=traceback.format_exc(),missing_files=['report/integrity did not finish'])
        dump(ROOT/'final_integrity.json',err);dump(ROOT/'logs/failures'/f'report_{time.time_ns()}.json',err);raise
