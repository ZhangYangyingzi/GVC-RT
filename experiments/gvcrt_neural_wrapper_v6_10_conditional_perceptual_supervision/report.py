"""Validated raw exports, predetermined figures, and integrity only."""
import math,statistics
from v68_io import *
def main():
    import torch,numpy as np
    torch.set_num_threads(2);cfg=frozen(True);raw=[];fid=[];bootstrap=[];macro=[];heldout=[]
    for d in DATASETS:
        own=[]
        for m in METHODS:
            for v in sources(d):
                rows=[]
                for q in range(10):
                    r=load(point(d,m,v['video_index'],q));validate_point(r,v,q,m)
                    r.update(fps=v['rate_accounting_fps']);rows.append(r)
                write(ROOT/'results'/d/'per_video'/f'{m}_{v["video_index"]:02d}.csv',rows);own.extend(rows)
            for q in range(10):
                rs=[r for r in own if r['method']==m and r['QP']==q];assert len(rs)==len(sources(d))
                macro.append(dict(dataset=d,method=m,QP=q,sequences=len(rs),
                    **{k:statistics.mean(r[k] for r in rs) for k in ('kbps','bpp',*METRICS)}))
                r=load(ROOT/'parts/fid'/d/m/f'qp{q}.json')
                assert r['status']=='PASS' and math.isfinite(r['FID'])
                assert r['real_bytes']==sum(x['real_bytes'] for x in rs)
                assert r['num_frames']==sum(x['frames'] for x in rs)
                assert sha(r['bootstrap_path'])==r['bootstrap_sha256']
                for pt in r['source_points']:assert sha(pt['path'])==pt['sha256']
                draws=read(r['bootstrap_path']);assert len(draws)==20 and all(math.isfinite(float(x['FID'])) for x in draws)
                fid.append(r);bootstrap.extend(draws)
        write(ROOT/'results'/d/'raw_rd.csv',own);raw.extend(own)
    assert len(raw)==2480 and len(fid)==320 and len(bootstrap)==6400
    for name,rows in [('raw_rd',raw),('dataset_macro_rd',macro),('fid_raw',fid),('fid_bootstrap',bootstrap)]:
        write(ROOT/'results'/f'{name}.csv',rows)
    write(ROOT/'results/fid_bootstrap_summary.csv',[dict(dataset=r['dataset'],method=r['method'],QP=r['QP'],**r['bootstrap']) for r in fid])
    manifest=load(ROOT/'audits/vimeo_heldout_manifest.json');heldsummary=[];compute_fid=fid_function()
    for m in METHODS:
        for q in manifest['external_qps']:
            gt=[];rec=[];rs=[]
            for v in manifest['clips']:
                r=load(ROOT/'parts/heldout'/m/f'video_{v["index"]:03d}_qp{q}.json')
                assert r['status']=='PASS' and r['source_frame_rgb_sha256']==v['frame_rgb_sha256'] and r['frames']==7
                assert r['checkpoint_sha256']==('' if m=='original' else cfg['checkpoints'][m]['sha256'])
                assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'] and r['real_bytes']==r['bytes_consumed']
                assert r['compression_hash_before']==r['compression_hash_after']==cfg['compression_hash']
                for key in ('bitstream','feature','frame_metrics','transitions'):assert sha(r[key+'_path'])==r[key+'_sha256']
                assert all(math.isfinite(r[k]) for k in (*METRICS,'kbps','bpp'))
                with np.load(r['feature_path']) as f:gt.append(f['real']);rec.append(f['reconstruction'])
                rs.append(r);heldout.append(r)
            fval=compute_fid(np.concatenate(gt),np.concatenate(rec));assert math.isfinite(fval)
            heldsummary.append(dict(method=m,QP=q,FID=fval,frames=224,real_bytes=sum(r['real_bytes'] for r in rs),
                **{k:statistics.mean(r[k] for r in rs) for k in (*METRICS,'kbps','bpp')}))
    assert len(heldout)==768
    write(ROOT/'results/vimeo_heldout_raw.csv',heldout);write(ROOT/'results/vimeo_heldout_summary.csv',heldsummary)
    for b in BRANCHES:
        rows=[json.loads(x) for x in (ROOT/'training_logs'/f'{b}.jsonl').read_text().splitlines()]
        assert [r['absolute_step'] for r in rows]==list(range(1001,2001))
        assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,float))
        assert load(ROOT/'branches'/b/'training_integrity.json')['compression_core_frozen']
        for k in (1,100,500,1000):
            g=load(ROOT/'branches'/b/'parts'/f'gradient_components_{k}.json')
            assert g['extra_optimizer_updates']==0
        if b!='C_plain':assert load(ROOT/'branches'/b/'warmup_integrity.json')['status']=='PASS'
        for step in (1250,1500,2000):
            meta=load(ROOT/'branches'/b/'checkpoint_hashes.json')[str(step)]
            assert sha(meta['path'])==meta['sha256']
            s=torch.load(meta['path'],map_location='cpu',weights_only=True)
            assert s['code_hashes']=={p.name:sha(p) for p in ROOT.glob('*.py')}
            assert s['compression_hash']==cfg['compression_hash'] and s['training_plan_position']==step-1000
            if step in (1500,2000):
                inf=torch.load(meta['inference']['path'],map_location='cpu',weights_only=True);assert set(inf)=={'wrapper','bridge','generator'}
                assert all(tensor_hash(inf[k])==tensor_hash(s[k]) for k in inf)
        write(ROOT/'results'/f'training_{tag(b)}.csv',rows)
    argv=['/data1/anaconda3_new/anaconda_program/bin/python','-B',str(ROOT/'report.py'),'--plots-only']
    with (ROOT/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=argv,role='existing_matplotlib_environment_only',unix=time.time()))+'\n')
    subprocess.run(argv,cwd=REPO,check=True)
    visualization(raw)
    frozen(True)
    assert load(ROOT/'audits/objective_smoke.json')['status']=='PASS'
    artifacts=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    dump(ROOT/'generated_files.json',dict(files=[str(p.relative_to(ROOT)) for p in sorted(artifacts)]))
    dump(ROOT/'final_integrity.json',dict(status='PASS',counts=dict(raw_RD=len(raw),FID=len(fid),bootstrap=len(bootstrap),heldout=len(heldout)),
        checked=dict(all_expected_points=True,finite_metrics=True,independent_decode=True,real_bytes=True,
        frozen_compression=True,old_source_files_unchanged=True,checkpoint_hashes=True,discriminator_excluded_from_inference=True),
        output_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*') if p.is_file()},
        finished_unix=time.time()))
    print('INTEGRITY PASS',flush=True)
def plot(macro,fid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for d in DATASETS:
        for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
            fig,ax=plt.subplots(figsize=(9,6))
            for m in METHODS:
                rows=sorted([r for r in (fid if metric=='FID' else macro) if r['dataset']==d and r['method']==m],key=lambda r:r['kbps'])
                assert len(rows)==10
                ax.plot([r['kbps'] for r in rows],[r[metric] for r in rows],marker='o',markersize=3,label=m)
            ax.set_xlabel('Bitrate (kbps)');ax.set_ylabel(metric+' (lower is better)');ax.set_title(d)
            ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
            fig.savefig(ROOT/'results'/d/f'Rate_{metric}.png',dpi=180);plt.close(fig)
def visualization(raw):
    import numpy as np
    from PIL import Image,ImageDraw
    io=module('v610_plot_sources',V62B/'v62b_io.py')
    chosen=load(ROOT/'audits/visualization_plan.json')['comparison_frames']
    rows=[r for r in raw if r.get('visualization_dir')]
    for r in rows:
        v=next(v for v in sources(r['dataset']) if v['video_index']==r['video_index'])
        wanted=set(i for i in chosen if i<v['frames'])
        for i,arr in enumerate(io.arrays(v)):
            if i not in wanted:continue
            p=Path(r['visualization_dir'])/f'frame_{i:06d}.png';assert p.exists()
            left=Image.fromarray(arr);right=Image.open(p).convert('RGB')
            assert left.size==right.size
            canvas=Image.new('RGB',(left.width*2,left.height+28),'white')
            canvas.paste(left,(0,28));canvas.paste(right,(left.width,28))
            draw=ImageDraw.Draw(canvas);draw.text((8,7),'Source',fill='black');draw.text((left.width+8,7),r['method'],fill='black')
            dst=ROOT/'results'/r['dataset']/'reconstruction_pairs'/f'{r["method"]}_v{r["video_index"]}_qp{r["QP"]}_f{i}.png'
            dst.parent.mkdir(parents=True,exist_ok=True);canvas.save(dst)
if __name__=='__main__':
    if '--plots-only' in sys.argv:
        data=[]
        for name in ('dataset_macro_rd','fid_raw'):
            rs=read(ROOT/'results'/f'{name}.csv')
            for r in rs:
                for k in ('kbps','LPIPS','DISTS','FloLPIPS','FID'):
                    if k in r:r[k]=float(r[k])
            data.append(rs)
        plot(*data)
    else:main()
