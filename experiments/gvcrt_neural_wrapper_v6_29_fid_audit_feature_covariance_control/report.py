from comparison import *
def main():
    import numpy as np,torch
    from adapter import validate
    torch.set_num_threads(2);frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    fid=module('gop21_fid',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py').low_rank_fid
    stored={};counts={};poolaudit={};unavailable=[];reused=0
    for ip in IPS:
        for d in DATASETS:
            vs=sources(d);out=ROOT/'results'/f'ip_{ip}'/d;raw=[];gt=[];windows=[]
            for v in vs:
                for m in METHODS:
                    for q in range(10):
                        r=load(point(d,m,v['video_index'],q,ip));validate(r,v,q,m,ip);sourcefps=float(v.get('source_fps_metadata',v['rate_accounting_fps']));r.update(source_fps_metadata=sourcefps,kbps_source_fps=r['bits_per_frame']*sourcefps/1000,kbps_rate_accounting_fps=r['bits_per_frame']*v['rate_accounting_fps']/1000);dump(point(d,m,v['video_index'],q,ip),r);reused+=bool(r.get('reused'));raw.append(dict(r,rate_axis='actual bpp',aggregation='per-video measured RD point'))
                        fr=read(r['frame_metrics_path']);tr=read(r['transitions_path'])
                        for i,f in enumerate(fr):
                            last=0 if ip<0 else i//ip*ip
                            f.update(frame_type='I' if i==last else 'P',reference_age=i-last,IP=ip)
                        for t in tr:t.update(IP=ip,cross_I_refresh=ip>0 and int(t['to_frame'])%ip==0)
                        prefix=f"video_{v['video_index']:02d}_qp{q}.csv";write(out/'per_frame'/m/prefix,fr);write(out/'transitions'/m/prefix,tr)
                        windows.extend(window_rows(r,fr,tr))
                r=next(r for r in raw if r['method']=='original' and r['video_index']==v['video_index'] and r['QP']==0)
                with np.load(r['feature_path']) as f:gt.append(f['real'].copy())
            reference=np.concatenate(gt);assert reference.shape==(64*len(vs),2048)
            gt_path=(V27 if d=='mcljcv' else V20)/'features'/f'GT_{d}.npy';inherited=np.load(gt_path);assert np.allclose(reference,inherited,atol=1e-6,rtol=1e-5)
            # Fixed inherited per-dataset reference features, never another dataset's pool.
            reference=inherited
            poolhash=hashlib.sha256(json.dumps([(v['source_sha256'],v['source_frame_indices'],v['rgb_sha256']) for v in vs],sort_keys=True).encode()).hexdigest()
            poolaudit[d]=dict(path=str(gt_path),sha256=sha(gt_path),frame_pool_sha256=poolhash,frames=len(reference),fixed_GT=True)
            pooled=[];macro=[]
            for m in METHODS:
                for q in range(10):
                    rs=[next(r for r in raw if r['method']==m and r['QP']==q and r['video_index']==v['video_index']) for v in vs];recs=[]
                    for i,r in enumerate(rs):
                        with np.load(r['feature_path']) as f:
                            assert np.allclose(f['real'],reference[i*64:(i+1)*64],atol=1e-6,rtol=1e-5)
                            assert f['reconstruction'].shape==(64,2048) and np.isfinite(f['reconstruction']).all();recs.append(f['reconstruction'])
                    value=float(fid(reference,np.concatenate(recs)));assert math.isfinite(value)
                    pooled.append(dict(IP=ip,dataset=d,method=m,QP=q,bpp=statistics.mean(r['bpp'] for r in rs),FID=value,samples=len(reference),sequences=len(vs),rate_axis='dataset mean actual bpp',aggregation='pooled dataset features, not average per-video FID',GT_pool_sha256=poolhash))
                    macro.append(dict(IP=ip,dataset=d,method=m,QP=q,**{k:statistics.mean(r[k] for r in rs) for k in ('bpp','kbps','kbps_20fps','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM')},FID=value,sequences=len(vs),rate_axis='dataset mean actual bpp',aggregation='equal videos; pooled FID',samples=len(reference)))
            write(out/'per_sequence_all_qp.csv',raw);write(out/'all_qp_summary.csv',macro);write(out/'fid_pooled.csv',pooled)
            for m in METHODS:
                write(out/m/'per_sequence_all_qp.csv',[r for r in raw if r['method']==m]);write(out/m/'window_metrics.csv',[r for r in windows if r['method']==m])
            per=[];grid=[]
            for v in vs:
                for metric in ('LPIPS','DISTS','FloLPIPS'):
                    rr,gg=comparisons(d,v['name'],metric,[r for r in raw if r['video_index']==v['video_index']]);per.extend(rr);grid.extend(gg)
            fidrows,gg=comparisons(d,'pooled_dataset','FID',pooled,True);grid.extend(gg)
            for name,rows in [('equal_rate_per_sequence',per),('equal_rate_summary',summarize(d,per,len(vs))),('equal_rate_fid_pooled',fidrows),('common_rate_grid',grid or [dict(status='unavailable')])]:write(out/(name+'.csv'),[dict(r,IP=ip) for r in rows])
            stored[(ip,d)]=(raw,pooled);counts[f'ip_{ip}/{d}']=dict(expected=len(vs)*len(METHODS)*10,completed=len(raw),pooled_FID=len(pooled))
            unavailable.extend(dict(r,IP=ip) for r in per+fidrows if r['status']=='unavailable');print('TABLES',ip,d,len(raw),flush=True)
    dump(ROOT/'audits/GT_feature_pools.json',poolaudit)
    subprocess.run([PLOT_PYTHON,'-B',str(ROOT/'plot.py')],check=True,cwd=REPO)
    dump(ROOT/'audits/aggregation_integrity.json',dict(status='PASS',counts=counts,reused=reused,expected=1840,completed=sum(r['completed'] for r in counts.values()),GT_pools=poolaudit,unavailable=unavailable))
    print('AGGREGATION PASS',flush=True)
if __name__=='__main__':main()
