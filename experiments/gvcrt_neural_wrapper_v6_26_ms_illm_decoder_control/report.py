"""Raw fixed-endpoint values and common-rate interpolation; no interpretation."""
import ast,math,statistics,traceback
from io21 import *
PAIRS=(('B_decoder_conditional_projected_1000','A_decoder_continue_1000'),('C_decoder_illm_1000','B_decoder_conditional_projected_1000'),('C_decoder_illm_1000','A_decoder_continue_1000'),*((m,b) for m in METHODS[2:] for b in METHODS[:2]))
def comparisons(dataset,sequence,metric,rows,pooled=False,methods=METHODS,pairs=PAIRS):
    source=(V15/'report.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='comparisons')
    code=ast.get_source_segment(source,node).replace('expected_models=6','expected_models=len(METHODS)').replace('available_models=6','available_models=len(METHODS)').replace('six-model','all-model')
    ns=dict(globals(),METHODS=methods,PAIRS=pairs);exec(compile(code,str(V15/'report.py'),'exec'),ns)
    results,grid=ns['comparisons'](dataset,sequence,metric,rows,pooled)
    for r in results:
        r['relative_change_percent']=100*r['delta']/r['reference_mean'] if r['status']=='available' and r['reference_mean'] else None
        r['relative_change_definition']='100*(method_mean-reference_mean)/reference_mean; NOT BD-rate'
    return results,grid
def summarize(d,per,n,pairs=PAIRS):
    out=[]
    for method,reference in pairs:
        for metric in ('LPIPS','DISTS','FloLPIPS'):
            group=[r for r in per if r['method']==method and r['reference']==reference and r['metric']==metric];ok=[r for r in group if r['status']=='available'];complete=len(group)==len(ok)==n
            r=dict(dataset=d,method=method,reference=reference,metric=metric,status='available' if complete else 'unavailable',expected_sequences=n,available_sequences=len(ok),rate_axis='per-video actual bpp',aggregation='equal video weight; unavailable if any sequence unavailable',grid_points_per_sequence=100,shared_interval='all methods common per-video interval; see per-sequence/grid',reason='' if complete else 'one or more videos unavailable')
            for k in ('method_mean','reference_mean','delta','better_grid_fraction','relative_change_percent'):r[k]=statistics.mean(t[k] for t in ok) if complete and all(t[k] is not None for t in ok) else None
            r['mean_per_video_relative_change_percent']=r.pop('relative_change_percent')
            r['relative_change_percent']=100*(r['method_mean']-r['reference_mean'])/r['reference_mean'] if complete and r['reference_mean'] else None
            r['relative_change_definition']='100*(dataset method_mean-dataset reference_mean)/dataset reference_mean; NOT BD-rate';out.append(r)
    return out
def window_rows(r,fr,tr):
    out=[]
    for lo,hi in ((0,1),(1,8),(8,16),(16,32),(32,48),(48,64)):
        fs=fr[lo:hi];ts=[x for x in tr if lo<=int(x['to_frame'])<hi];ni=sum(x['frame_type']=='I' for x in fs)
        out.append(dict(IP=r['IP'],dataset=r['dataset'],method=r['method'],sequence=r['sequence'],QP=r['QP'],window_1based=f'{lo+1}-{hi}',start_frame0=lo,end_frame0_exclusive=hi,I_frames=ni,P_frames=len(fs)-ni,transition_count=len(ts),transition_rule='destination frame within window, includes cross-refresh',**{k:statistics.mean(float(x[k]) for x in fs) for k in ('LPIPS','DISTS','PSNR','SSIM','MS_SSIM')},FloLPIPS=statistics.mean(float(x['FloLPIPS']) for x in ts) if ts else None,real_bits=sum(int(x['real_bits']) for x in fs),payload_bits=sum(int(x['payload_bits']) for x in fs),rate_axis='full64 actual bpp',full64_bpp=r['bpp']))
    return out
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
            inherited=np.load(V20/'features'/f'GT_{d}.npy');assert np.allclose(reference,inherited,atol=1e-6,rtol=1e-5)
            # Fixed inherited per-dataset reference features, never another dataset's pool.
            reference=inherited
            poolhash=hashlib.sha256(json.dumps([(v['source_sha256'],v['source_frame_indices'],v['rgb_sha256']) for v in vs],sort_keys=True).encode()).hexdigest()
            poolaudit[d]=dict(path=str(V20/'features'/f'GT_{d}.npy'),sha256=sha(V20/'features'/f'GT_{d}.npy'),frame_pool_sha256=poolhash,frames=len(reference),fixed_GT=True)
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
    branches={}
    for b in BRANCHES:
        folder=ROOT/'branches'/b;ti=load(folder/'training_integrity.json');assert ti['status']=='PASS' and ti['updates']==1000
        rows=[json.loads(x) for x in (folder/'training_logs/train.jsonl').read_text().splitlines()]
        plan=load(ROOT/'plans/shared.json')['updates'];assert len(rows)==1000
        for r,p in zip(rows,plan):
            assert all(r[k]==p[k] for k in p) and r['supervision_frames']==15 and len(r['frame_trace'])==15
            assert [t['window_position'] for t in r['frame_trace']]==list(range(49,64))
            assert r['warmup_frames']==48 and r['all_finite']
            assert r['lambda_rec']==.1383589248954108
            assert r['lambda_eff']==((.05*r['B1_gradient_norm']/(r['unweighted_adv_gradient_norm']+1e-12) if min(r['B1_gradient_norm'],r['unweighted_adv_gradient_norm'])>1e-12 else 0.) if b!='A_decoder_continue' else 0.)
            assert r['adv_B1_ratio']<=.05+1e-8
            assert abs(r['weighted_reconstruction_L1']-.1383589248954108*r['reconstruction_L1'])<1e-7
            if b!='A_decoder_continue':
                assert r['G_D_buffers_unchanged'] and r['D_PBG_unchanged']
                assert abs(r['weighted_projected_adv']-r['lambda_adv']*r['projected_adv'])<1e-7
                assert r['feature_projection_hash']==load(ROOT/'audits/'+('illm_source.json' if b=='C_decoder_illm' else 'projected_source.json'))['module_hash' if b=='C_decoder_illm' else 'feature_hash']
            assert r['D_optimizer_updates']==(r['adaptation_step'] if b!='A_decoder_continue' else 0)
        if b!='A_decoder_continue':write(folder/'training_logs/discriminator.csv',[{k:r[k] for k in ('adaptation_step','D_loss','D_grad_norm','D_optimizer_updates','D_lr','projected_adv','weighted_projected_adv','lambda_adv','feature_projection_hash','G_D_buffers_unchanged','D_PBG_unchanged')} for r in rows])
        index=load(folder/'checkpoint_index.json');assert sorted(map(int,index))==[0,250,500,1000]
        for cp in index.values():assert sha(cp['path'])==cp['sha256'] and sha(cp['inference']['path'])==cp['inference']['sha256']
        assert index['0']['module_hashes']==load(ROOT/'config.json')['source_checkpoint']['module_hashes'];branches[b]=dict(updates=1000,supervised_P_frames=15000,warmup_P_frames=48000,D_updates=(1000 if b!='A_decoder_continue' else 0),checkpoints=index)
    assert all(load(ROOT/'audits'/f'{b}_pretraining.json')['status']==load(ROOT/'audits'/f'{b}_codec_equivalence.json')['status']=='PASS' for b in BRANCHES)
    assert all(r['expected']==r['completed'] for r in counts.values())
    assert sum(r['completed'] for r in counts.values())==800
    prior=load(ROOT/'audits/historical_inventory.json');now=historical_inventory();changed=[p for p,s in prior.items() if now.get(p)!=s];assert not changed,changed
    frozen()
    assert load(ROOT/'historical/final_integrity.json')['status']=='PASS'
    binary=[]
    for path in ROOT.rglob('*'):
        if path.is_file() and path.suffix in ('.pt','.pth','.ckpt','.npz','.npy','.bin') and 'vendor' not in path.parts:
            binary.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),server_only=True))
    indexed={x['path'] for x in binary};external={Path(load(ROOT/'config.json')['source_checkpoint']['path'])}
    for p in (ROOT/'parts').rglob('*.json'):
        r=load(p)
        for key in ('bitstream_path','feature_path','checkpoint_path'):
            if r.get(key):external.add(Path(r[key]))
    for p in external:
        if str(p) not in indexed:
            assert p.is_file(),('missing indexed server artifact',str(p))
            binary.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),server_only=True,referenced_historical_artifact=True));indexed.add(str(p))
    dump(ROOT/'binary_artifacts_index.json',dict(artifacts=binary))
    dump(ROOT/'audits/computational_integrity.json',dict(status='PASS',branches=branches,expected_RD=800,completed_RD=800,counts=counts,reused=reused,fresh=800-reused,unavailable_comparisons=unavailable,missing=[],failures=[],historical_files_unchanged=True,source_checkpoint_sha256=EXPECTED,lambda_ceiling=None,wrapper_frozen=True,lambda_rec=.1383589248954108,real_RANS=True,independent_decode=True,projected_feature_buffers_frozen=True,RD_PNG=len(list((ROOT/'results').rglob('Rate_*.png'))),source_hashes=load(ROOT/'audits/dependencies.json'),finished_unix=time.time()))
    print('COMPUTATIONAL PASS 800/800',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'audits/computational_integrity.json',dict(status='FAIL',error=traceback.format_exc()));raise

