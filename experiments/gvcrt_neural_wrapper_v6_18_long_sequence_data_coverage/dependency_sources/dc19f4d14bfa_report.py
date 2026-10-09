import argparse,math
import numpy as np
from v61_io import *
def methods(split):return ['original']+[f'step_{s}' for s in load(ROOT/'config.json')['checkpoint_additional_updates']] if split=='validation' else ['original','selected']
def collect(split):
    cfg=load(ROOT/'config.json');vs=load(ROOT/('validation_sources.json' if split=='validation' else 'final_sources.json'))['videos'];vs=[v for v in vs if v['dataset']==split];qps=cfg['validation_qps'] if split=='validation' else list(range(10));rows=[]
    for method in methods(split):
        for v in vs:
            for q in qps:
                r=load(point(split,method,v['video_index'],q))
                assert (r['dataset'],r['method'],r['video_index'],r['external_qp'])==(split,method,v['video_index'],q)
                assert r['source_record']==v and r['config_sha256']==sha(ROOT/'config.json')
                assert r['frames']==v['frames'] and r['num_transitions']==v['frames']-1 and r['force_zero_thres']==.12
                assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass']
                assert r['compression_hash_before']==r['compression_hash_after']
                for k in ('bitstream','feature','frame_metrics'):assert sha(r[k+'_path'])==r[k+'_sha256']
                assert Path(r['bitstream_path']).stat().st_size==r['real_bytes']==r['bytes_consumed']
                assert math.isclose(r['kbps'],r['real_bytes']*8*v['rate_fps']/v['frames']/1000,rel_tol=1e-12)
                assert math.isclose(r['bpp'],r['real_bytes']*8/(v['frames']*v['width']*v['height']),rel_tol=1e-12)
                assert r['rate_accounting_fps']==v['rate_fps'] and r['metric_crop']==[v['width'],v['height']]
                assert r['processing_canvas']==[(v['width']+63)//64*64,(v['height']+63)//64*64]
                if method!='original':
                    cp=Path(load(ROOT/'checkpoint_selection.json')['checkpoint']) if method=='selected' else checkpoint(int(method.split('_')[-1]));assert r['checkpoint_sha256']==sha(cp)
                frame=read(r['frame_metrics_path']);assert len(frame)==v['frames'] and sum(int(f['real_bits']) for f in frame)==r['real_bytes']*8
                for i,f in enumerate(frame):
                    assert int(f['frame'])==i and int(f['external_qp'])==q
                    assert int(f['actual_qp'])==(q if i==0 else q+[0,2,1][[0,1,0,2,0,2,0,2][i%8]])
                    assert all(math.isfinite(float(f[k])) for k in SPATIAL)
                transitions=read(Path(r['feature_path']).with_suffix('.transitions.csv'))
                assert [(int(t['from_frame']),int(t['to_frame'])) for t in transitions]==[(i,i+1) for i in range(v['frames']-1)]
                assert all(math.isfinite(float(t['FloLPIPS'])) for t in transitions)
                assert abs(np.mean([float(t['FloLPIPS']) for t in transitions])-r['FloLPIPS'])<1e-10
                assert all(math.isfinite(float(r[k])) for k in (*SPATIAL,'FloLPIPS','kbps','bpp'))
                rows.append(r)
    return rows
def pooled(rows,split):
    fid=module('v61_fid_report',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py');result=[];references={}
    for method in methods(split):
        for q in sorted({r['external_qp'] for r in rows}):
            own=[r for r in rows if r['method']==method and r['external_qp']==q];real=[];recon=[]
            for r in own:
                with np.load(r['feature_path']) as f:
                    assert f['real'].shape==f['reconstruction'].shape==(r['frames'],2048)
                    assert np.isfinite(f['real']).all() and np.isfinite(f['reconstruction']).all()
                    if r['video_index'] in references:assert np.allclose(references[r['video_index']],f['real'],rtol=1e-5,atol=1e-6)
                    else:references[r['video_index']]=f['real'].copy()
                    real.append(f['real']);recon.append(f['reconstruction'])
            a,b=np.concatenate(real),np.concatenate(recon);value=float(fid.low_rank_fid(a,b));assert math.isfinite(value)
            result.append(dict(dataset=split,method=method,external_qp=q,QP=q,kbps=float(np.mean([r['kbps'] for r in own])),
                bpp=float(np.mean([r['bpp'] for r in own])),real_bytes=sum(r['real_bytes'] for r in own),
                **{m:float(np.mean([r[m] for r in own])) for m in SPATIAL},
                FloLPIPS=sum(r['FloLPIPS']*r['num_transitions'] for r in own)/sum(r['num_transitions'] for r in own),FID=value,
                FID_num_samples=len(a),num_sequences=len(own),num_frames=sum(r['frames'] for r in own),num_transitions=sum(r['num_transitions'] for r in own)))
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('validation','ulong','uvg','virat720'),required=True);args=p.parse_args();split=args.split
    frozen();rows=collect(split);summary=pooled(rows,split);out=ROOT/'results'/split;out.mkdir(parents=True,exist_ok=True)
    write(out/'all_qp_summary.csv',summary);write(out/'per_sequence_all_qp.csv',[dict(r,bits=r['real_bytes']*8) for r in rows])
    sys.path.insert(0,str(V52));from rd_analysis import compare,points
    from scipy.interpolate import PchipInterpolator
    anchor=[r for r in summary if r['method']=='original'];eq=[];bd=[];diagnostics=[]
    for method in methods(split)[1:]:
        candidate=[r for r in summary if r['method']==method]
        for m in METRICS:
            e,b=compare(anchor,candidate,m,'original',method);eq.append(e);bd.append(b)
            x,y=points(anchor,m);curve=PchipInterpolator(x,y,extrapolate=False)
            for c in candidate:
                a=next(a for a in anchor if a['QP']==c['QP']);inside=x.min()<=math.log(c['kbps'])<=x.max()
                d=float(c[m]-curve(math.log(c['kbps']))) if inside else None
                diagnostics.append(dict(dataset=split,method=method,QP=c['QP'],metric=m,Original_kbps=a['kbps'],candidate_kbps=c['kbps'],
                    same_QP_bitrate_delta=c['kbps']-a['kbps'],same_QP_bitrate_delta_percent=100*(c['kbps']/a['kbps']-1),
                    Original_metric=a[m],candidate_metric=c[m],same_QP_metric_delta=c[m]-a[m],equal_rate_perceptual_delta=d,
                    equal_rate_status='IN_RANGE' if inside else 'OUT_OF_RANGE'))
    write(out/'equal_rate_summary.csv',eq);write(out/'bd_rate.csv',bd);write(out/'operating_point_diagnostics.csv',diagnostics)
    if split=='validation':
        write(ROOT/'validation_per_video.csv',rows);write(ROOT/'validation_summary.csv',summary);ranks=[]
        for step in load(ROOT/'config.json')['checkpoint_additional_updates']:
            method=f'step_{step}';e=[r for r in eq if r['method']==method and r['metric'] in ('LPIPS','DISTS')];b=[r for r in bd if r['method']==method and r['metric'] in ('LPIPS','DISTS')]
            assert len(e)==2;valid=all(r['status']=='valid' for r in e)
            score=sum(float(r['mean_equal_rate_delta'])/float(np.mean([a[r['metric']] for a in anchor])) for r in e)/2 if valid else None
            gate=valid and all(float(r['mean_equal_rate_delta'])<0 for r in e) and all(r['status']!='valid' or float(r['BD_rate_percent'])<=0 for r in b)
            candidate=[r for r in summary if r['method']==method];reduction=float(np.mean([1-c['kbps']/next(a['kbps'] for a in anchor if a['QP']==c['QP']) for c in candidate]))
            ranks.append(dict(additional_updates=step,global_step=20000+step,method=method,gate_pass=gate,valid_equal_rate=valid,normalized_equal_rate_score=score,mean_bitrate_reduction_fraction=reduction))
        eligible=[r for r in ranks if r['gate_pass']];fallback=not eligible
        if eligible:best=min(eligible,key=lambda r:(-r['mean_bitrate_reduction_fraction'],r['additional_updates']))
        else:
            usable=[r for r in ranks if r['valid_equal_rate']];assert usable,'No validation common-rate overlap; cannot safely select checkpoint'
            best=min(usable,key=lambda r:(r['normalized_equal_rate_score'],r['additional_updates']))
        write(ROOT/'checkpoint_selection.csv',ranks);cp=checkpoint(best['additional_updates'])
        dump(ROOT/'checkpoint_selection.json',dict(status='PASS',checkpoint=str(cp),checkpoint_sha256=sha(cp),additional_updates=best['additional_updates'],global_step=best['global_step'],
            used_final_test=False,selection_datasets=['validation_ulong'],fallback_used=fallback,rule=load(ROOT/'config.json')['selection_rule'],
            validation_per_video_sha256=sha(ROOT/'validation_per_video.csv'),validation_summary_sha256=sha(ROOT/'validation_summary.csv'),selection_csv_sha256=sha(ROOT/'checkpoint_selection.csv')))
    dump(out/'report_integrity.json',dict(status='PASS',points=len(rows),summary_rows=len(summary),expected_FID_samples=summary[0]['FID_num_samples'],threshold=.12))
    print('REPORT PASS',split,flush=True)
if __name__=='__main__':main()
