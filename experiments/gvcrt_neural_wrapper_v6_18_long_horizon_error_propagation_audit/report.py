"""Deterministic frame-window, drift, delta and OLS numbers; no interpretation."""
import math,traceback
from io18 import *
WINDOWS=((1,8),(9,16),(17,32),(33,48),(49,64))
def main():
    import numpy as np
    frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    from adapter import validate
    series={};perframe=[];rawpoints=[];checks=[]
    for d in DATASETS:
        for v in sources(d):
            for m in METHODS:
                for q in QPS:
                    path=point(d,m,v['video_index'],q);r=load(path);validate(r,v,q,m)
                    fs=read(r['frame_metrics_path']);tr=read(r['transitions_path']);assert len(fs)==64 and len(tr)==63
                    transitions={int(x['to_frame']):float(x['FloLPIPS']) for x in tr};assert set(transitions)==set(range(1,64))
                    key=(d,v['name'],m,q);vals={k:[] for k in METRICS};cum=0
                    for i,f in enumerate(fs):
                        assert int(f['frame'])==i and int(f['actual_qp'])==r['actual_qps'][i]
                        bits=int(f['real_bits']);assert bits>0 and bits%8==0;cum+=bits//8
                        values={k:float(f[k]) for k in ('LPIPS','DISTS','PSNR','SSIM')};flo=float('nan') if i==0 else transitions[i]
                        assert all(math.isfinite(x) for x in values.values()) and (i==0 or math.isfinite(flo))
                        for k in METRICS:vals[k].append(flo if k=='FloLPIPS' else values[k])
                        perframe.append(dict(dataset=d,sequence=v['name'],method=m,external_QP=q,frame_index=i+1,source_frame_index=v['source_frame_indices'][i],actual_QP=int(f['actual_qp']),**values,FloLPIPS=flo,frame_bits=bits,frame_bytes=bits//8,cumulative_bytes=cum,sequence_actual_bpp=r['bpp'],sequence_real_bytes=r['real_bytes'],rate_accounting_fps=v['rate_accounting_fps'],comparison='diagnostic same-external-QP; not equal-rate'))
                    assert cum==r['real_bytes']
                    for k in METRICS:assert math.isclose(float(np.nanmean(vals[k])),r[k],rel_tol=1e-10,abs_tol=1e-12)
                    series[key]={k:np.array(a,dtype=float) for k,a in vals.items()}
                    rawpoints.append(r);checks.append(dict(dataset=d,sequence=v['name'],method=m,external_QP=q,point_path=str(path),point_SHA256=sha(path),real_bytes=r['real_bytes'],bpp=r['bpp'],actual_QPs=r['actual_qps'],independent_decode=r['independent_decode_pass'],state_sync=r['state_sync_pass'],compression_hash=r['compression_hash_after'],reused=bool(r.get('temporal_audit_reused'))))
        print('PER-FRAME PASS',d,flush=True)
    assert len(series)==240 and len(perframe)==15360
    write(ROOT/'evaluation/per_frame_metrics.csv',perframe);dump(ROOT/'evaluation/point_manifest.json',checks)
    dump(ROOT/'evaluation/raw_point_outputs.json',rawpoints)
    window=[];drift=[];excess=[];delta=[];slopes=[];plotrows=[]
    for (d,s,m,q),metrics in series.items():
        for k,a in metrics.items():
            ref=series[(d,s,'original',q)][k];da=a-ref
            for lo,hi in WINDOWS:
                values=a[lo-1:hi];valid=values[np.isfinite(values)]
                assert len(valid)==(hi-lo+1)-(k=='FloLPIPS' and lo==1)
                window.append(dict(dataset=d,sequence=s,method=m,external_QP=q,metric=k,window=f'{lo}-{hi}',mean=float(valid.mean()),std=float(valid.std(ddof=0)),count=len(valid)))
            early=float(np.nanmean(a[:8]));late=float(a[48:].mean());re=float(np.nanmean(ref[:8]));rl=float(ref[48:].mean())
            drift.append(dict(dataset=d,sequence=s,method=m,external_QP=q,metric=k,early_1_8=early,late_49_64=late,late_minus_early=late-early))
            excess.append(dict(dataset=d,sequence=s,method=m,external_QP=q,metric=k,method_early=early,method_late=late,original_early=re,original_late=rl,method_drift=late-early,original_drift=rl-re,excess_drift=(late-early)-(rl-re)))
            r=next(x for x in rawpoints if x['dataset']==d and x['sequence']==s and x['method']==m and x['QP']==q);rr=next(x for x in rawpoints if x['dataset']==d and x['sequence']==s and x['method']=='original' and x['QP']==q)
            for i in range(64):delta.append(dict(dataset=d,sequence=s,method=m,external_QP=q,metric=k,frame_index=i+1,metric_value=float(a[i]),original_metric_value=float(ref[i]),delta_vs_original=float(da[i]),method_sequence_actual_bpp=r['bpp'],original_sequence_actual_bpp=rr['bpp'],comparison='diagnostic same-external-QP; not equal-rate'))
            x=np.arange(2,65,dtype=float);y=a[1:];dy=da[1:];assert np.isfinite(y).all() and np.isfinite(dy).all()
            xc=x-x.mean();den=float(np.dot(xc,xc));b=float(np.dot(xc,y-y.mean())/den);bd=float(np.dot(xc,dy-dy.mean())/den)
            slopes.append(dict(dataset=d,sequence=s,method=m,external_QP=q,metric=k,raw_slope=b,delta_vs_original_slope=bd,n_frames=63,first_frame=2,last_frame=64,OLS_intercept=float(y.mean()-b*x.mean()),delta_OLS_intercept=float(dy.mean()-bd*x.mean())))
    def aggregate(rows,fields):
        out=[]
        for d in DATASETS:
            for m in METHODS:
                for q in QPS:
                    for k in METRICS:
                        rs=[r for r in rows if (r['dataset'],r['method'],r['external_QP'],r['metric'])==(d,m,q,k)];assert len(rs)==len(sources(d))
                        out.append(dict(dataset=d,method=m,external_QP=q,metric=k,**{f:float(np.mean([r[f] for r in rs])) for f in fields},**{f+'_sequence_std':float(np.std([r[f] for r in rs],ddof=0)) for f in fields},n_sequences=len(rs),aggregation='equal sequence weights',std_definition='population between-sequence ddof=0'))
        return out
    tables={'window_metrics':window,'temporal_drift_per_sequence':drift,'temporal_drift_dataset_summary':aggregate(drift,('early_1_8','late_49_64','late_minus_early')),'excess_drift_vs_original':excess,'excess_drift_dataset_summary':aggregate(excess,('method_early','method_late','original_early','original_late','method_drift','original_drift','excess_drift')),'per_frame_delta_vs_original':delta,'frame_index_slopes':slopes,'frame_index_slopes_dataset_summary':aggregate(slopes,('raw_slope','delta_vs_original_slope'))}
    for name,rows in tables.items():
        write(ROOT/'results'/f'{name}.csv',rows)
        if name!='per_frame_delta_vs_original':dump(ROOT/'results'/f'{name}.json',rows)
    for d in DATASETS:
        for m in METHODS:
            for q in QPS:
                for k in METRICS:
                    stack=np.stack([series[(d,v['name'],m,q)][k] for v in sources(d)]);original=np.stack([series[(d,v['name'],'original',q)][k] for v in sources(d)])
                    for i in range(64):
                        valid=np.isfinite(stack[:,i]);assert valid.all() or (k=='FloLPIPS' and i==0 and not valid.any())
                        plotrows.append(dict(dataset=d,method=m,external_QP=q,metric=k,frame_index=i+1,mean=float(stack[:,i].mean()) if valid.all() else float('nan'),delta_vs_original_mean=float((stack[:,i]-original[:,i]).mean()) if valid.all() else float('nan'),n_sequences=int(valid.sum()),aggregation='equal sequence weights'))
    write(ROOT/'results/frame_curve_dataset_means.csv',plotrows)
    dump(ROOT/'evaluation_integrity.json',dict(status='PASS',expected_points=240,completed_points=240,expected_frames=15360,completed_frames=len(perframe),methods=list(METHODS),external_qps=list(QPS),frames_per_point=64,missing_metric_files=[],NaN_FloLPIPS_frame1=240,other_NaN=0,table_row_counts={k:len(v) for k,v in tables.items()},slope_frame_range=[2,64],all_point_artifacts_verified=True))
    dump(ROOT/'codec_integrity.json',dict(status='PASS',real_RANS=True,independent_decode=True,state_synchronization=True,recurrent_state_not_reset_between_frames=True,teacher_forcing=False,frozen_compression_verified=True,per_frame_bits_sum_equals_real_stream_bytes=True,actual_QP_mapping_verified=True,live_module_audit=load(ROOT/'audits/live_codec_modules.json'),causal_code=load(ROOT/'audits/causal_state_code.json'),points=checks))
    lines=['# V6.18 numeric artifacts','', 'Protocol: 64 causal frames; external QP 0, 4, 9; diagnostic same-external-QP comparison; FP16 codec / FP32 wrapper; force_zero_thres 0.12.','', 'Frame windows: 1–8, 9–16, 17–32, 33–48, 49–64. Population standard deviation (ddof=0). FloLPIPS targets transitions; frame1 is undefined and excluded from window means.','', 'Drift: late_49_64 − early_1_8. Excess drift: method drift − original drift. OLS slopes: frames2–64. Dataset aggregates: equal sequence weights.','', '| Dataset | Sequences | Methods | QPs | Points | Frames |','|---|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        n=len(sources(d));lines.append(f'| {d} | {n} | 5 | 3 | {n*15} | {n*15*64} |')
    lines+=['','| File | Rows |','|---|---:|']+[f'| results/{k}.csv | {len(v)} |' for k,v in tables.items()]+['| evaluation/per_frame_metrics.csv | 15360 |','| results/frame_curve_dataset_means.csv | 11520 |','','Plots: `plots/<dataset>_qp<QP>_<metric>_vs_frame.png` and `plots/<dataset>_qp<QP>_<metric>_delta_vs_original.png`.','']
    (ROOT/'results/README_NUMBERS_ONLY.md').write_text('\n'.join(lines))
    print('NUMERIC TABLES PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'evaluation_integrity.json',dict(status='FAIL',error=traceback.format_exc()));raise
