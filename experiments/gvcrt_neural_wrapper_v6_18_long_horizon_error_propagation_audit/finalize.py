"""Independent numeric/accounting checks and before/after historical SHA audit."""
import math,statistics,traceback
from io18 import *
def main():
    import numpy as np
    from PIL import Image
    frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    ei=load(ROOT/'evaluation_integrity.json');ci=load(ROOT/'codec_integrity.json');assert ei['status']==ci['status']=='PASS'
    rows=read(ROOT/'evaluation/per_frame_metrics.csv');assert len(rows)==15360
    grouped={}
    for r in rows:
        key=(r['dataset'],r['sequence'],r['method'],int(r['external_QP']));grouped.setdefault(key,[]).append(r)
        for k in (*METRICS,'PSNR','SSIM','sequence_actual_bpp'):
            x=float(r[k]);assert math.isfinite(x) or (k=='FloLPIPS' and int(r['frame_index'])==1 and math.isnan(x))
    assert len(grouped)==240 and set(k[2] for k in grouped)==set(METHODS) and set(k[3] for k in grouped)==set(QPS)
    for key,rs in grouped.items():
        assert [int(r['frame_index']) for r in rs]==list(range(1,65))
        assert sum(int(r['frame_bytes']) for r in rs)==int(rs[-1]['cumulative_bytes'])==int(rs[-1]['sequence_real_bytes'])
    windows=read(ROOT/'results/window_metrics.csv');drift=read(ROOT/'results/temporal_drift_per_sequence.csv');excess=read(ROOT/'results/excess_drift_vs_original.csv');slopes=read(ROOT/'results/frame_index_slopes.csv');delta=read(ROOT/'results/per_frame_delta_vs_original.csv')
    close=lambda a,b:math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12)
    for r in windows:
        rs=grouped[(r['dataset'],r['sequence'],r['method'],int(r['external_QP']))];lo,hi=map(int,r['window'].split('-'));ys=[float(x[r['metric']]) for x in rs[lo-1:hi] if math.isfinite(float(x[r['metric']]))]
        assert int(r['count'])==len(ys) and close(r['mean'],statistics.mean(ys)) and close(r['std'],statistics.pstdev(ys))
    for r in drift:assert close(r['late_minus_early'],float(r['late_49_64'])-float(r['early_1_8']))
    for r in excess:
        assert close(r['method_drift'],float(r['method_late'])-float(r['method_early']))
        assert close(r['original_drift'],float(r['original_late'])-float(r['original_early']))
        assert close(r['excess_drift'],float(r['method_drift'])-float(r['original_drift']))
    for r in slopes:
        key=(r['dataset'],r['sequence'],r['method'],int(r['external_QP']));rs=grouped[key];ori=grouped[(key[0],key[1],'original',key[3])];k=r['metric'];ys=np.array([float(x[k]) for x in rs[1:]]);ys0=np.array([float(x[k]) for x in ori[1:]])
        X=np.column_stack([np.ones(63),np.arange(2,65)]);b=np.linalg.lstsq(X,ys,rcond=None)[0];bd=np.linalg.lstsq(X,ys-ys0,rcond=None)[0];assert close(r['raw_slope'],b[1]) and close(r['delta_vs_original_slope'],bd[1]) and int(r['n_frames'])==63
    for r in delta:
        x=float(r['delta_vs_original']);a=float(r['metric_value']);b=float(r['original_metric_value'])
        if math.isnan(x):assert r['metric']=='FloLPIPS' and int(r['frame_index'])==1 and math.isnan(a) and math.isnan(b)
        else:assert math.isfinite(x) and close(x,a-b)
    for stem,fields in [('temporal_drift',('early_1_8','late_49_64','late_minus_early')),('excess_drift',('excess_drift',)),('frame_index_slopes',('raw_slope','delta_vs_original_slope'))]:
        detailed=drift if stem=='temporal_drift' else excess if stem=='excess_drift' else slopes
        for r in read(ROOT/'results'/f'{stem}_dataset_summary.csv'):
            group=[x for x in detailed if all(x[k]==r[k] for k in ('dataset','method','external_QP','metric'))];assert int(r['n_sequences'])==len(group)==len(sources(r['dataset']))
            for field in fields:assert close(r[field],statistics.mean(float(x[field]) for x in group))
    pngs=list((ROOT/'plots').glob('*.png'));assert len(pngs)==72
    for p in pngs:
        with Image.open(p) as im:im.verify()
    before=load(ROOT/'audits/historical_sha256_before.json');after={};changed=[];missing=[]
    for p,h in before.items():
        if not Path(p).exists():missing.append(p);continue
        after[p]=sha(p)
        if after[p]!=h:changed.append(dict(path=p,before=h,after=after[p]))
    # Historical experiment trees themselves must have no new/deleted audited files.
    for folder in (V15,V16,V17):
        now={str(p) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.pid','.tmp')};then={p for p in before if Path(p).is_relative_to(folder)}
        assert now==then,('historical inventory changed',folder,now-then,then-now)
    dump(ROOT/'audits/historical_sha256_after.json',after);dump(ROOT/'audits/historical_unchanged.json',dict(status='PASS' if not changed and not missing else 'FAIL',files_checked=len(before),changed=changed,missing=missing))
    assert not changed and not missing
    checkpoints=load(ROOT/'checkpoint_integrity.json');assert checkpoints['status']=='PASS'
    for m,cp in checkpoints['checkpoints'].items():
        if m=='original':
            for k,p in cp['absolute_path'].items():assert sha(p)==cp['SHA256'][k]
        else:assert sha(cp['absolute_path'])==cp['SHA256'] and sha(cp['inference_export']['path'])==cp['inference_export']['sha256']
    schedules=load(ROOT/'gpu_schedule.json');assert schedules['status']=='PASS'
    outputs={str(p.relative_to(ROOT)):sha(p) for folder in ('evaluation','results','plots') for p in (ROOT/folder).rglob('*') if p.is_file()}
    reuse=load(ROOT/'evaluation/reuse_manifest.json');fresh=240-reuse['reused_points']
    dump(ROOT/'audits/numeric_integrity.json',dict(status='PASS',window_rows=len(windows),drift_rows=len(drift),excess_rows=len(excess),slope_rows=len(slopes),delta_rows=len(delta),independent_population_std=True,independent_OLS_lstsq=True,sequence_equal_weight_means=True,CSV_NaN_rule_verified=True,PNG_decode_verified=True))
    dump(ROOT/'final_integrity.json',dict(status='PASS',no_training=True,historical_files_unchanged=True,historical_files_SHA_verified=len(before),source_checkpoint_hashes_verified=True,frozen_compression_verified=True,real_RANS=True,independent_decode=True,state_synchronization=True,recurrent_state_not_reset_between_frames=True,teacher_forcing=False,source_frames_verified=True,actual_QP_mapping_verified=True,expected_evaluation_points=240,completed_evaluation_points=240,failed_evaluation_points=0,expected_per_frame_rows=15360,completed_per_frame_rows=len(rows),missing_metric_files=[],no_NaN_except_metric_definition_permitted=True,per_frame_FloLPIPS_frame1_NaN_count=240,per_frame_delta_FloLPIPS_frame1_NaN_count=240,other_NaN_count=0,methods=list(METHODS),external_qps=list(QPS),frames_per_point=64,reused_points=reuse['reused_points'],fresh_points=fresh,expected_plots=72,completed_plots=len(pngs),datasets={d:dict(sequences=len(sources(d)),evaluation_points=len(sources(d))*15,frames=len(sources(d))*15*64) for d in DATASETS},diagnostic_comparison='same-external-QP; not equal-rate',missing=[],failed=[],output_SHA256=outputs,finished_unix=time.time()))
    print('FINAL INTEGRITY PASS 240/240',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'final_integrity.json',dict(status='FAIL',error=traceback.format_exc()));raise
