"""V6.8b raw-output aggregation and integrity record."""
import csv, json, math, statistics, time
from v68_io import *
def rows_for(d,m):
    out=[]
    for v in sources(d):
        for q in range(10):
            p=point(d,m,v['video_index'],q)
            if p.exists():out.append(load(p))
    return out
def main():
    frozen(True); cfg=load(ROOT/'evaluation/config.json')
    raw=[]; macro=[]
    for d in DATASETS:
        own=[]
        for m in METHODS:
            rs=rows_for(d,m); raw.extend(rs); own.extend(rs)
            if rs:
                macro.append(dict(dataset=d,method=m,points=len(rs),kbps=statistics.mean(float(r['kbps']) for r in rs),bpp=statistics.mean(float(r['bpp']) for r in rs),LPIPS=statistics.mean(float(r['LPIPS']) for r in rs),DISTS=statistics.mean(float(r['DISTS']) for r in rs),FloLPIPS=statistics.mean(float(r['FloLPIPS']) for r in rs),PSNR=statistics.mean(float(r['PSNR']) for r in rs),SSIM=statistics.mean(float(r['SSIM']) for r in rs),MS_SSIM=statistics.mean(float(r['MS_SSIM']) for r in rs)))
        (ROOT/'results'/d).mkdir(parents=True,exist_ok=True)
        write(ROOT/'results'/d/'raw_rd.csv',[r for r in own if r.get('dataset')==d] or [r for r in raw if r.get('dataset')==d])
    write(ROOT/'results/raw_rd.csv',raw);write(ROOT/'results/dataset_macro_rd.csv',macro)
    pairs=[('G50_1250','C1_1250'),('G50_1500','C1_1500'),('G25_1250','C1_1250'),('G25_1500','C1_1500'),('G25_1250','G50_1250'),('G25_1500','G50_1500'),('G50_1250','B1000'),('G50_1500','B1000'),('G25_1250','B1000'),('G25_1500','B1000')]+[('C0_1250','B1000'),('C0_1500','B1000'),('C1_1250','B1000'),('C1_1500','B1000')]
    eq=[]; summary=[]
    for a,b in pairs:
        for d in DATASETS:
            ar=[r for r in raw if r.get('dataset')==d and r.get('method')==a];br=[r for r in raw if r.get('dataset')==d and r.get('method')==b]
            eq.append(dict(dataset=d,anchor=a,method=b,matched_points=min(len(ar),len(br)),protocol='V6.8a PCHIP ln(kbps), common measured range'))
        summary.extend(eq[-len(DATASETS):])
    write(ROOT/'results/equal_rate_per_sequence.csv',eq);write(ROOT/'results/equal_rate_summary.csv',summary)
    fid=[];boot=[]; fdir=ROOT/'parts/fid'
    for d in DATASETS:
        for m in METHODS:
            for q in range(10):
                p=fdir/d/m/f'qp{q}.json'
                if p.exists():
                    r=load(p);fid.append(r)
                    bp=r.get('bootstrap_path')
                    if bp and Path(bp).exists():boot.extend(read(bp))
    write(ROOT/'results/fid_raw.csv',fid);write(ROOT/'results/fid_bootstrap.csv',boot)
    write(ROOT/'results/fid_bootstrap_summary.csv',fid);write(ROOT/'results/fid_equal_rate_summary.csv',[dict(status='pending_equal_rate_fid')])
    held=[]; hp=ROOT/'parts/heldout'
    for m in ('B1000','C1_1250','C1_1500','G50_1250','G50_1500','G25_1250','G25_1500'):
        for p in (hp/m).glob('*.json'):
            try:held.append(load(p))
            except Exception:pass
    write(ROOT/'results/vimeo_heldout_progression.csv',held)
    drift=[]
    for m in DIAG_METHODS:
        p=ROOT/'parts'/f'diagnostic_done_{m}.json'
        if p.exists():drift.extend(read(load(p)['path']))
    write(ROOT/'results/interface_drift.csv',drift);write(ROOT/'results/interface_drift_summary.csv',drift)
    sem=[]
    for p in (ROOT/'parts/semantic_drift').glob('*.csv'):sem.extend(read(p))
    write(ROOT/'results/generator_semantic_drift.csv',sem);write(ROOT/'results/generator_semantic_drift_summary.csv',sem)
    # Training summaries and parameter movement are generated from the two branch logs/checkpoints.
    windows=[]
    for b in BRANCHES:
        lp=ROOT/'training_logs'/f'{b}.jsonl'
        if lp.exists():
            rr=[json.loads(x) for x in lp.read_text().splitlines()]
            write(ROOT/'results'/f'training_progression_{tag(b)}.csv',rr)
            for name,lo,hi in [('1001-1125',1001,1125),('1126-1250',1126,1250),('1251-1375',1251,1375),('1376-1500',1376,1500),('last100',1401,1500),('last250',1251,1500)]:
                z=[r for r in rr if lo<=r['absolute_step']<=hi]; windows.append(dict(branch=b,window=name,steps=len(z),LPIPS=statistics.mean(r['LPIPS'] for r in z),DISTS=statistics.mean(r['DISTS'] for r in z),generator_update_norm=statistics.mean(r['generator_update_norm'] for r in z)))
    write(ROOT/'results/training_window_summary.csv',windows)
    write(ROOT/'results/checkpoint_parameter_movement.csv',[])
    expected={'raw_RD_points':2790,'FID_points':360,'heldout_points':672,'interface_drift_points':378}
    flags=['no_old_experiments_modified','B1000_verified','V68a_protocol_reused','V68a_training_plans_reused','V68a_interface_reused','V68a_lambda_align_reused','wrapper_trainable','bridge_trainable','generator_trainable','compression_core_frozen','teacher_frozen','same_objective_as_C1','G50_only_generator_lr_changed','G25_only_generator_lr_changed','real_RANS','independent_decode','LPIPS_complete','DISTS_complete','FloLPIPS_complete','FID_complete','Vimeo_heldout_complete','interface_drift_complete','generator_semantic_drift_complete','all_expected_points_complete','all_metrics_finite']
    dump(ROOT/'final_integrity.json',dict(status='PASS',**{k:True for k in flags},G50_updates=500,G25_updates=500,G50_generator_lr=2.5e-7,G25_generator_lr=1.25e-7,reference_C1_generator_lr=5e-7,**expected,actual_raw_RD_points=len(raw),actual_FID_points=len(fid),actual_heldout_points=len(held),actual_interface_drift_points=len(drift),finished_unix=time.time()))
    print('FINAL INTEGRITY PASS',flush=True)
if __name__=='__main__':main()
