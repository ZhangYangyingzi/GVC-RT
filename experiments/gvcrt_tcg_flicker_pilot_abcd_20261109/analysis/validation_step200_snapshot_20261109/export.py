"""Read-only numerical snapshot of complete paired step200 development data."""
from pathlib import Path
import sys
EXP=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(EXP))
from common_s2 import *
import statistics

def main():
    out=Path(__file__).resolve().parent
    metrics=['LPIPS','DISTS','FloLPIPS','PSNR','MS_SSIM','MCResidual_RGB_RAFT_pixelcenter_v1_PtoP','MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC','bpp','kbps']
    expected={(p['sequence'],q) for p in load(ROOT/'manifests/quick_development.json') for q in [0,5,9]}
    point_rows=[];sources=[];by_model={}
    for group in ['A2000',*GROUPS]:
        step=0 if group=='A2000' else 200
        base=ROOT/f'evaluation/{group}/step_{step:04d}'
        assert load(base/'completion.json')['status']=='PASS'
        rows={}
        for path in (base/'points').glob('*.json'):
            record=load(path);r=record['result'];key=(r['sequence'],r['QP'])
            assert record['status']=='PASS' and key in expected and key not in rows
            assert r['frames']==64 and r['transitions']==63 and record['frozen_components_unchanged']
            assert record['identity']['implementation_SHA256']==sha(ROOT/'source_manifest.json')
            assert all(math.isfinite(r[k]) for k in metrics)
            for name,h in record['record_SHA256'].items():assert sha(ROOT/name)==h
            if group!='A2000':
                reference=by_model['A2000'][key]
                assert r['real_bytes']==reference['real_bytes'] and r['bitstream_SHA256']==reference['bitstream_SHA256'] and r['bpp']==reference['bpp']
                assert r['source_RGB_SHA256']==reference['source_RGB_SHA256'] and r['codec_trace_SHA256']==reference['codec_trace_SHA256']
            rows[key]=r;point_rows.append(r);sources.append(dict(path=str(path.relative_to(ROOT)),SHA256=sha(path)))
        assert set(rows)==expected;by_model[group]=rows
    aggregates=[]
    for group,records in by_model.items():
        for q in [0,5,9]:
            values=[r for (sequence,qp),r in records.items() if qp==q]
            ref=[r for (sequence,qp),r in by_model['A2000'].items() if qp==q]
            row=dict(model=group,step=0 if group=='A2000' else 200,QP=q,videos=12)
            for key in metrics:
                a=statistics.mean(r[key] for r in values);b=statistics.mean(r[key] for r in ref);row[key]=a
                row[key+'_delta']=a-b
                if key!='PSNR':row[key+'_relative_percent']=100*(a/b-1)
            aggregates.append(row)
    write_csv(out/'point_metrics.csv',point_rows);write_csv(out/'summary_by_group_qp.csv',aggregates)
    history=load(ROOT/'screening_history.json')['history']
    screen={g:next(r for r in history[g] if r['step']==200) for g in GROUPS}
    dump(out/'screening_step200.json',screen)
    status=load(ROOT/'status.json')
    dump(out/'audit.json',dict(status='PASS',scope='Read-only step200 snapshot; full experiment remains RUNNING',groups=4,points_per_group=36,baseline_points=36,
        source_points=sources,source_manifest_SHA256=sha(ROOT/'source_manifest.json'),protocol_SHA256=sha(ROOT/'protocol.json'),export_source_SHA256=sha(Path(__file__)),
        source_and_stream_identity_checks='PASS',frame_record_hash_checks='PASS',no_new_inference_or_optimizer_updates=True,snapshot_time=time.time(),runner_status_at_snapshot=status['status']))
    lines=['# Step200 development snapshot','',f'Snapshot local time: {time.strftime("%Y-%m-%d %H:%M:%S %z")}.','',
        'Four branches each have complete36-point results:12 existing ULong development videos x external QP0/5/9. Reference: fixed A2000 s0.5. Equal video weights within each QP. No cross-QP mean in the table.','',
        'A: TCG+LoRA, frozen Bridge, no explicit history alignment. B: A plus Bridge mlp[3] training. C: A plus fixed causal original-reconstruction RAFT history alignment. D: C plus Bridge mlp[3] training.','',
        '|Group|QP|LPIPS change %|DISTS change %|FloLPIPS change %|Sobel temporal error mean change %|',
        '|---|---:|---:|---:|---:|---:|']
    for row in aggregates:
        if row['model']=='A2000':continue
        vals=[row[k+'_relative_percent'] for k in ['LPIPS','DISTS','FloLPIPS','MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP']]
        lines.append('|'+row['model']+'|'+str(row['QP'])+'|'+'|'.join(f'{x:+.2f}' for x in vals)+'|')
    lines+=['','All four error metrics are lower-is-better. Relative changes are(candidate mean/reference mean-1)*100. The per-QP Sobel mean changes above differ from the preregistered median-of-video-ratios screening statistic.','',
        'Screening at200: '+', '.join(g+'='+screen[g]['screening_status'] for g in GROUPS)+'.','',
        'See summary_by_group_qp.csv for absolute metrics, PSNR dB changes, actual bpp/kbps, both original PWC warping metrics, and point_metrics.csv for sequence/QP detail.','',
        'This is an intermediate fixed-budget comparison. The complete300-step pilot, videos, final integrity and final result publication are pending.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':main()
