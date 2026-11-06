"""Report measured precision differences without mixing historical and matched scores."""
from common_s2 import *
from reuse import read_csv
from evaluate import METRICS

def update():
    records=[];deltas=[];completed=[]
    for split,label in [('development','B_historical_development'),('final','B_historical_final')]:
        root=ROOT/'evaluation'/evaluation_id(split,candidate='matched_baseline')
        if not (root/'integrity.json').exists():continue
        assert load(root/'integrity.json')['status']=='PASS'
        historical=read_csv(ROOT/f'evaluation/reused/{label}.csv')
        old={(r['sequence'],int(r['QP'])):r for r in historical};new=read_csv(root/'metrics.csv')
        assert len(new)==len(historical)==(96 if split=='development' else 135)
        for r in new:
            key=(r['sequence'],int(r['QP']));h=old[key];point=load(root/f'points/{r["domain"]}_{r["sequence"]}_qp{r["QP"]}.json')
            assert point['candidate']=='matched_baseline' and point['input_stream_reused'] and not point['fresh_encode']
            assert point['baseline_sha256']==BASELINE_SHA and r['bitstream_sha256']==h['bitstream_sha256'] and int(r['real_bytes'])==int(h['real_bytes'])
            metric_differences={m:float(r[m])-float(h[m]) for m in METRICS}
            row=dict(split=split,sequence=r['sequence'],QP=int(r['QP']),historical_model='B1500_historical',matched_model='B1500_matched_FP32',
                same_stream=True,same_bytes=True,same_reconstruction=point['reconstruction_sha256']==point['historical_reconstruction_sha256'],
                all_metrics_exact_equal=all(x==0 for x in metric_differences.values()),metric_differences=metric_differences,
                new_point_sha256=sha(root/f'points/{r["domain"]}_{r["sequence"]}_qp{r["QP"]}.json'))
            records.append(row)
            for m,d in metric_differences.items():deltas.append(dict(split=split,sequence=r['sequence'],QP=int(r['QP']),metric=m,historical=float(h[m]),matched=float(r[m]),matched_minus_historical=d))
        completed.append(split)
    assert 'development' in completed,'MATCHED_DEVELOPMENT_BASELINE_REQUIRED'
    write_csv(ROOT/'precision_baseline_deltas.csv',deltas)
    summary=dict(status='COMPLETE' if 'final' in completed else 'DEVELOPMENT_COMPLETE_FINAL_PENDING',baseline_sha256=BASELINE_SHA,
        historical_precision='TCG and deployment nonTCG FP16',matched_precision='TCG parameters/internal FP32, output restored to incoming FP16; remaining deployment modules unchanged',
        historical_normalization_used=False,normalization_baseline='B1500_matched_FP32',no_baseline_optimizer_or_updates=True,
        completed_splits=completed,points=len(records),strictly_same_reconstruction_and_metrics=all(r['same_reconstruction'] and r['all_metrics_exact_equal'] for r in records),
        preserve_both_precision_versions=True,records=records,time=time.time())
    dump(ROOT/'precision_baseline_comparison.json',summary);return summary
if __name__=='__main__':env();update()
