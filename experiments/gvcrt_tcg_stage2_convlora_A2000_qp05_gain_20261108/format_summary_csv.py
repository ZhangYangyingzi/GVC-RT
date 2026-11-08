"""Reporting-only: expose per-QP means, deltas and direction counts in one tidy CSV."""
from common_s2 import *

def main():
    env();verify_sources();assert load(ROOT/'final_integrity.json')['status']=='PASS'
    summary=load(ROOT/'summary.json');rows=[]
    for qp in [0,5,9]:
        node=summary['per_QP'][str(qp)]
        for gain,means in node['means'].items():
            for metric,value in means.items():
                rows.append(dict(record_type='mean',external_QP=qp,inference_gain=float(gain),videos=32,metric=metric,mean=value,
                    metric_direction=summary['metric_directions'][metric],weighting='equal videos within this QP'))
        for label,changes in node['changes'].items():
            reference=float(label.rsplit('s',1)[-1])
            for metric,values in changes.items():
                c=node['video_direction_counts'][label][metric];assert c['improved']+c['worsened']+c['equal']==32
                rows.append(dict(record_type='paired_summary',external_QP=qp,inference_gain=.5,reference_gain=reference,videos=32,metric=metric,
                    metric_direction=summary['metric_directions'][metric],weighting='equal videos within this QP',**values,
                    improved_videos=c['improved'],worsened_videos=c['worsened'],equal_videos=c['equal']))
    assert len(rows)==135 and sum(r['record_type']=='mean' for r in rows)==81
    meanrows=[dict(external_QP=int(q),inference_gain=float(g),videos=32,**values) for q,n in summary['per_QP'].items() for g,values in n['means'].items()]
    write_csv(ROOT/'summary_means_by_qp.csv',meanrows)
    write_csv(ROOT/'summary_by_qp.csv',rows)
    note='\n## Summary CSV layout\n\nsummary_by_qp.csv contains81 per-QP/gain/metric mean rows and54 per-QP/reference/metric paired summaries,including absolute/relative changes and improved/worsened/equal video counts. summary_means_by_qp.csv retains the9-row wide mean table. Values come directly from summary.json; no metric or inference result is changed.\n'
    report=ROOT/'REPORT.md'
    if '## Summary CSV layout' not in report.read_text():report.write_text(report.read_text()+note)
    dump(ROOT/'audits/summary_csv_layout.json',dict(status='PASS',scope='Reporting schema only; no inference/training/protocol changes',input_summary_SHA256=sha(ROOT/'summary.json'),
        script_SHA256=sha(ROOT/'format_summary_csv.py'),mean_rows=81,paired_summary_rows=54,QP_pooled_mean=False,
        absolute_relative_changes_and_direction_counts_in_CSV=True,raw_measurements_unchanged=True,time=time.time()))
    dump(ROOT/'reporting_sources.json',dict(inference_sources_SHA256_unchanged=sha(ROOT/'sources.json'),
        added_reporting_script=dict(path='format_summary_csv.py',sha256=sha(ROOT/'format_summary_csv.py')),reason='CSV includes all requested per-QP means,changes and counts',model_or_metric_runtime_changed=False))
    integrity=load(ROOT/'final_integrity.json');integrity.update(summary_CSV_means_changes_counts=True,summary_CSV_rows=135);dump(ROOT/'final_integrity.json',integrity)
    manifest=load(ROOT/'artifact_manifest.json')
    for item in manifest['files']:
        p=ROOT/item['path'];item.update(bytes=p.stat().st_size,sha256=sha(p))
    known={x['path'] for x in manifest['files']}
    for name in ['format_summary_csv.py','summary_means_by_qp.csv','reporting_sources.json','audits/summary_csv_layout.json']:
        if name not in known:
            p=ROOT/name;manifest['files'].append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p)))
    dump(ROOT/'artifact_manifest.json',manifest)
    print('REPORTING_CSV_81_MEANS_54_PAIRED_SUMMARIES_PASS')

if __name__=='__main__':
    with lock('runner.lock'),lock('publication.lock'):main()
