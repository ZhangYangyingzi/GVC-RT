import argparse
import math
import numpy as np
from audit_utils import *
from stream_audit import structure
from rd_analysis import compare

SCOPES={'0-3_seen':list(range(4)),'4-9_unseen':list(range(4,10)),'0-9':list(range(10))}

def mean(rows,key):return float(np.mean([float(r[key]) for r in rows]))
def normalized_point(record):
    """Accept cached numeric strings without changing source JSON or tolerances."""
    record=dict(record)
    for key in NUMERIC:
        if isinstance(record[key],bool):raise ValueError(f'{key}: boolean is not a metric')
        value=float(record[key])
        if not math.isfinite(value):raise ValueError(f'{key}: nonfinite metric')
        record[key]=value
    return record

def qp_fields(qps):
    return dict(external_qp=json.dumps(qps),actual_i_qp=json.dumps(qps),actual_p_qp=json.dumps(sorted({q+s for q in qps for s in (0,1,2)})))

def collect(dataset):
    config=load('config.json');rows=[];references={}
    for method in METHODS:
        for i,video in enumerate(videos(dataset)):
            count=64 if dataset=='ulong' else video['frames_evaluated']
            for q in range(10):
                r=normalized_point(load(point_path(dataset,method,i,q)))
                assert (r['method'],r['video_index'],r['external_qp'])==(method,i,q)
                assert r['checkpoint_sha256']==config['checkpoints'][method]['sha256']
                assert r['metric_audit_sha256']==config['metric_audit_sha256']
                assert r['manifest_sha256']==config['ulong_manifest_sha256' if dataset=='ulong' else 'uvg_manifest_sha256']
                assert int(r['num_frames'])==count and int(r['num_transitions'])==count-1
                assert r['decode_status']=='PASS' and all(truth(r[k]) for k in ('independent_decode_pass','metric_decode_pass','state_sync_pass','position_decode_pass'))
                assert r['compression_hash_before']==r['compression_hash_after']
                for k in ('bitstream','feature','frame_metrics'):assert sha(r[k+'_path'])==r[k+'_sha256']
                assert Path(r['bitstream_path']).stat().st_size==int(r['real_bytes'])==int(r['bytes_consumed'])
                header=structure(Path(r['bitstream_path']),q,count);frame=read(r['frame_metrics_path'])
                assert len(frame)==count and sum(int(f['real_bits']) for f in frame)==r['real_bytes']*8
                for f,h in zip(frame,header):
                    assert int(f['frame'])==h['frame'] and int(f['actual_qp'])==h['actual_qp'] and int(f['external_qp'])==q
                    assert all(math.isfinite(float(f[k])) for k in ('LPIPS','DISTS','PSNR','SSIM','MS_SSIM','real_bits'))
                for k in ('LPIPS','DISTS','SSIM','MS_SSIM'):assert abs(mean(frame,k)-r[k])<2e-6
                for k in NUMERIC:assert math.isfinite(float(r[k]))
                assert r['kbps']>0 and r['bpp']>0
                transitions=read(Path(r['feature_path']).with_suffix('.transitions.csv'))
                assert [(int(t['from_frame']),int(t['to_frame'])) for t in transitions]==[(n,n+1) for n in range(count-1)]
                assert all(math.isfinite(float(t['FloLPIPS'])) for t in transitions)
                assert abs(mean(transitions,'FloLPIPS')-r['FloLPIPS'])<1e-10
                with np.load(r['feature_path']) as f:
                    assert f['real'].shape==f['reconstruction'].shape==(count,2048)
                    assert np.isfinite(f['real']).all() and np.isfinite(f['reconstruction']).all()
                    if i in references:assert np.allclose(references[i],f['real'],rtol=1e-5,atol=1e-6)
                    else:references[i]=f['real'].copy()
                rows.append(r)
    assert len(rows)==len(videos(dataset))*40
    return rows

def pooled(rows,dataset):
    fid=module('v51_fid_report',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py')
    summary=[];flows=[];fid_rows=[]
    for method in METHODS:
        for q in range(10):
            own=[r for r in rows if r['method']==method and r['external_qp']==q]
            real=[];recon=[]
            for r in own:
                with np.load(r['feature_path']) as f:real.append(f['real']);recon.append(f['reconstruction'])
                flows.append(dict(dataset=r['dataset'],method=method,video_index=r['video_index'],scope='video',
                    external_qp=q,actual_i_qp=q,actual_p_qp=json.dumps(r['actual_p_qp']),num_transitions=r['num_transitions'],FloLPIPS=r['FloLPIPS']))
            a=np.concatenate(real);b=np.concatenate(recon);value=fid.low_rank_fid(a,b);assert math.isfinite(value)
            count=sum(r['num_frames'] for r in own);nt=sum(r['num_transitions'] for r in own)
            assert a.shape==b.shape==(count,2048)
            flow=sum(r['FloLPIPS']*r['num_transitions'] for r in own)/nt
            row=dict(dataset='Fresh_U_Long' if dataset=='ulong' else 'UVG',method=method,external_qp=q,actual_i_qp=q,
                actual_p_qp=json.dumps(sorted({v for r in own for v in r['actual_p_qp']})),
                mean_real_kbps=mean(own,'kbps'),mean_bpp=mean(own,'bpp'),kbps=mean(own,'kbps'),
                **{k:mean(own,k) for k in ('LPIPS','DISTS','PSNR','SSIM','MS_SSIM')},FloLPIPS=flow,FID=value,
                num_videos=len(own),num_frames=count,FID_num_samples=count,num_transitions=nt,
                metric_aggregation='spatial arithmetic mean over videos; FloLPIPS arithmetic mean over all within-sequence transitions; FID pooled frames')
            summary.append(row)
            fid_rows.append({k:row[k] for k in ('dataset','method','external_qp','actual_i_qp','actual_p_qp','mean_real_kbps','FID','FID_num_samples')})
            flows.append(dict(dataset=row['dataset'],method=method,video_index='',scope='pooled_QP',external_qp=q,
                actual_i_qp=q,actual_p_qp=row['actual_p_qp'],num_transitions=nt,FloLPIPS=flow))
    write(f'{dataset}_all_qp_summary.csv',summary);write(f'{dataset}_fid.csv',fid_rows);write(f'{dataset}_flolpips.csv',flows)
    return summary

def analysis(summary,dataset):
    equal=[];bd=[];cross=[];pair=[];general=[];monotone=[]
    for method in METHODS:
        own=sorted((r for r in summary if r['method']==method),key=lambda r:r['external_qp'])
        for metric in METRICS:
            rate_bad=[b['external_qp'] for a,b in zip(own,own[1:]) if b['kbps']<=a['kbps']]
            metric_bad=[b['external_qp'] for a,b in zip(own,own[1:]) if b[metric]>=a[metric]]
            monotone.append(dict(dataset=own[0]['dataset'],method=method,metric=metric,**qp_fields(list(range(10))),
                rate_monotone=not rate_bad,metric_monotone=not metric_bad,violating_qps=json.dumps(sorted(set(rate_bad+metric_bad))),
                rate_violating_qps=json.dumps(rate_bad),metric_violating_qps=json.dumps(metric_bad),
                expected_progression='strictly increasing bitrate and strictly decreasing distortion with external QP; all points preserved'))
    for scope,qps in SCOPES.items():
        scoped=[r for r in summary if r['external_qp'] in qps]
        for anchor,candidates in [('original',METHODS),('clip4_control',('clip8',))]:
            a=sorted((r for r in scoped if r['method']==anchor),key=lambda r:r['external_qp'])
            for method in candidates:
                c=sorted((r for r in scoped if r['method']==method),key=lambda r:r['external_qp'])
                record=dict(dataset=a[0]['dataset'],method=method,anchor=anchor,external_qp_scope=scope,**qp_fields(qps),
                    mean_bitrate_ratio=mean(c,'kbps')/mean(a,'kbps'),
                    mean_same_QP_rate_ratio=float(np.mean([y['kbps']/x['kbps'] for x,y in zip(a,c)])))
                for metric in METRICS:
                    e,b=compare(a,c,metric,anchor,method)
                    tag=dict(dataset=a[0]['dataset'],external_qp_scope=scope,**qp_fields(qps))
                    e.update(tag);b.update(tag);equal.append(e);bd.append(b)
                    record.update({metric+'_equal_rate_delta':e['mean_equal_rate_delta'],metric+'_equal_rate_status':e['status'],
                        metric+'_equal_rate_reason':e['reason'],metric+'_better_fraction':e['fraction_of_common_rate_range_better'],
                        metric+'_common_rate_min_kbps':e['common_rate_min_kbps'],metric+'_common_rate_max_kbps':e['common_rate_max_kbps'],
                        metric+'_BD_rate_percent':b['BD_rate_percent'],metric+'_BD_status':b['status'],metric+'_BD_reason':b['reason']})
                if anchor=='original':
                    cross.append(record)
                    general.append(dict(dataset=a[0]['dataset'],method=method,external_qp_scope=scope,**qp_fields(qps),
                        mean_real_kbps=mean(c,'kbps'),mean_same_QP_rate_ratio=record['mean_same_QP_rate_ratio'],
                        ratio_of_mean_kbps=record['mean_bitrate_ratio'],
                        **{m:mean(c,m) for m in METRICS},
                        **{m+'_same_QP_delta':float(np.mean([y[m]-x[m] for x,y in zip(a,c)])) for m in METRICS},
                        FID_scope_aggregation='mean of per-QP dataset-pooled FIDs; never mean of per-video FIDs',
                        seen_unseen_definition='relative only to wrapper fine-tuning QP set'))
                else:pair.append(record)
    write(f'{dataset}_equal_rate_summary.csv',equal);write(f'{dataset}_bd_rate.csv',bd)
    write(f'parts/{dataset}_cross_dataset_generalization.csv',cross);write(f'parts/{dataset}_clip8_vs_clip4_allqp.csv',pair)
    write(f'parts/{dataset}_qp_generalization_summary.csv',general);write(f'parts/{dataset}_all_qp_monotonicity_audit.csv',monotone)
    if dataset=='ulong':write('ulong_seen_unseen_qp_summary.csv',[r for r in general if r['external_qp_scope']!='0-9'])
    for name in ('cross_dataset_generalization','clip8_vs_clip4_allqp','qp_generalization_summary','all_qp_monotonicity_audit'):
        merged=[]
        for d in ('ulong','uvg'):
            if (ROOT/f'parts/{d}_{name}.csv').exists():merged.extend(read(f'parts/{d}_{name}.csv'))
        write(name+'.csv',merged)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('ulong','uvg'),required=True);args=p.parse_args()
    rows=collect(args.dataset);write(f'{args.dataset}_rd_points.csv',rows)
    summary=pooled(rows,args.dataset);analysis(summary,args.dataset)
    if args.dataset=='ulong':
        old=read(V5/'final_qp_summary.csv')
        for r in summary:
            if r['external_qp']>=4:continue
            a=next(v for v in old if v['method']==r['method'] and int(v['qp'])==r['external_qp'])
            for key in ('kbps','LPIPS','DISTS','FloLPIPS','FID','PSNR','SSIM','MS_SSIM'):assert abs(float(a[key])-r[key])<2e-6
    dump(f'parts/{args.dataset}_report_done.json',dict(status='PASS',num_points=len(rows),num_qp_summary_rows=len(summary),
        summary_sha256=sha(ROOT/f'{args.dataset}_all_qp_summary.csv')))
    print(args.dataset,'REPORT PASS',len(rows),'points',flush=True)

if __name__=='__main__':main()
