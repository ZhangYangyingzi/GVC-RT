"""Audited dataset pooling and measured-rate comparisons, V5-A.2."""
import argparse
import math
import numpy as np
from audit_utils import *
from stream_audit import structure
from rd_analysis import compare

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
    assert len(rows)==len(videos(dataset))*30
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('ulong','uvg'),required=True);args=parser.parse_args()
    dataset=args.dataset;rows=collect(dataset);config=load('config.json')
    audit=load('canonical_source_audit.json');audit_sha=sha(ROOT/'canonical_source_audit.json')
    rate_checks=[]
    for r in rows:
        source=next(v for v in audit['records'] if v['dataset']==dataset and v['video_index']==r['video_index'])
        assert r['protocol']=='V5-A.2-30fps-RGB' and r['fresh_real_RANS'] and r['fps']==30 and r['num_frames']==64
        assert r['canonical_audit_sha256']==audit_sha and r['source_frame_hash']==source['whole_sequence_rgb_hash']
        assert r['canonical_loader_sha256']==sha(ROOT/'canonical_loader.py')
        assert Path(r['bitstream_path']).resolve().is_relative_to(ROOT/'bitstreams'/dataset)
        assert r['physical_gpu'] in (4,5,6,7)
        bits=r['real_bytes']*8
        assert math.isclose(r['bits_per_frame'],bits/64,rel_tol=1e-12)
        assert math.isclose(r['bpp'],bits/(64*1920*1080),rel_tol=1e-12)
        assert math.isclose(r['kbps'],bits/(64/30)/1000,rel_tol=1e-12)
    write(f'{dataset}_rd_points.csv',rows)
    summary=pooled(rows,dataset)
    for s in summary:
        own=[r for r in rows if r['method']==s['method'] and r['external_qp']==s['external_qp']]
        s['mean_bits_per_frame']=mean(own,'bits_per_frame')
        s['mean_real_bytes']=mean(own,'real_bytes')
        s['canonical_fps']=30.0
        total_bytes=sum(r['real_bytes'] for r in own)
        bpf=total_bytes*8/(len(own)*64)
        record=dict(dataset=dataset,method=s['method'],external_qp=s['external_qp'],real_bytes=total_bytes,
                    real_bytes_definition='sum across dataset videos at method x QP',num_videos=len(own),num_frames=len(own)*64,
                    bits_per_frame=bpf,bpp=s['mean_bpp'],kbps=s['mean_real_kbps'],expected_kbps=bpf*30/1000,
                    absolute_error=abs(s['mean_real_kbps']-bpf*30/1000),fps=30.0)
        record['pass']=math.isclose(record['kbps'],record['expected_kbps'],rel_tol=1e-12,abs_tol=1e-10)
        assert record['pass'];rate_checks.append(record)
    write(f'{dataset}_all_qp_summary.csv',summary)
    write(f'parts/{dataset}_rate_sanity.csv',rate_checks)
    equal=[];bd=[]
    for anchor,candidate in [('original','v41_20000'),('original','clip8'),('v41_20000','clip8')]:
        a=[r for r in summary if r['method']==anchor];c=[r for r in summary if r['method']==candidate]
        for metric in METRICS:
            e,b=compare(a,c,metric,anchor,candidate)
            e.update(dataset=dataset,candidate=candidate,common_rate_min=e['common_rate_min_kbps'],
                     common_rate_max=e['common_rate_max_kbps'],better_fraction=e['fraction_of_common_rate_range_better'])
            b.update(dataset=dataset,candidate=candidate)
            equal.append(e);bd.append(b)
    write(f'{dataset}_equal_rate_summary.csv',equal);write(f'{dataset}_bd_rate.csv',bd)
    if dataset=='uvg':
        table=[]
        for r in rows:
            original=next(a for a in rows if a['video_index']==r['video_index'] and a['external_qp']==r['external_qp'] and a['method']=='original')
            item={k:r[k] for k in ('sequence_name','method','external_qp','kbps','bpp','LPIPS','DISTS','FloLPIPS')}
            item.update({k+'_same_QP_delta_vs_original':r[k]-original[k] for k in ('kbps','bpp','LPIPS','DISTS','FloLPIPS')})
            table.append(item)
        write('uvg_per_sequence_summary.csv',table)
    dump(f'parts/{dataset}_report_done.json',dict(status='PASS',num_points=len(rows),num_qp_summary_rows=len(summary),
                                               summary_sha256=sha(ROOT/f'{dataset}_all_qp_summary.csv')))
    print(dataset,'REPORT PASS',len(rows),'points',flush=True)

if __name__=='__main__':main()

