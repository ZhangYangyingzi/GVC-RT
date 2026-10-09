"""Numerical reports only. Performance gates are separate from integrity."""
import argparse,math,sys
import numpy as np
from parallel_utils import *

def collect(root,config):
    rows=[]
    for method in config['methods']:
        for v in config['videos']:
            for q in config['qps']:
                r=load(point(root,v['dataset'],method,v['video_index'],q))
                assert (r['dataset'],r['method'],r['video_index'],r['external_qp'])==(v['dataset'],method,v['video_index'],q)
                assert r['protocol']==config['experiment'] and r['source_rgb_sha256']==v['rgb_sha256']
                assert r['config_sha256']==sha(root/'config.json') and r['physical_gpu'] in config['gpus']
                assert r['frames']==v['frames'] and r['num_transitions']==v['frames']-1
                assert r['independent_decode_pass'] and r['state_sync_pass'] and r['real_RANS']
                assert r['compression_hash_before']==r['compression_hash_after']
                assert r['P']==(config['methods'][method][0] or 'identity') and r['receiver']==(config['methods'][method][1] or 'original')
                if r['P']=='identity':assert r['identity_P_exact'] and r['proxy_max_abs']==0
                for k in ('bitstream','feature','frame_metrics'):assert sha(r[k+'_path'])==r[k+'_sha256']
                assert Path(r['bitstream_path']).resolve().is_relative_to(root/'bitstreams')
                assert Path(r['bitstream_path']).stat().st_size==r['real_bytes']==r['bytes_consumed']
                assert math.isclose(r['bits_per_frame'],r['real_bytes']*8/v['frames'],rel_tol=1e-12)
                assert math.isclose(r['bpp'],r['bits_per_frame']/(v['width']*v['height']),rel_tol=1e-12)
                assert math.isclose(r['kbps'],r['bits_per_frame']*v['rate_fps']/1000,rel_tol=1e-12)
                assert r['metric_crop']==[v['width'],v['height']]
                assert r['processing_canvas']==[(v['width']+63)//64*64,(v['height']+63)//64*64]
                frame=read(r['frame_metrics_path']);assert len(frame)==v['frames']
                assert sum(int(t['real_bits']) for t in frame)==r['real_bytes']*8
                for i,t in enumerate(frame):
                    assert int(t['frame'])==i and int(t['external_qp'])==q
                    assert all(math.isfinite(float(t[k])) for k in SPATIAL)
                for k in ('LPIPS','DISTS','SSIM','MS_SSIM'):assert abs(np.mean([float(t[k]) for t in frame])-r[k])<2e-6
                transitions=read(Path(r['feature_path']).with_suffix('.transitions.csv'))
                assert [(int(t['from_frame']),int(t['to_frame'])) for t in transitions]==[(i,i+1) for i in range(v['frames']-1)]
                assert all(math.isfinite(float(t['FloLPIPS'])) for t in transitions)
                assert abs(np.mean([float(t['FloLPIPS']) for t in transitions])-r['FloLPIPS'])<1e-10
                assert all(math.isfinite(float(r[k])) for k in (*SPATIAL,'FloLPIPS','kbps','bpp'))
                rows.append(r)
    assert len(rows)==config['expected_points']
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=('A','B'),required=True);args=p.parse_args()
    root=exp(args.experiment);config=load(root/'config.json');rows=collect(root,config)
    fid=module('pooled_fid',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py')
    summaries=[];refs_by_video={}
    for dataset in dict.fromkeys(v['dataset'] for v in config['videos']):
        for method in config['methods']:
            for q in config['qps']:
                own=[r for r in rows if (r['dataset'],r['method'],r['external_qp'])==(dataset,method,q)]
                real=[];recon=[]
                for r in own:
                    with np.load(r['feature_path']) as feature:
                        assert feature['real'].shape==feature['reconstruction'].shape==(r['frames'],2048)
                        assert np.isfinite(feature['real']).all() and np.isfinite(feature['reconstruction']).all()
                        key=(dataset,r['video_index'])
                        if key in refs_by_video:assert np.allclose(refs_by_video[key],feature['real'],rtol=1e-5,atol=1e-6)
                        else:refs_by_video[key]=feature['real'].copy()
                        real.append(feature['real']);recon.append(feature['reconstruction'])
                a,b=np.concatenate(real),np.concatenate(recon);count=sum(r['frames'] for r in own)
                assert a.shape==b.shape==(count,2048)
                value=float(fid.low_rank_fid(a,b));assert math.isfinite(value)
                row=dict(dataset=dataset,method=method,QP=q,external_qp=q,kbps=float(np.mean([r['kbps'] for r in own])),
                    mean_bpp=float(np.mean([r['bpp'] for r in own])),**{k:float(np.mean([r[k] for r in own])) for k in SPATIAL},
                    FloLPIPS=sum(r['FloLPIPS']*r['num_transitions'] for r in own)/sum(r['num_transitions'] for r in own),
                    FID=value,num_videos=len(own),num_frames=count,FID_num_samples=count,num_transitions=sum(r['num_transitions'] for r in own))
                if args.experiment=='A':row['mean_real_kbps_20fps']=row['kbps']
                summaries.append(row)
    if args.experiment=='A':
        write(root/'per_video_all_qp.csv',rows);write(root/'virat_480p20_all_qp_summary.csv',summaries)
        gates=[]
        for q in config['qps']:
            a=next(r for r in summaries if r['method']=='original' and r['QP']==q)
            c=next(r for r in summaries if r['method']=='clip8' and r['QP']==q)
            row=dict(external_qp=q,Original_kbps=a['kbps'],clip8_kbps=c['kbps'],rate_change_percent=(c['kbps']/a['kbps']-1)*100,
                rate_pass=c['kbps']<a['kbps'],**{m+'_delta':c[m]-a[m] for m in METRICS},**{m+'_pass':c[m]<=a[m] for m in METRICS})
            row['strict_all_pass']=row['rate_pass'] and all(row[m+'_pass'] for m in METRICS);gates.append(row)
        write(root/'strict_per_qp_gate.csv',gates)
        sys.path.insert(0,str(V52));from rd_analysis import compare
        a=[r for r in summaries if r['method']=='original'];c=[r for r in summaries if r['method']=='clip8']
        eq=[];bd=[]
        for m in METRICS:
            e,b=compare(a,c,m,'original','clip8');eq.append(e);bd.append(b)
        write(root/'equal_rate_summary.csv',eq);write(root/'bd_rate.csv',bd)
        dump(root/'strict_gate_summary.json',dict(strict_pass_qps=sum(r['strict_all_pass'] for r in gates),total_qps=10))
    else:
        for r in summaries:
            anchor=next(a for a in summaries if a['dataset']==r['dataset'] and a['QP']==r['QP'] and a['method']=='ORIGINAL')
            r.update(rate_delta=r['kbps']-anchor['kbps'],**{m+'_delta':r[m]-anchor[m] for m in METRICS})
        write(root/'module_attribution_summary.csv',summaries);write(root/'per_video_attribution.csv',rows)
        stats=[]
        for v in config['videos']:
            for sender in ('v41','clip8'):
                stats.append(load(root/'parts/proxy'/v['dataset']/sender/f"video_{v['video_index']}.json"))
        assert len(stats)==30
        for d in ('ulong','uvg'):
            for sender in ('v41','clip8'):
                own=[s for s in stats if s['scope']=='video' and s['dataset']==d and s['P']==sender]
                keys=[k for k,v in own[0].items() if isinstance(v,(float,int))]
                stats.append(dict(dataset=d,P=sender,scope='dataset',video='ALL',num_videos=len(own),aggregation='equal-weight mean of per-video statistics including per-video p95',
                                  **{k:float(np.mean([r[k] for r in own])) for k in keys}))
        write(root/'proxy_domain_statistics.csv',stats)
    rate=[{k:r[k] for k in ('dataset','method','video','external_qp','real_bytes','frames','bits_per_frame','bpp','kbps','rate_accounting_fps')} for r in rows]
    write(root/'rate_sanity_audit.csv',[dict(r,rate_sanity_pass=True) for r in rate])
    dump(root/'report_integrity.json',dict(status='PASS',num_points=len(rows),summary_rows=len(summaries),FID_pooling='dataset x method x QP; concatenate all frames',
        FID_samples={d:sum(v['frames'] for v in config['videos'] if v['dataset']==d) for d in dict.fromkeys(v['dataset'] for v in config['videos'])}))
    print('REPORT PASS',args.experiment,flush=True)
if __name__=='__main__':main()
