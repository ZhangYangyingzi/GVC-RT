import argparse
import math
import sys
from pathlib import Path
import numpy as np
from v5_utils import ROOT,V41,METHODS,BRANCHES,sha,load,dump,read,write,module,truth,checkpoint
from rd_analysis import METRICS,compare
sys.path.insert(0,str(V41))
base=module('v5_metric_reporting_source',V41/'report.py')
base.ROOT=ROOT;base.load_json=load;base.read=read;base.write=write;base.sha=sha

def mean(rows,key):return float(np.mean([float(r[key]) for r in rows]))

def collect(split):
    n,count=(6,32) if split=='validation' else (8,64);rows=[];protocol=sha(ROOT/'metric_implementation_audit.json')
    manifest=sha(ROOT/('validation_manifest.json' if split=='validation' else 'test_manifest.json'))
    frozen=load('initialization_audit.json')['branches']['clip4_control']['hashes']['compression']
    digests={m:sha(checkpoint(m)) if checkpoint(m) else '' for m in METHODS}
    for method in METHODS:
        for index in range(n):
            for qp in range(4):
                r=load(f'parts/{split}/{method}/video_{index}_qp{qp}.json')
                assert (r['method'],r['split'],r['video_index'],r['qp'])==(method,split,index,qp)
                assert int(r['num_frames'])==count and int(r['num_transitions'])==count-1
                assert r['metric_audit_sha256']==protocol and r['manifest_sha256']==manifest
                assert r['checkpoint_sha256']==digests[method]
                assert r['compression_hash_before']==r['compression_hash_after']==frozen
                assert r['decode_status']=='PASS' and all(truth(r[k]) for k in ('state_sync_pass','independent_decode_pass','metric_decode_pass','position_decode_pass'))
                assert sha(r['bitstream_path'])==r['bitstream_sha256']
                assert Path(r['bitstream_path']).stat().st_size==int(r['real_bytes'])==int(r['bytes_consumed'])
                assert sha(r['feature_path'])==r['feature_sha256'] and sha(r['frame_metrics_path'])==r['frame_metrics_sha256']
                for k in ('kbps','bpp','LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM'):
                    r[k]=float(r[k]);assert math.isfinite(r[k])
                assert r['kbps']>0 and r['bpp']>0
                frame=read(r['frame_metrics_path']);assert [int(x['frame']) for x in frame]==list(range(count))
                assert sum(int(x['real_bits']) for x in frame)==int(r['real_bytes'])*8
                assert all(math.isfinite(float(x[k])) for x in frame for k in ('real_bits','LPIPS','DISTS','PSNR','SSIM','MS_SSIM'))
                for k in ('LPIPS','DISTS'):assert abs(mean(frame,k)-r[k])<2e-6
                transitions=read(Path(r['feature_path']).with_suffix('.transitions.csv'))
                assert [(int(x['from_frame']),int(x['to_frame'])) for x in transitions]==[(i,i+1) for i in range(count-1)]
                assert all(math.isfinite(float(x['FloLPIPS'])) for x in transitions)
                assert abs(mean(transitions,'FloLPIPS')-r['FloLPIPS'])<1e-10
                rows.append(r)
    assert len(rows)==len(METHODS)*n*4
    return rows

def temporal_positions(rows,split):
    output=[];count=32 if split=='validation' else 64
    groups={name:set(int(x) for x in arr) for name,arr in zip(('early','middle','late'),np.array_split(np.arange(1,count),3))}
    raw=[]
    for r in rows:
        for frame in read(r['frame_metrics_path']):
            index=int(frame['frame'])
            if index==0:continue
            raw.append(dict(method=r['method'],qp=int(r['qp']),video_index=r['video_index'],frame_index=index,
                band=next(name for name,indices in groups.items() if index in indices),
                real_bits=int(frame['real_bits']),LPIPS=float(frame['LPIPS']),DISTS=float(frame['DISTS'])))
    for method in METHODS:
        for qp in (0,1,2,3,'all'):
            own=[r for r in raw if r['method']==method and (qp=='all' or r['qp']==qp)]
            for position in list(range(1,count))+['early','middle','late']:
                subset=[r for r in own if r['frame_index']==position] if isinstance(position,int) else [r for r in own if r['band']==position]
                output.append(dict(method=method,qp=qp,position=position,num_P_frames=len(subset),
                    frame_index_min=min(r['frame_index'] for r in subset),frame_index_max=max(r['frame_index'] for r in subset),
                    mean_real_bits=mean(subset,'real_bits'),mean_LPIPS=mean(subset,'LPIPS'),mean_DISTS=mean(subset,'DISTS'),
                    position_definition='continuous distance from only I-frame reset at frame0; DPB size1; np.array_split thirds',
                    bits_definition='actual serialized P-frame bytes including NAL/QP headers; initial SPS belongs to I-frame'))
    write(f'{split}_temporal_position_summary.csv',output)

def rd_tables(summary):
    equal=[];bd=[]
    for anchor,candidates in [('original',('v41_20000','clip4_control','clip8')),('clip4_control',('clip8',))]:
        a=[r for r in summary if r['method']==anchor]
        for method in candidates:
            c=[r for r in summary if r['method']==method]
            for metric in METRICS:
                e,b=compare(a,c,metric,anchor,method);equal.append(e);bd.append(b)
    return equal,bd

def pair(rows,equal):
    a=[r for r in rows if r['method']=='clip4_control'];c=[r for r in rows if r['method']=='clip8']
    result=dict(clip8_rate_change_vs_clip4_percent=100*(mean(c,'kbps')/mean(a,'kbps')-1),
                rate_aggregation='ratio of mean real kbps over matched video/QP points')
    for metric in METRICS:
        r=next(r for r in equal if r['anchor']=='clip4_control' and r['method']=='clip8' and r['metric']==metric)
        result.update({'mean_equal_rate_delta_'+metric:r['mean_equal_rate_delta'],metric+'_status':r['status'],metric+'_reason':r['reason'],metric+'_better_fraction':r['fraction_of_common_rate_range_better']})
    return result

def result_gate(comparison,split):
    result=dict(split=split,clip8_rate_change_vs_clip4=comparison['clip8_rate_change_vs_clip4_percent'],rate_change_units='percent',
                checkpoint_selection=False,automatic_followup_experiment=False)
    for metric,label in [('FloLPIPS','clip8_temporal_improvement'),('LPIPS','clip8_lpips_nonworse'),('DISTS','clip8_dists_nonworse'),('FID','clip8_fid_nonworse')]:
        valid=comparison[metric+'_status']=='valid'
        value=comparison['mean_equal_rate_delta_'+metric]
        result[label]=(value<0 if metric=='FloLPIPS' else value<=0) if valid else None
        result['mean_equal_rate_delta_'+metric+'_vs_clip4']=value if valid else None
        if not valid:result[metric+'_invalid_reason']=comparison[metric+'_reason']
    result['status']='MEASURED' if all(comparison[m+'_status']=='valid' for m in METRICS) else 'PARTIALLY_INVALID'
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('validation','final'),required=True);args=p.parse_args();split=args.split
    if split=='final':assert load('parts/validation_done.json')['validation_sha256']==sha(ROOT/'checkpoint_validation.csv')
    rows=collect(split);write('checkpoint_validation.csv' if split=='validation' else 'final_rd_points.csv',rows)
    fids=base.metrics(rows,split);summary=base.summarize(rows)
    for r in summary:
        f=next(x for x in fids if x['method']==r['method'] and x['qp']==r['qp'])
        assert abs(r['kbps']-f['mean_real_kbps'])<1e-10
        r['FID']=f['FID'];r['FID_num_samples']=f['num_real_samples']
    write(f'{split}_qp_summary.csv',summary)
    equal,bd=rd_tables(summary)
    write(f'{split}_equal_rate_summary.csv',equal);write(f'{split}_perceptual_bd_rate.csv',bd)
    comparison=pair(rows,equal);write(f'{split}_clip8_vs_clip4_summary.csv',[comparison])
    temporal_positions(rows,split)
    ablation=[];same=[]
    for method in METHODS:
        own=[r for r in rows if r['method']==method]
        record=dict(split=split,method=method,clip_length=BRANCHES[method]['clip_length'] if method in BRANCHES else (4 if method=='v41_20000' else ''),
            optimizer_updates=BRANCHES[method]['updates'] if method in BRANCHES else 0,
            optimized_P_frames=21000 if method in BRANCHES else 0,mean_real_kbps=mean(own,'kbps'),
            budget_scope='new V5-A continuation only',**{'mean_'+m:mean(own,m) for m in METRICS if m!='FID'})
        for m in METRICS:
            if method=='original':
                record.update({m+'_BD_rate_vs_Original':0,m+'_BD_status':'anchor',m+'_BD_reason':'','mean_equal_rate_delta_'+m:0,m+'_better_fraction':0})
            else:
                er=next(r for r in equal if r['anchor']=='original' and r['method']==method and r['metric']==m)
                br=next(r for r in bd if r['anchor']=='original' and r['method']==method and r['metric']==m)
                record.update({m+'_BD_rate_vs_Original':br['BD_rate_percent'],m+'_BD_status':br['status'],m+'_BD_reason':br['reason'],
                    'mean_equal_rate_delta_'+m:er['mean_equal_rate_delta'],m+'_equal_rate_status':er['status'],m+'_better_fraction':er['fraction_of_common_rate_range_better']})
        if method=='clip8':record.update({'vs_clip4_'+k:v for k,v in comparison.items()})
        ablation.append(record)
        if method!='original':
            for qp in (0,1,2,3,'all'):
                a=[r for r in rows if r['method']=='original' and (qp=='all' or r['qp']==qp)]
                c=[r for r in own if qp=='all' or r['qp']==qp]
                same.append(dict(method=method,qp=qp,mean_rate_change_percent=100*(mean(c,'kbps')/mean(a,'kbps')-1),
                    **{m+'_change':mean(c,m)-mean(a,m) for m in METRICS if m!='FID'}))
    write(f'{split}_same_qp_summary.csv',same);write(f'{split}_clip_ablation_summary.csv',ablation)
    other='final' if split=='validation' else 'validation'
    existing=read(f'{other}_clip_ablation_summary.csv') if (ROOT/f'{other}_clip_ablation_summary.csv').exists() else []
    write('clip_ablation_summary.csv',existing+ablation)
    gate=result_gate(comparison,split);dump(f'parts/{split}_result_gate.json',gate)
    if split=='final':
        gate['validation_reference']=load('parts/validation_result_gate.json');dump('v5a_result_gate.json',gate)
    done=dict(status='PASS',split=split,rows=len(rows),fid_rows=len(fids),all_four_methods=True,checkpoint_selection=False)
    if split=='validation':done['validation_sha256']=sha(ROOT/'checkpoint_validation.csv')
    dump(f'parts/{split}_done.json',done)
    print(split.upper(),'REPORT PASS',len(rows),flush=True)

if __name__=='__main__':main()
