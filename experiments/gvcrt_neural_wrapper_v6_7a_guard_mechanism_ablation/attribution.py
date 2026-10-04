"""Fresh P-only real RANS points; full and Original from the exact evaluation cohort."""
import argparse,math
from v67_io import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--video',type=int,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);frozen();v=next(v for v in sources(a.dataset) if v['video_index']==a.video);assert a.dataset in ('uvg','ulong')
    frames=module('v67_attribution_frames',V62B/'v62b_io.py').frames_for(v);e=engine();cps=load(ROOT/'checkpoint_hashes.json');ev=eval_adapter();rows=[];audits=[]
    for b in ATTR_BRANCHES:
        cfg=dict(experiment='B',methods={b:[b,None]},checkpoints=cps,force_zero_thres=.12);runtime=e.Runtime(cfg,b,torch.device('cuda:0'))
        for q in (0,4,9):
            dest=ROOT/'parts/attribution'/a.dataset/b/f'video_{a.video:02d}_qp{q}.json'
            if dest.exists():r=load(dest)
            else:
                sub=Path(a.dataset)/b/f'video_{a.video:02d}_qp{q}';stream=(ROOT/'attribution_bitstreams'/sub).with_suffix('.bin');feature=(ROOT/'attribution_features'/sub).with_suffix('.npz')
                r,fr=runtime.run(frames,q,stream,feature);ff=dest.with_suffix('.frames.csv');write(ff,fr)
                r.update(status='PASS',dataset=a.dataset,video_index=a.video,sequence=v['name'],method=b,QP=q,actual_qps=[x['actual_qp'] for x in fr],source_rgb_sha256=v['rgb_sha256'],checkpoint_sha256=cps[b]['sha256'],frame_metrics_path=str(ff),frame_metrics_sha256=sha(ff),transitions_path=str(feature.with_suffix('.transitions.csv')),transitions_sha256=sha(feature.with_suffix('.transitions.csv')));dump(dest,r)
            if r.get('reused'):assert sha(r['reused_from'])==r['reused_point_sha256']
            for k in ('bitstream','feature','frame_metrics','transitions'):assert sha(r[k+'_path'])==r[k+'_sha256']
            full=load(point(a.dataset,b,a.video,q));original=load(point(a.dataset,'original',a.video,q));ev.validate(full,v,q,b);ev.validate(original,v,q,'original')
            assert r['source_rgb_sha256']==v['rgb_sha256'] and r['checkpoint_sha256']==cps[b]['sha256']
            assert r['actual_qps']==full['actual_qps'] and r['compression_hash_before']==r['compression_hash_after']==full['compression_hash_before']
            assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass']
            ok=r['real_bytes']==full['real_bytes'] and r['bitstream_sha256']==full['bitstream_sha256'];audits.append(dict(dataset=a.dataset,sequence=v['name'],branch=b,QP=q,status='PASS' if ok else 'FAIL',P_only_sha256=r['bitstream_sha256'],full_sha256=full['bitstream_sha256'],P_only_bytes=r['real_bytes'],full_bytes=full['real_bytes']))
            write(ROOT/'audits'/f'attribution_bitstream_{a.dataset}_{a.video}.csv',audits);assert ok,'P-only/full sender bitstreams differ'
            for metric in METRICS:
                assert all(math.isfinite(x[metric]) for x in (r,full,original))
                rows.append(dict(dataset=a.dataset,sequence=v['name'],video_index=a.video,branch=b,QP=q,metric=metric,metric_direction='lower_is_better' if metric in ('DISTS','LPIPS','FloLPIPS') else 'higher_is_better',receiver_source='adapted_'+b+'_BG',P_effect_originalBG=r[metric]-original[metric],BG_effect_learnedP=full[metric]-r[metric],Full_gap=full[metric]-original[metric],P_only_value=r[metric],full_value=full[metric],original_value=original[metric],formula='P_effect=P_only-Original; BG_effect=Full-P_only; Full_gap=Full-Original'))
            print('ATTRIBUTION',a.dataset,a.video,b,q,flush=True)
        del runtime
    output=ROOT/'parts/attribution'/f'{a.dataset}_{a.video}.csv';write(output,rows);dump(ROOT/'parts'/f'attribution_done_{a.dataset}_{a.video}.json',dict(status='PASS',csv=str(output),sha256=sha(output)))
if __name__=='__main__':main()
