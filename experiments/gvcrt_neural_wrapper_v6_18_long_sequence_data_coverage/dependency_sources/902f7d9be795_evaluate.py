"""Read-only frozen model evaluation through the established real-RANS engine."""
import argparse
import fcntl
import math
import traceback
from v62b_io import *

def visualization(v,q):
    if q not in (0,4,9):return None
    chosen=(v['dataset']=='ulong' and v['video_index']<2) or (v['dataset']=='uvg' and v['name'] in ('Beauty','Jockey','ReadySteadyGo')) or (v['dataset'].startswith('virat') and v['video_index']<2)
    return v['name'] if chosen else None

def validate(r,v,q,method,check_files=True):
    cfg=load(ROOT/'config.json');cp='' if method=='original' else cfg['checkpoints'][method]['sha256']
    assert r['dataset']==v['dataset'] and r['method']==method and r['video_index']==v['video_index'] and r['external_qp']==q
    assert r['checkpoint_sha256']==cp and r['source_rgb_sha256']==v['rgb_sha256']
    assert r['source_sha256']==v['source_sha256'] and r['force_zero_thres']==.12
    assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'] and r['fresh_run']
    assert r['real_bytes']==r['real_RANS_bytes']==r['bytes_consumed']>0
    assert r['total_bits']==r['real_bytes']*8
    assert r['num_frames']==r['frames']==v['frames'] and r['num_transitions']==v['frames']-1
    assert r['rate_accounting_fps']==v['rate_accounting_fps']
    assert math.isclose(r['bits_per_frame'],r['total_bits']/v['frames'],rel_tol=1e-12)
    assert math.isclose(r['kbps'],r['bits_per_frame']*v['rate_accounting_fps']/1000,rel_tol=1e-12)
    assert math.isclose(r['bpp'],r['total_bits']/(v['frames']*v['width']*v['height']),rel_tol=1e-12)
    assert r['metric_crop']==[v['width'],v['height']]
    assert r['processing_canvas']==[(v['width']+63)//64*64,(v['height']+63)//64*64]
    assert r['compression_hash_before']==r['compression_hash_after']
    assert all(math.isfinite(r[k]) for k in (*METRICS,'bpp','kbps'))
    assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:v['frames']]
    if check_files:
        for k in ('bitstream','feature','frame_metrics','transitions'):
            assert sha(r[k+'_path'])==r[k+'_sha256'],k
        assert Path(r['bitstream_path']).stat().st_size==r['real_bytes']
        fr=read(r['frame_metrics_path']);assert len(fr)==v['frames']
        assert sum(int(x['real_bits']) for x in fr)==r['total_bits']
        assert [int(x['frame']) for x in fr]==list(range(v['frames']))
        assert [int(x['actual_qp']) for x in fr]==r['actual_qps']
        assert all(int(x['external_qp'])==q for x in fr)
        assert all(math.isfinite(float(x[k])) for x in fr for k in SPATIAL)
        tr=read(r['transitions_path'])
        assert [(int(x['from_frame']),int(x['to_frame'])) for x in tr]==[(i,i+1) for i in range(v['frames']-1)]
        assert all(math.isfinite(float(x['FloLPIPS'])) for x in tr)
        assert math.isclose(sum(float(x['FloLPIPS']) for x in tr)/len(tr),r['FloLPIPS'],rel_tol=1e-10,abs_tol=1e-12)
        if r['visualization_dir']:
            pngs=sorted(Path(r['visualization_dir']).glob('frame_*.png'));assert len(pngs)==v['frames']
            assert [sha(p) for p in pngs]==r['visualization_sha256']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--method',choices=METHODS,required=True);parser.add_argument('--video',type=int,required=True)
    parser.add_argument('--gpu',choices=(4,5,6,7),type=int,required=True);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    import numpy as np
    torch.set_num_threads(2);cfg=check_frozen();assert load(ROOT/'preflight_audit.json')['status']=='PASS'
    assert torch.cuda.device_count()==1
    lock=(ROOT/'parts'/f'eval_{a.dataset}_{a.method}_{a.video}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v=next(v for v in sources(a.dataset) if v['video_index']==a.video)
    qs=[0,9] if a.smoke else list(range(10));pending=[]
    for q in qs:
        p=point(a.dataset,a.method,a.video,q)
        if p.exists():validate(load(p),v,q,a.method)
        else:pending.append(q)
    if not pending:print('ALREADY COMPLETE',a.dataset,a.method,a.video);return
    sys.path.insert(0,str(ENGINE));from engine import Runtime,metrics
    cps={} if a.method=='original' else {a.method:cfg['checkpoints'][a.method]}
    pair=[None,None] if a.method=='original' else [a.method,a.method]
    # Mode B preserves the same aggregate-MSE PSNR implementation for all datasets.
    runtime=Runtime(dict(experiment='B',methods={a.method:pair},checkpoints=cps,force_zero_thres=.12),a.method,torch.device('cuda:0'))
    frames=frames_for(v);compression=None
    for q in pending:
        t=time.time();relative=Path(a.dataset)/a.method/v['name']/f'qp{q}'
        stream=(ROOT/'bitstreams'/relative).with_suffix('.bin');feature=(ROOT/'features'/relative).with_suffix('.npz')
        recon=ROOT/'visualizations'/relative if visualization(v,q) else None
        if recon is not None:recon.mkdir(parents=True,exist_ok=True)
        r,rows=runtime.run(frames,q,stream,feature,recon_dir=recon)
        out=point(a.dataset,a.method,a.video,q);ff=out.with_suffix('.frames.csv');write(ff,rows)
        with np.load(feature) as f:
            assert f['real'].shape==f['reconstruction'].shape==(v['frames'],2048)
            fid_value=float(metrics.fid.low_rank_fid(f['real'],f['reconstruction']));assert math.isfinite(fid_value)
        actual=[row['actual_qp'] for row in rows]
        assert actual==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:v['frames']]
        trans=feature.with_suffix('.transitions.csv')
        r.update(dataset=a.dataset,method=a.method,video_index=a.video,sequence=v['name'],external_qp=q,QP=q,
            checkpoint_sha256='' if a.method=='original' else cfg['checkpoints'][a.method]['sha256'],
            checkpoint_path='' if a.method=='original' else cfg['checkpoints'][a.method]['path'],
            source_sha256=v['source_sha256'],source_rgb_sha256=v['rgb_sha256'],source_frame_indices=v['source_frame_indices'],
            source_frame_indices_sha256=hashlib.sha256(json.dumps(v['source_frame_indices'],separators=(',',':')).encode()).hexdigest(),
            rate_accounting_fps=v['rate_accounting_fps'],kbps=r['bits_per_frame']*v['rate_accounting_fps']/1000,
            total_bits=r['real_bytes']*8,FID=fid_value,FID_num_samples=v['frames'],force_zero_thres=.12,
            frame_metrics_path=str(ff),frame_metrics_sha256=sha(ff),transitions_path=str(trans),transitions_sha256=sha(trans),
            actual_qps=actual,physical_gpu=a.gpu,fresh_run=True,temporal_resampling=False,elapsed_seconds=time.time()-t,
            config_sha256=sha(ROOT/'config.json'),evaluator_sha256=sha(Path(__file__)),
            visualization_dir=str(recon) if recon else '',visualization_sha256=[sha(p) for p in sorted(recon.glob('frame_*.png'))] if recon else [])
        assert compression is None or r['compression_hash_before']==compression
        compression=r['compression_hash_before'];validate(r,v,q,a.method);dump(out,r)
        print('DONE',a.dataset,a.method,a.video,'QP',q,'bytes',r['real_bytes'],'seconds',round(r['elapsed_seconds'],2),flush=True)
    check_frozen()
    dump(ROOT/'parts'/f'eval_done_{a.dataset}_{a.method}_{a.video}{"_smoke" if a.smoke else ""}.json',dict(status='PASS',points=qs,gpu=a.gpu,finished_unix=time.time()))
if __name__=='__main__':
    try:main()
    except Exception:
        p=argparse.ArgumentParser(add_help=False);p.add_argument('--dataset');p.add_argument('--method');p.add_argument('--video');a,_=p.parse_known_args()
        dump(ROOT/'logs'/f'failure_{a.dataset}_{a.method}_{a.video}_{time.time_ns()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
