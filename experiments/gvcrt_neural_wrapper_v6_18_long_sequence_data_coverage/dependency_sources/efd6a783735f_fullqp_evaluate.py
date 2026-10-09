"""Real complete bitstreams, independent fresh decode, unchanged metrics."""
import argparse
import fcntl
import math
import tempfile
import traceback
from fullqp_common import *

def point_path(branch,step,split,index,q):
    return ROOT/'parts'/'fullqp'/branch/f'step_{step:04d}'/split/f'video_{index:02d}_qp{q}.json'

def read_frames(v):
    import numpy as np
    import torch
    assert sha(v['path'])==v['sha256']
    command=['ffmpeg','-v','error','-threads','1','-i',v['path'],'-map','0:v:0','-frames:v',str(v['frames']),
             '-vsync','0','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    with tempfile.TemporaryFile(dir=ROOT/'parts') as err:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err);frames=[];digest=hashlib.sha256()
        try:
            size=v['width']*v['height']*3
            for _ in range(v['frames']):
                raw=process.stdout.read(size)
                assert len(raw)==size,'incomplete validation video frame'
                digest.update(raw)
                arr=np.frombuffer(raw,np.uint8).reshape(v['height'],v['width'],3).copy()
                frames.append(torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()/255)
            assert process.stdout.read()==b''
            if process.wait()!=0:
                err.seek(0);raise RuntimeError(err.read().decode(errors='replace'))
        finally:
            process.stdout.close()
            if process.poll() is None:process.terminate();process.wait()
    return frames,digest.hexdigest()

def validate_point(r,v,q,cp_hash):
    assert r['revision']==REVISION and r['checkpoint_sha256']==cp_hash
    assert r['external_qp']==q and r['source_sha256']==v['sha256']
    assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass']
    for key in ('bitstream','feature','frame_metrics','transitions'):
        assert sha(r[key+'_path'])==r[key+'_sha256'],key
    assert Path(r['bitstream_path']).stat().st_size==r['real_bytes']==r['bytes_consumed']
    assert r['frames']==v['frames'] and r['num_transitions']==v['frames']-1
    assert r['bits']==r['real_bytes']*8
    assert math.isclose(r['bpp'],r['bits']/(v['width']*v['height']*v['frames']),rel_tol=1e-12)
    assert math.isclose(r['kbps'],r['bits']*v['fps']/v['frames']/1000,rel_tol=1e-12)
    assert r['metric_crop']==[v['width'],v['height']]
    assert r['compression_hash_before']==r['compression_hash_after']
    assert all(math.isfinite(r[m]) for m in (*SPATIAL,'FloLPIPS','kbps','bpp','proxy_RGB_L1','proxy_RGB_MSE','proxy_edge_difference'))
    rows=read(r['frame_metrics_path']);assert len(rows)==v['frames']
    assert sum(int(x['real_bits']) for x in rows)==r['bits']
    for i,row in enumerate(rows):
        assert int(row['frame'])==i and int(row['external_qp'])==q
        assert int(row['actual_qp'])==r['actual_qps'][i]
        assert all(math.isfinite(float(row[m])) for m in SPATIAL)
    transitions=read(r['transitions_path'])
    assert [(int(t['from_frame']),int(t['to_frame'])) for t in transitions]==[(i,i+1) for i in range(v['frames']-1)]
    assert all(math.isfinite(float(t['FloLPIPS'])) for t in transitions)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--branch',required=True,choices=['original']+list(BRANCHES))
    parser.add_argument('--step',type=int,choices=STEPS,default=0);parser.add_argument('--split',choices=['normal','hard'],required=True)
    parser.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True)
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=4)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    torch.set_num_threads(2);cfg=check_gate();assert_old_unchanged()
    assert 0<=args.shard<args.shards
    lockpath=ROOT/'parts'/f'fullqp_eval_{args.branch}_{args.step}_{args.split}_{args.shard}.lock'
    lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(OLD_ENGINE));from engine import Runtime,eco
    videos=load(ROOT/f'validation_{args.split}_manifest.json')['videos']
    if args.branch=='original':
        assert args.step==0;cp=None;digest='';cps={};methodpair=[None,None]
    else:
        cp=cp_path(args.branch,args.step);record=load(ROOT/'branches'/args.branch/'checkpoint_hashes.json')[str(args.step)]
        digest=sha(cp);assert digest==record['sha256']
        payload=torch.load(cp,map_location='cpu',weights_only=True)
        assert payload['revision']==REVISION and payload['branch']==args.branch and payload['step']==args.step
        assert model_hashes(payload)==record['module_hashes']
        cps={'candidate':dict(path=str(cp),sha256=digest,module_hashes=model_hashes(payload))}
        methodpair=['candidate','candidate'];del payload
    ecfg=dict(experiment='B',methods={'current':methodpair},checkpoints=cps,force_zero_thres=.12)
    runtime=None;done=0;start_all=time.time()
    qs=[0,9] if args.smoke else QPS
    if args.smoke:videos=videos[:1]
    from src.models.video_model_gvcrt import DMC
    semantics_model=DMC()
    expected_qps={q:[frame_qp(semantics_model,q,i) for i in range(32)] for q in qs};del semantics_model
    for ordinal,v in enumerate(videos):
        if ordinal%args.shards!=args.shard:continue
        pending=[]
        for q in qs:
            out=point_path(args.branch,args.step,args.split,v['video_index'],q)
            if out.exists():validate_point(load(out),v,q,digest)
            else:pending.append(q)
        if not pending:continue
        if runtime is None:runtime=Runtime(ecfg,'current',torch.device('cuda:0'))
        frames,rgb_sha=read_frames(v);vals=[]
        with torch.inference_mode():
            for f in frames:
                x=f.to('cuda:0');y=x if runtime.wrapper is None else runtime.wrapper(x)
                dx=(y[:,:,:,1:]-y[:,:,:,:-1])-(x[:,:,:,1:]-x[:,:,:,:-1])
                dy=(y[:,:,1:,:]-y[:,:,:-1,:])-(x[:,:,1:,:]-x[:,:,:-1,:])
                vals.append((float((y-x).abs().mean()),float((y-x).square().mean()),float((dx.abs().mean()+dy.abs().mean())/2)))
        diagnostics=dict(proxy_RGB_L1=sum(x[0] for x in vals)/len(vals),proxy_RGB_MSE=sum(x[1] for x in vals)/len(vals),
                         proxy_edge_difference=sum(x[2] for x in vals)/len(vals),proxy_edge_definition='Mean absolute RGB horizontal/vertical first-difference change')
        for q in pending:
            out=point_path(args.branch,args.step,args.split,v['video_index'],q);out.parent.mkdir(parents=True,exist_ok=True)
            relative=Path('fullqp')/args.branch/f'step_{args.step:04d}'/args.split/f'video_{v["video_index"]:02d}_qp{q}'
            stream=(ROOT/'bitstreams'/relative).with_suffix('.bin');feature=(ROOT/'features'/relative).with_suffix('.npz');t=time.time()
            result,rows=runtime.run(frames,q,stream,feature)
            actual=[r['actual_qp'] for r in rows];assert actual==expected_qps[q]
            framefile=out.with_suffix('.frames.csv');write(framefile,rows);transition=feature.with_suffix('.transitions.csv')
            result.update(revision=REVISION,branch=args.branch,checkpoint_update=args.step,dataset=args.split,validation_split=args.split,
                method=args.branch,external_qp=q,QP=q,video_index=v['video_index'],video=v['name'],source_sha256=v['sha256'],
                source_rgb_sha256=rgb_sha,checkpoint=str(cp) if cp else '',checkpoint_sha256=digest,
                frame_metrics_path=str(framefile),frame_metrics_sha256=sha(framefile),transitions_path=str(transition),transitions_sha256=sha(transition),
                bits=result['real_bytes']*8,kbps=result['real_bytes']*8*v['fps']/v['frames']/1000,rate_accounting_fps=v['fps'],actual_qps=actual,
                force_zero_thres=.12,physical_gpu=args.gpu,elapsed_seconds=time.time()-t,**diagnostics)
            validate_point(result,v,q,digest);dump(out,result);done+=1
            print('DONE',args.branch,args.step,args.split,v['video_index'],q,'bytes',result['real_bytes'],flush=True)
        del frames
    assert_old_unchanged()
    dump(ROOT/'parts'/f'fullqp_eval_done_{args.branch}_{args.step}_{args.split}_{args.shard}{"_smoke" if args.smoke else ""}.json',
         dict(status='PASS',revision=REVISION,branch=args.branch,checkpoint_update=args.step,split=args.split,shard=args.shard,shards=args.shards,
              smoke_only=args.smoke,new_points=done,elapsed_seconds=time.time()-start_all))
if __name__=='__main__':main()
