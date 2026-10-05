"""Separate fixed official Vimeo test-set diagnostic, native seven-frame clips."""
import argparse,math
from v68_io import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=B_METHODS,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch,numpy as np
    from PIL import Image
    torch.set_num_threads(2);cfg=frozen();manifest=load(ROOT/'audits/vimeo_heldout_manifest.json');assert manifest['status']=='AVAILABLE';e=engine()
    runtime=e.Runtime(dict(experiment='B',methods={a.method:[None,None] if a.method=='original' else [a.method,a.method]},checkpoints=cfg['checkpoints'],force_zero_thres=.12),a.method,torch.device('cuda:0'))
    for v in (manifest['clips'][:1] if a.smoke else manifest['clips']):
        frames=[]
        for p,h,rgb in zip(v['paths'],v['file_sha256'],v['frame_rgb_sha256']):
            assert sha(p)==h
            with Image.open(p) as im:arr=np.asarray(im,dtype=np.uint8).copy()
            assert arr.shape==(256,448,3) and hashlib.sha256(arr.tobytes()).hexdigest()==rgb
            frames.append(torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()/255)
        for q in ([0] if a.smoke else manifest['external_qps']):
            out=ROOT/'parts/heldout'/a.method/f'video_{v["index"]:03d}_qp{q}.json'
            if out.exists():
                r=load(out)
                for k in ('bitstream','feature','frame_metrics','transitions'):assert sha(r[k+'_path'])==r[k+'_sha256']
                continue
            sub=Path(a.method)/f'video_{v["index"]:03d}_qp{q}';stream=(ROOT/'heldout_bitstreams'/sub).with_suffix('.bin');feature=(ROOT/'heldout_features'/sub).with_suffix('.npz');r,fr=runtime.run(frames,q,stream,feature)
            ff=out.with_suffix('.frames.csv');write(ff,fr);tr=feature.with_suffix('.transitions.csv')
            assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'] and r['real_bytes']==r['bytes_consumed']
            assert r['compression_hash_before']==r['compression_hash_after']==cfg['compression_hash']
            assert [x['actual_qp'] for x in fr]==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:7]
            assert all(math.isfinite(r[k]) for k in ('LPIPS','DISTS','bpp'))
            r.update(status='PASS',dataset='vimeo_official_test_separate',method=a.method,QP=q,sequence=v['sequence'],video_index=v['index'],source_frame_rgb_sha256=v['frame_rgb_sha256'],checkpoint_sha256='' if a.method=='original' else cfg['checkpoints'][a.method]['sha256'],frame_metrics_path=str(ff),frame_metrics_sha256=sha(ff),transitions_path=str(tr),transitions_sha256=sha(tr),actual_qps=[x['actual_qp'] for x in fr]);dump(out,r)
            print('HELDOUT',a.method,v['index'],q,'PASS',flush=True)
    if a.smoke:dump(ROOT/'audits/heldout_smoke.json',dict(status='PASS',method=a.method,QP=0,native_frames=7))
    else:dump(ROOT/'parts'/f'heldout_done_{a.method}.json',dict(status='PASS',points=len(manifest['clips'])*len(manifest['external_qps'])))
if __name__=='__main__':main()
