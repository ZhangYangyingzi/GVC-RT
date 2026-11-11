"""Pre-registered native FFV1 clips and identical-position 1:1 ROIs, RGB lossless."""
from common_s2 import *
import argparse


def writer(path,w,h,fps):
    path.parent.mkdir(parents=True,exist_ok=True)
    return subprocess.Popen(['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{w}x{h}','-framerate',str(fps),'-i','pipe:0','-an','-c:v','ffv1','-level','3',
        '-pix_fmt','bgr0','-threads','2',str(path)],stdin=subprocess.PIPE,
        stderr=(ROOT/'logs/media_encode.log').open('ab'))


def export(index):
    env();import torch
    from cache_codec import cache_paths
    from experiment_model import construct,reconstruct,restore
    from model_context import assert_reset
    case=load(ROOT/'manifests/media_cases.json')['cases'][index];v=case['video'];q=case['QP'];row=v.get('source_row',v)
    key,p,rp=cache_paths(row,q,False)
    with lock('locks/native_'+digest(dict(row=row,QP=q))+'.lock',blocking=True):
        if not p.exists() or not rp.exists():
            spec=ROOT/f'tmp/media_row{index}.json';dump(spec,row)
            subprocess.run([PYTHON,'-B','-u','cache_codec.py','--row',str(spec),'--qp',str(q)],cwd=ROOT,env=os.environ,check=True)
            key,p,rp=cache_paths(row,q,False)
        assert sha(p)==load(rp)['cache_sha256'];d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        info=d['info'];h,w=info['native_shape'];roi=case['ROI'];x,y,rw,rh=[roi[k] for k in ['x','y','width','height']]
        fps=row.get('source_fps',v.get('source_fps',v.get('fps')));assert fps
        records=[]
        for model in ['GT','Original','B300',*GROUPS]:
            step=100 if model in GROUPS else 0;tag=model if model not in GROUPS else 'A100' if model==GROUPS[0] else 'B100'
            proof=ROOT/f'media/manifests/{v["sequence"]}_{tag}.json'
            if proof.exists():
                a=load(proof);assert all(sha(a[k]['path'])==a[k]['SHA256'] for k in ['native','ROI']);records.append(a);continue
            c=None
            if model not in ['GT','Original']:
                c=construct(model,False,0,with_adapter=model in GROUPS)
                if model in GROUPS:restore(c,torch.load(ROOT/model/'training/checkpoints/step_0100.pt',map_location='cpu',weights_only=True))
                assert_reset(c);before=tensor_hash(c['pm'].state_dict().items())
            native=ROOT/f'media/native/{v["sequence"]}_QP{q}_{tag}.mkv'
            crop=ROOT/f'media/ROI/{v["sequence"]}_QP{q}_{tag}_ROI_1to1.mkv'
            nt=native.with_name(native.stem+'.partial.mkv');ct=crop.with_name(crop.stem+'.partial.mkv')
            nw=writer(nt,w,h,fps);cw=writer(ct,rw,rh,fps);nh=[];ch=[];first=None;firstcrop=None
            with torch.no_grad():
                for i in range(64):
                    if model=='GT':rgb=d['GT'][i].numpy()
                    else:
                        raw=d['original'][i].to('cuda:0') if i==0 or c is None else reconstruct(c,d['features'][i],info['actual_qps'][i])
                        rgb=(((raw.float().clamp(-1,1)+1)/2)[0,:,:h,:w].permute(1,2,0)*255).round().clamp(0,255).byte().cpu().numpy()
                    a=rgb.tobytes();b=rgb[y:y+rh,x:x+rw].copy().tobytes();nw.stdin.write(a);cw.stdin.write(b)
                    nh.append(hashlib.sha256(a).hexdigest());ch.append(hashlib.sha256(b).hexdigest())
                    if first is None:first=a;firstcrop=b
            nw.stdin.close();cw.stdin.close();assert nw.wait()==0 and cw.wait()==0;nt.replace(native);ct.replace(crop)
            for path,expected in [(native,first),(crop,firstcrop)]:
                actual=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'])
                assert actual==expected,'FFV1_RGB_ROUNDTRIP_FAILED'
            if c:assert tensor_hash(c['pm'].state_dict().items())==before;assert_reset(c)
            a=dict(status='PASS',sequence=v['sequence'],category=case['category'],model=model,step=step,QP=q,
                frames=64,source_frame_indices=v['source_frame_indices'],fps=fps,width=w,height=h,ROI_geometry=dict(x=x,y=y,width=rw,height=rh),
                ROI_scale=1,stream_SHA256=info['stream_SHA256'],real_bytes=info['real_bytes'],
                native=dict(path=str(native),SHA256=sha(native),bytes=native.stat().st_size,RGB8_frame_SHA256=nh),
                ROI=dict(path=str(crop),SHA256=sha(crop),bytes=crop.stat().st_size,RGB8_frame_SHA256=ch),
                codec='FFV1 level3 bgr0',color='RGB24 input and exact RGB24 first-frame decode; float output rounded once to RGB8',
                selection_frozen_before_training=True,ROI_unscaled=True,physical_GPU=physical_gpu(0),time=time.time())
            dump(proof,a);records.append(a);del c;torch.cuda.empty_cache()
        dump(ROOT/f'audits/media_case{index}.json',dict(status='PASS',case=index,records=[r['sequence']+'_'+r['model'] for r in records],time=time.time()))
        del d
        if p.is_relative_to(ROOT) and p.exists():p.unlink()


def summarize():
    files=sorted((ROOT/'media/manifests').glob('*.json'));records=[load(p) for p in files]
    dump(ROOT/'video_manifest.json',dict(status='PASS' if len(records)==15 else 'IN_PROGRESS',native_clips=len(records),ROI_clips=len(records),
        expected_native_clips=15,expected_ROI_clips=15,records=records,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int);a=p.parse_args();env()
    if a.case is not None:export(a.case)
    summarize()
