"""Inherited GT-selected development clips; native FPS/geometry, FFV1 and 1:1 crops."""
from common_s2 import *
import argparse


def metadata(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames','-of','json',str(path)],text=True))['streams'][0]


def render(cache,original_path,clip):
    import torch,numpy as np
    from experiment_model import construct,restore,reconstruct
    from model_context import assert_reset
    d=torch.load(cache,map_location='cpu',weights_only=True,mmap=True);orig=torch.load(original_path,map_location='cpu',weights_only=True,mmap=True)
    info=d['info'];row=info['source_row'];sequence=clip['sequence'];fps=clip['row']['source_fps'];h,w=info['native_shape'];assert info['frames']==128
    roi=load(ROOT/'manifests/visual_ROIs.json')[sequence]
    x,y,cw,ch=[roi[k] for k in ['x','y','width','height']]
    for model in ['GT','Original','B300',*GROUPS]:
        step=500 if model in GROUPS else 0
        folder=ROOT/'media/native';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'{sequence}_QP5_{model}_step{step:04d}.mkv';record=ROOT/'media/receipts'/f'{path.stem}.json'
        if path.exists():
            r=load(record);assert sha(path)==r['SHA256'];continue
        c=None
        if model in ['B300',*GROUPS]:
            c=construct(model,False,0,losses=False)
            if model in GROUPS:restore(c,torch.load(ROOT/model/'training/checkpoints/step_0500.pt',map_location='cpu',weights_only=True),False)
            assert_reset(c)
        tmp=ROOT/'tmp'/f'{path.stem}_{os.getpid()}.mkv'
        log=(ROOT/'logs'/f'ffv1_{path.stem}.log').open('a')
        p=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2',str(tmp)],stdin=subprocess.PIPE,stderr=log)
        hashes=[]
        try:
            with torch.no_grad():
                for i in range(128):
                    if model=='GT':array=d['GT'][i].numpy()
                    else:
                        if model=='Original':raw=orig['values'][i]
                        elif i==0:raw=d['original'][0]
                        else:raw=reconstruct(c,d['features'][i],info['actual_qps'][i])
                        array=(((raw.float().clamp(-1,1)+1)/2)[0,:,:h,:w].permute(1,2,0)*255).round().clamp(0,255).byte().cpu().numpy()
                    array=np.ascontiguousarray(array);hashes.append(hashlib.sha256(array.tobytes()).hexdigest());p.stdin.write(array.tobytes())
            p.stdin.close();assert p.wait()==0
        except BaseException:
            p.kill();p.wait();raise
        meta=metadata(tmp);assert (meta['width'],meta['height'],int(meta['nb_read_frames']),meta['codec_name'])==(w,h,128,'ffv1')
        from fractions import Fraction
        assert Fraction(meta['r_frame_rate'])==Fraction(fps)
        # Verify RGB8 losslessness against the exact bytes supplied to FFmpeg.
        decoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-i',str(tmp),'-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'],stdout=subprocess.PIPE)
        for expected in hashes:
            buf=bytearray()
            while len(buf)<h*w*3:
                chunk=decoder.stdout.read(h*w*3-len(buf));assert chunk;buf.extend(chunk)
            assert hashlib.sha256(buf).hexdigest()==expected
        assert not decoder.stdout.read(1) and decoder.wait()==0
        tmp.replace(path)
        crop=ROOT/'media/crops'/path.name;crop.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-vf',f'crop={cw}:{ch}:{x}:{y}','-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2',str(crop)],check=True)
        dump(record,dict(status='PASS',model=model,step=step,sequence=sequence,QP=5,path=str(path),SHA256=sha(path),bytes=path.stat().st_size,frames=128,shape=[h,w],fps=fps,codec='FFV1',RGB8_lossless_roundtrip=True,RGB8_frame_hashes=hashes,source_SHA256=row['source_sha256'],stream_SHA256=info['stream_SHA256'],crop=dict(path=str(crop),SHA256=sha(crop),ROI=roi,scale='1:1',frames=128),precision=DEPLOYMENT))
        del c;torch.cuda.empty_cache()
    for kind in ['native','crops']:
        paths=[ROOT/f'media/{kind}'/f'{sequence}_QP5_{m}_step{500 if m in GROUPS else 0:04d}.mkv' for m in ['GT','Original','B300',*GROUPS]]
        panel=ROOT/f'media/panels/{sequence}_{kind}_GT_Original_B300_J0_J1_J2_J3.mkv';panel.parent.mkdir(parents=True,exist_ok=True)
        if not panel.exists():
            inputs=[arg for path in paths for arg in ['-i',str(path)]]
            subprocess.run(['ffmpeg','-nostdin','-v','error','-filter_complex_threads','1',*inputs,'-filter_complex','hstack=inputs=7','-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2',str(panel)],check=True)
        dump(ROOT/f'media/receipts/{sequence}_{kind}_panel.json',dict(path=str(panel),SHA256=sha(panel),bytes=panel.stat().st_size,order=['GT','Original','B300',*GROUPS],metadata=metadata(panel),scale='1:1',fps=fps))


def main():
    from cache_codec import cache_paths
    clips=load(ROOT/'manifests/long_videos.json');dev={x['sequence'] for x in load(ROOT/'manifests/development_videos.json')}
    for clip in clips:
        assert clip['sequence'] in dev and clip['source_frames']>=128
        row=dict(clip['row']);row['long_video']=True;row['source_frame_indices']=row['canonical_indices']=list(range(128))
        row['actual_frame_qps']=[5 if i==0 else 5+[0,2,1][[0,1,0,2,0,2,0,2][i%8]] for i in range(128)]
        row['external_qp']=5
        req=ROOT/'cache/requests/video.json';dump(req,[dict(row=row,qp=5,training=False)])
        subprocess.run([PYTHON,'cache_job.py',str(req)],check=True,cwd=ROOT)
        key,path,receipt=cache_paths(row,5,False);original=ROOT/'cache/original_FP32'/f'{key}.pt'
        if not original.exists():subprocess.run([PYTHON,'original_reconstruction.py',str(path),str(original)],cwd=ROOT,check=True)
        request=ROOT/'cache/requests/video_clip.json';dump(request,clip)
        subprocess.run([PYTHON,'videos.py','--cache',str(path),'--original',str(original),'--clip',str(request)],cwd=ROOT,check=True)
        if path.is_relative_to(ROOT):path.unlink()
        original.unlink()
    records=[load(p) for p in sorted((ROOT/'media/receipts').glob('*.json'))]
    dump(ROOT/'video_manifest.json',dict(status='PASS',records=records,clips=len(clips),GT_selection='inherited H GT-only selection and ROIs',no_final_test=True,missing_categories='occlusion category not explicitly annotated in inherited metadata; no new final-test example selected'))


if __name__=='__main__':
    env();p=argparse.ArgumentParser();p.add_argument('--cache');p.add_argument('--original');p.add_argument('--clip');a=p.parse_args()
    if a.cache:render(a.cache,a.original,load(a.clip))
    else:main()
