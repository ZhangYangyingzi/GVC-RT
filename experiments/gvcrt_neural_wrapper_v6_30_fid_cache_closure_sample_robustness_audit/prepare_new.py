"""Freeze exactly max(old source indices)+65..+128; never substitute clips."""
from audit_io import *
import subprocess, numpy as np
from PIL import Image
def main():
    output_manifest={};issues=[]
    for d in DATASETS:
        vs=[]
        for old in sources(d):
            v=dict(old);start=max(old['source_frame_indices'])+65;indices=list(range(start,start+64));path=Path(old['source_path'])
            assert sha(path)==old['source_sha256'],('source YUV mismatch',path)
            pixel=old['pixel_format'];assert pixel in ('yuv420p','yuv420p10le')
            framebytes=1920*1080*(3 if pixel=='yuv420p10le' else 1.5);assert framebytes==int(framebytes)
            n,rem=divmod(path.stat().st_size,int(framebytes));assert rem==0,('truncated YUV',path)
            v.update(source_total_frames=n,source_frame_indices=indices,new_start=start,old_frame_indices=old['source_frame_indices'],status='PLANNED')
            if indices[-1]>=n:
                v.update(status='SKIPPED',reason=f'fixed new clip needs source frame {indices[-1]}, source has {n} complete frames');issues.append(dict(dataset=d,sequence=v['name'],reason=v['reason']));vs.append(v);continue
            directory=ROOT/'cache/new_rgb'/d/v['name'];directory.mkdir(parents=True,exist_ok=True)
            oldcmd=old.get('ffmpeg_command',old.get('conversion_command'));vf=oldcmd[oldcmd.index('-vf')+1]
            cmd=['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-f','rawvideo','-pixel_format',pixel,'-video_size','1920x1080','-framerate',str(old['rate_accounting_fps']),'-i',str(path),'-vf',f"select='between(n,{start},{start+63})',"+vf,'-vsync','0','-frames:v','64','-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
            # Extraction is deterministic; completed manifest signatures protect reuse.
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE);rawhash=[];filehash=[];whole=hashlib.sha256();frame_size=1920*1080*3
            for i in range(64):
                raw=proc.stdout.read(frame_size);assert len(raw)==frame_size,('short extraction',v['name'],i)
                a=np.frombuffer(raw,np.uint8).reshape(1080,1920,3);whole.update(raw);rawhash.append(hashlib.sha256(raw).hexdigest())
                filename=f'im{i+1:05d}.png' if d=='hevc_b' else f'frame_{i:06d}.png';p=directory/filename
                Image.fromarray(a,'RGB').save(p);filehash.append(sha(p))
            assert not proc.stdout.read(1);err=proc.stderr.read().decode();assert proc.wait()==0,err
            v.update(status='AVAILABLE',input_dir=str(directory),frame_rgb_sha256=rawhash,rgb_sha256=whole.hexdigest(),frames=64,new_conversion_command=cmd,old_conversion_command=oldcmd,no_resampling=True)
            v['frame_png_sha256' if d=='hevc_b' else 'frame_file_sha256']=filehash;vs.append(v)
        output_manifest[d]=vs
        dump(ROOT/'new_clips_manifest.json',output_manifest)
    dump(ROOT/'audits/new_clip_preparation.json',dict(status='PASS',issues=issues,available=sum(v['status']=='AVAILABLE' for vs in output_manifest.values() for v in vs),expected_new_points=12*sum(v['status']=='AVAILABLE' for vs in output_manifest.values() for v in vs),choice_frozen_before_any_metrics=True))
if __name__=='__main__':main()
