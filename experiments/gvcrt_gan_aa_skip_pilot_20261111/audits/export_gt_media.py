"""Reuse verified GT RGB only, re-export native FFV1 and test 1:1 crops now.

No candidate model, training output, flow or final-test identity is involved.
These exact GT files are reused by the prescribed later candidate video export.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common_s2 import *
import hashlib,numpy as np
from fractions import Fraction
from videos import metadata


def frames(path,h,w):
    p=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'],stdout=subprocess.PIPE)
    try:
        for i in range(128):
            buf=bytearray()
            while len(buf)<h*w*3:
                chunk=p.stdout.read(h*w*3-len(buf));assert chunk;buf.extend(chunk)
            yield bytes(buf)
        assert not p.stdout.read(1) and p.wait()==0
    finally:
        if p.poll() is None:p.kill();p.wait()


def main():
    env();old=load(ROOT/'manifests/video_manifest.json')['native_clips'];records=[]
    for clip in load(ROOT/'manifests/long_videos.json'):
        sequence=clip['sequence'];r=next((a for a in old if a['model']=='GT' and a['sequence']==sequence),None)
        source_SHA=sha(clip['row']['source_path']);assert source_SHA==clip['row']['source_sha256']
        fps=clip['row']['source_fps']
        if r is not None:
            source=Path(r['path']);assert sha(source)==r['SHA256']
            assert r['source_frame_indices']==list(range(128)) and r['frames']==128
            assert source_SHA==r['source_SHA256'];h,w=r['height'],r['width']
            assert Fraction(r['fps'])==Fraction(fps)
            expected_hashes=r['RGB8_frame_hashes'];input_proof=dict(path=str(source),SHA256=r['SHA256'],same_native_frames_and_fps_verified=True)
            pixels=lambda:frames(source,h,w)
        else:
            from reader import read_arrays
            row=dict(clip['row'],long_video=True,source_frame_indices=list(range(128)))
            arrays=read_arrays(row,False);h,w=arrays[0].shape[:2]
            expected_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
            development=next(v for v in load(ROOT/'manifests/development_videos.json') if v['sequence']==sequence)
            by_source=dict(zip(development['source_frame_indices'],development['frame_rgb_sha256']))
            overlap=[i for i in range(128) if i in by_source]
            assert all(expected_hashes[i]==by_source[i] for i in overlap)
            input_proof=dict(path=clip['row']['source_path'],SHA256=source_SHA,real_native_frames=128,overlapping_development_frames_checked=overlap,development_source_indices=development['source_frame_indices'],long_demo_indices=list(range(128)),decoder='frozen reader.py/OpenCV RGB')
            pixels=lambda:(a.tobytes() for a in arrays)
        path=ROOT/'media/native'/f'{sequence}_QP5_GT_step0000.mkv';path.parent.mkdir(parents=True,exist_ok=True)
        receipt=ROOT/'media/receipts'/f'{path.stem}.json'
        if path.exists():
            a=load(receipt);assert sha(path)==a['SHA256'];records.append(a);continue
        tmp=ROOT/'tmp'/f'GT_{sequence}_{os.getpid()}.mkv';roi=load(ROOT/'manifests/visual_ROIs.json')[sequence]
        x,y,cw,ch=[roi[k] for k in ['x','y','width','height']]
        log=(ROOT/'logs/GT_FFV1_export.log').open('a')
        writer=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',fps,'-i','pipe:0','-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2',str(tmp)],stdin=subprocess.PIPE,stderr=log)
        crop_hashes=[]
        try:
            for i,raw in enumerate(pixels()):
                assert hashlib.sha256(raw).hexdigest()==expected_hashes[i]
                writer.stdin.write(raw)
                image=np.frombuffer(raw,dtype=np.uint8).reshape(h,w,3)
                crop_hashes.append(hashlib.sha256(image[y:y+ch,x:x+cw].copy().tobytes()).hexdigest())
            writer.stdin.close();assert writer.wait()==0
        except BaseException:
            writer.kill();writer.wait();raise
        m=metadata(tmp);assert (m['width'],m['height'],int(m['nb_read_frames']),m['codec_name'])==(w,h,128,'ffv1') and Fraction(m['r_frame_rate'])==Fraction(fps)
        for i,raw in enumerate(frames(tmp,h,w)):assert hashlib.sha256(raw).hexdigest()==expected_hashes[i]
        tmp.replace(path)
        crop=ROOT/'media/crops'/path.name;crop.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-vf',f'crop={cw}:{ch}:{x}:{y}','-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2',str(crop)],check=True)
        for i,raw in enumerate(frames(crop,ch,cw)):assert hashlib.sha256(raw).hexdigest()==crop_hashes[i]
        a=dict(status='PASS',model='GT',step=0,sequence=sequence,QP=5,path=str(path),SHA256=sha(path),bytes=path.stat().st_size,frames=128,shape=[h,w],fps=fps,codec='FFV1',RGB8_lossless_roundtrip=True,RGB8_frame_hashes=expected_hashes,source_SHA256=source_SHA,source_frame_indices=list(range(128)),stream_SHA256=None,GT_has_no_model_or_bitstream_dependency=True,reused_GT_input=input_proof,crop=dict(path=str(crop),SHA256=sha(crop),ROI=roi,scale='1:1',frames=128,RGB8_lossless_roundtrip=True),precision='native decoded RGB8 GT')
        dump(receipt,a);records.append(a)
        dump(ROOT/'audits/GT_video_exports.json',dict(status='RUNNING' if len(records)<4 else 'PASS',GT_clips=len(records),records=records,candidate_videos_not_generated=True,driver_SHA256=sha(Path(__file__)),time=time.time()))


if __name__=='__main__':main()
