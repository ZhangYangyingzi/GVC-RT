"""GT-only window/format verification, before fixed ROI/model evaluation."""
from common_s2 import *
import numpy as np

def main():
    env()
    with lock('prepare.lock'):
        source=ORIGINAL_EVAL/'videos.json';catalog=load(source)
        profiles=[]
        for sequence,alias in [('dce599d3-d196-4d87-b2b1-698acdbbb3d4','person_dog'),('Cactus','Cactus')]:
            sealed=next(r for r in catalog if r['sequence']==sequence)
            path=Path(sealed['source_path']);assert sha(path)==sealed['source_sha256']
            start=sealed['source_frame_indices'][0];indices=list(range(start,start+128));h,w=sealed['height'],sealed['width']
            proofpath=ROOT/'audits/data'/f'{sequence}.json'
            if proofpath.exists():
                proof=load(proofpath);assert proof['status']=='PASS' and sha(ROOT/proof['RGB_cache'])==proof['RGB_cache_SHA256']
                values=np.load(ROOT/proof['RGB_cache'],mmap_mode='r');hashes=proof['frame_RGB_SHA256'];fps=proof['fps'];conversion=proof['conversion']
            else:
                if sequence=='Cactus':
                    reference=PROJECT/'experiments/gvcrt_neural_wrapper_v5a_2_fair_30fps_rgb/ffmpeg_conversion_audit.json'
                    prior=PROJECT/'experiments/gvcrt_neural_wrapper_v6_2b_final_generalization_test/results/hevc_b/source_manifest.json'
                    declaration=next(r for r in load(prior)['videos'] if r['name']=='Cactus')
                    assert declaration['pixel_format']=='yuv420p' and declaration['bit_depth']==8 and declaration['color_matrix']=='bt709' and declaration['source_range']=='limited' and declaration['RGB_range']=='full'
                    assert declaration['source_sha256']==sealed['source_sha256'] and start==0
                    assert path.stat().st_size//(w*h*3//2)>=128
                    command=['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pixel_format','yuv420p','-video_size',f'{w}x{h}','-framerate','50','-i',str(path),'-vf',load(reference)['filter'],'-frames:v','128','-vsync','0','-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
                    raw=subprocess.check_output(command);assert len(raw)==128*h*w*3
                    values=np.frombuffer(raw,np.uint8).reshape(128,h,w,3);fps='50/1'
                    conversion=dict(method='Exact inherited HEVC-B FFmpeg conversion command, extended to128frames',reference=str(reference),reference_SHA256=sha(reference),prior_manifest=str(prior),prior_manifest_SHA256=sha(prior),filter=load(reference)['filter'],command=command,pixel_format='yuv420p',bit_depth=8,color_matrix='bt709',YUV_range='limited',RGB_range='full')
                else:
                    import cv2
                    cv2.setNumThreads(1);cap=cv2.VideoCapture(str(path));assert cap.get(cv2.CAP_PROP_FRAME_COUNT)>=indices[-1]+1
                    fps='30000/1001';assert abs(cap.get(cv2.CAP_PROP_FPS)-30000/1001)<1e-6
                    cap.set(cv2.CAP_PROP_POS_FRAMES,start);images=[]
                    try:
                        for i in indices:
                            ok,bgr=cap.read();assert ok,('NATIVE_SOURCE_FRAME_MISSING',sequence,i)
                            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert rgb.shape==(h,w,3);images.append(rgb)
                    finally:cap.release()
                    values=np.stack(images);conversion=dict(method='Inherited native OpenCV BGR->RGB; no resize/resampling',fps_metadata=fps)
                hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in values]
                assert hashes[:64]==sealed['frame_rgb_sha256'],'SEALED_RGB_FIRST64_MISMATCH'
                cache=ROOT/'cache/RGB'/f'{sequence}.npy';cache.parent.mkdir(parents=True,exist_ok=True)
                temp=cache.with_suffix('.tmp')
                with temp.open('wb') as output:np.save(output,values)
                temp.replace(cache)
                proof=dict(status='PASS',source_path=str(path),source_SHA256=sealed['source_sha256'],source_frame_indices=indices,native_geometry=[w,h],fps=fps,
                    RGB_cache=str(cache.relative_to(ROOT)),RGB_cache_SHA256=sha(cache),frame_RGB_SHA256=hashes,sealed_first64_frame_hashes_exact_match=True,sealed_videos_manifest=str(source),sealed_videos_manifest_SHA256=sha(source),conversion=conversion)
                dump(proofpath,proof)
            row=dict(sequence=sequence,domain=sealed['domain'],split=sealed['split'],video=sequence+path.suffix,source_path=str(path),source_sha256=sealed['source_sha256'],
                source_frame_indices=indices,canonical_indices=indices,height=h,width=w,source_fps=fps,long_video=True,frame_RGB_SHA256=hashes)
            profiles.append(dict(sequence=sequence,alias=alias,row=row,posthoc_requested_test=True,existing_split=sealed['split']))
            from PIL import Image,ImageDraw
            image=Image.fromarray(np.asarray(values[64]));directory=ROOT/'GT_roi_selection';directory.mkdir(exist_ok=True);image.save(directory/f'{alias}_middle_frame.png')
            contact=Image.new('RGB',(960,540*3),'black');draw=ImageDraw.Draw(contact)
            for j,i in enumerate([0,64,127]):
                contact.paste(Image.fromarray(np.asarray(values[i])).resize((960,540)),(0,j*540));draw.text((10,j*540+10),f'{alias} source frame{indices[i]}',fill='red')
            contact.save(directory/f'{alias}_GT_first_middle_last.png')
            event('GT_window_verified',sequence=sequence,frames=128,first=indices[0],last=indices[-1],sealed_first64_match=True)
        dump(ROOT/'profiles.json',profiles)
        dump(ROOT/'status.json',dict(status='PREPARING',current_stage='GT_only_ROI_definition',training_updates=0,profiles=2,time=time.time()))

if __name__=='__main__':main()
