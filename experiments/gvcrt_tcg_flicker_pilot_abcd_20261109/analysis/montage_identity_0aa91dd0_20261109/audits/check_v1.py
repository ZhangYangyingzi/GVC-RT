"""Inspect saved viewing assets without neural inference or changing results."""
from pathlib import Path
import sys
EXPERIMENT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(EXPERIMENT))
from common_s2 import *
import concurrent.futures
import numpy as np

SEQ='0aa91dd0-0594-4082-9154-d8d2bc94692e'
MODELS=['GT','Original','A2000','A','B','C','D']
INDICES=[32,64,96]
OUT=Path(__file__).resolve().parent

def extract(path,w,h):
    filt='+'.join(f'eq(n\\,{i})' for i in INDICES)
    raw=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-threads','1','-i',str(path),'-vf','select='+filt,'-vsync','0','-frames:v','3','-f','rawvideo','-pix_fmt','bgr0','-threads','1','pipe:1'])
    assert len(raw)==3*w*h*4
    return np.frombuffer(raw,np.uint8).reshape(3,h,w,4).copy()

def scaled(frames):
    raw=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-f','rawvideo','-pixel_format','bgr0','-video_size','1920x1080','-framerate','30','-i','pipe:0','-vf','scale=960:540,format=bgr0','-frames:v','3','-f','rawvideo','-pix_fmt','bgr0','-threads','1','pipe:1'],input=frames.tobytes())
    return np.frombuffer(raw,np.uint8).reshape(3,540,960,4)[...,:3][...,::-1].copy()

def main():
    env();records={g:load(ROOT/'videos/manifests'/f'{SEQ}_{g}.json') for g in MODELS}
    montage=next(x for x in load(ROOT/'videos/manifest.json')['videos'] if x['sequence']==SEQ)
    assert sha(montage['lossless'])==montage['sha256'] and sha(montage['preview'])==montage['preview_sha256']
    def read(group):
        r=records[group];assert sha(r['path'])==r['sha256']
        frames=extract(r['path'],1920,1080);rgb=frames[...,:3][...,::-1].copy()
        for j,index in enumerate(INDICES):assert hashlib.sha256(rgb[j].tobytes()).hexdigest()==r['RGB8_frame_hashes'][index]
        return group,rgb,scaled(frames)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:loaded=list(pool.map(read,MODELS))
    native={g:r for g,r,s in loaded};small={g:s for g,r,s in loaded};joined=extract(montage['lossless'],3840,1080)[...,:3][...,::-1]
    panels={g:joined[:,(i//4)*540:(i//4+1)*540,(i%4)*960:(i%4+1)*960] for i,g in enumerate(MODELS)}
    matrix={g:{h:float(np.abs(panels[g][:,60:].astype(np.int16)-small[h][:,60:].astype(np.int16)).mean()) for h in MODELS} for g in MODELS}
    assert all(matrix[g][g]==0 for g in MODELS),'MONTAGE_PANEL_DOES_NOT_MATCH_ITS_LABELED_SOURCE'
    rows=[]
    for g in MODELS:
        for j,index in enumerate(INDICES):
            delta=native[g][j].astype(np.float64)-native['GT'][j].astype(np.float64)
            mse=float(np.square(delta).mean());bdelta=native[g][j].astype(np.float64)-native['A2000'][j].astype(np.float64)
            rows.append(dict(model=g,frame=index,MAE_RGB8_to_GT=float(np.abs(delta).mean()),PSNR_RGB8_to_GT_dB=None if mse==0 else -10*math.log10(mse/(255**2)),
                changed_pixel_fraction_to_GT=float(np.any(delta!=0,axis=2).mean()),MAE_RGB8_to_A2000=float(np.abs(bdelta).mean())))
    write_csv(OUT/'sampled_frame_pixel_differences.csv',rows)
    profile=next(r for r in load(ROOT/'manifests/long_videos.json') if r['sequence']==SEQ)
    path=Path(profile['row']['source_path']);assert sha(path)==profile['row']['source_sha256']
    import cv2
    cv2.setNumThreads(1);cap=cv2.VideoCapture(str(path))
    for i in range(max(INDICES)+1):
        ok,bgr=cap.read();assert ok
        if i in INDICES:assert np.array_equal(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),native['GT'][INDICES.index(i)])
    cap.release()
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(1920,580),'black');draw=ImageDraw.Draw(canvas)
    for i,g in enumerate(['GT','Original']):
        crop=native[g][1,270:810,480:1440];canvas.paste(Image.fromarray(crop),(i*960,40));draw.text((i*960+12,10),f'{g}: actual frame64, native center960x540, no resizing',fill='white')
    canvas.save(OUT/'frame64_GT_vs_Original_native_center.png')
    diff=np.minimum(np.abs(native['Original'][1].astype(np.int16)-native['GT'][1].astype(np.int16))*8,255).astype(np.uint8)
    Image.fromarray(diff[270:810,480:1440]).save(OUT/'frame64_absolute_GT_Original_difference_x8.png')
    video=OUT/f'{SEQ}_QP5_stream445002B_GT_vs_Original_native_center960x540.mp4'
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    filt=f'[0:v]crop=960:540:480:270,drawtext=fontfile={font}:text=GT:x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7[a];[1:v]crop=960:540:480:270,drawtext=fontfile={font}:text=Original GVC-RT QP5:x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7[b];[a][b]hstack=inputs=2,scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p[out]'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-i',records['GT']['path'],'-threads','1','-i',records['Original']['path'],'-filter_complex_threads','2','-filter_complex',filt,'-map','[out]','-frames:v','128','-c:v','libx264','-crf','12','-preset','medium','-threads','2','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv','-movflags','+faststart',str(video)],check=True)
    fps=float(__import__('fractions').Fraction(records['GT']['fps']));stream_bytes=records['Original']['stream_bytes']
    audit=dict(status='PASS',sequence=SEQ,original_montage_SHA256=montage['preview_sha256'],sampled_frames=INDICES,
        panel_mapping={g:i for i,g in enumerate(MODELS)},panel_MAE_matrix_RGB8_excluding_top60_label_rows=matrix,all_seven_lossless_panels_exactly_match_labeled_input=True,
        GT_matches_actual_source_frames=True,source_SHA256=profile['row']['source_sha256'],model_vs_GT_different_frames_in_all128={g:sum(a!=b for a,b in zip(records[g]['RGB8_frame_hashes'],records['GT']['RGB8_frame_hashes'])) for g in MODELS},
        frames=128,source_geometry=[1920,1080],montage_geometry=[3840,1080],panel_geometry=[960,540],panel_geometry_if_player_fits1920wide=[480,270],external_QP=5,
        stream_bytes=stream_bytes,bpp=stream_bytes*8/(128*1920*1080),kbps=stream_bytes*8*fps/128/1000,source_fps=records['GT']['fps'],
        comparison_preview=str(video),comparison_preview_SHA256=sha(video),comparison_is_fixed_crop_no_resizing=True,comparison_is_lossy_viewing_only=True,
        pixel_differences_are_viewing_RGB8_not_replacements_for_float32_neural_metrics=True,new_training_updates=0,new_neural_metric_points=0,check_source_SHA256=sha(Path(__file__)),time=time.time())
    dump(OUT/'audit.json',audit)
    for r in rows:
        if r['frame']==64:print(r,flush=True)
    print('INSPECTION_PASS',OUT/'audit.json',flush=True)

if __name__=='__main__':main()
