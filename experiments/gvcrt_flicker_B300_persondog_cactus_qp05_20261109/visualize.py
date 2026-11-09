"""GT-defined fixed crops, matched frames and native FPS; viewing-only movies."""
from common_s2 import *
from task import tasks,point_path
import concurrent.futures

def ffmpeg(arguments):
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y',*arguments],check=True)

def gt_reference(profile):
    from data_reader import read_arrays
    values=read_arrays(profile['row']);directory=ROOT/'videos';directory.mkdir(exist_ok=True)
    path=directory/f'{profile["alias"]}_GT_128frames.mkv';receipt=directory/f'{profile["alias"]}_GT.json'
    if receipt.exists():
        r=load(receipt);assert sha(path)==r['SHA256'];return path
    with (ROOT/'logs/GT_video_errors.log').open('ab') as error:
        p=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1920x1080','-framerate',profile['row']['source_fps'],'-i','pipe:0','-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2',str(path)],stdin=subprocess.PIPE,stderr=error)
        for frame in values:p.stdin.write(frame.tobytes())
        p.stdin.close();assert p.wait()==0
    actual=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    assert actual==values[0].tobytes()
    dump(receipt,dict(status='PASS',path=str(path),SHA256=sha(path),frames=128,source_indices=profile['row']['source_frame_indices'],lossless_RGB_roundtrip=True));return path

def render(task):
    profile=task['profile'];q=task['QP'];alias=profile['alias'];rois=load(ROOT/'rois.json')[alias]
    points={m:load(point_path(profile,q,m)) for m in ['Original','B']};inputs=[ROOT/f'videos/{alias}_GT_128frames.mkv',*[Path(points[m]['lossless_video']['path']) for m in ['Original','B']]]
    bytes_=points['Original']['result']['real_bytes'];kbps=points['Original']['result']['kbps'];out=ROOT/'comparisons';out.mkdir(exist_ok=True)
    manifest=out/f'{alias}_QP{q}.json'
    if manifest.exists():
        saved=load(manifest)
        for row in saved['files']:assert sha(row['path'])==row['SHA256']
        return saved
    # Full native single-model previews retain detail outside the fixed crops.
    files=[]
    for model,path in zip(['Original','B300'],inputs[1:]):
        dest=out/f'{alias}_QP{q}_{model}_native1920x1080_stream{bytes_}B_{kbps:.1f}kbps.mp4'
        ffmpeg(['-threads','1','-i',str(path),'-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','libx264','-crf','12','-preset','medium','-threads','2','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv','-movflags','+faststart',str(dest)])
        files.append(dict(kind=model+'_individual_native',path=str(dest),SHA256=sha(dest),viewing_lossy=True))
    layouts=[dict(name='overview',transform='scale=640:360:flags=bilinear',panel=[640,360])]
    for roi in rois:
        x,y,w,h=roi['crop_xywh'];factor=roi['display_integer_scale']
        operation=f'crop={w}:{h}:{x}:{y}'
        if factor!=1:operation+=f',scale={w*factor}:{h*factor}:flags=neighbor'
        layouts.append(dict(name=roi['name'],transform=operation,panel=[w*factor,h*factor],crop=roi['crop_xywh'],integer_scale=factor))
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    # Exact RGB8 stills come from lossless inputs, not from the H264 previews.
    import numpy as np
    from PIL import Image,ImageDraw
    proof=load(ROOT/'audits/data'/f'{profile["sequence"]}.json')
    rgb=[np.load(ROOT/proof['RGB_cache'],mmap_mode='r')[64]]
    for model,path in zip(['Original','B'],inputs[1:]):
        raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-vf','select=eq(n\\,64)','-vsync','0','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'])
        records=point_path(profile,q,model).parents[1]/'frames'/f'{profile["sequence"]}_qp{q}.csv'
        with records.open() as f:measured=list(csv.DictReader(f))[64]
        assert hashlib.sha256(raw).hexdigest()==measured['viewing_RGB8_SHA256']
        rgb.append(np.frombuffer(raw,np.uint8).reshape(1080,1920,3))
    for layout in layouts:
        args=[]
        for path in inputs:args+=['-threads','1','-i',str(path)]
        filters=[]
        for i,model in enumerate(['GT','Original GVC-RT','B300 s0.5']):
            label=f'{model} QP{q} {layout["name"]}'
            filters.append(f'[{i}:v]setpts=PTS-STARTPTS,{layout["transform"]},format=rgb24,drawtext=fontfile={font}:text={label}:x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7[v{i}]')
        filters.append('[v0][v1][v2]hstack=inputs=3,scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p[out]')
        dest=out/f'{alias}_QP{q}_{layout["name"]}_GT_Original_B300_stream{bytes_}B_{kbps:.1f}kbps.mp4'
        ffmpeg(args+['-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[out]','-frames:v','128','-c:v','libx264','-crf','12','-preset','medium','-threads','2','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv','-movflags','+faststart',str(dest)])
        files.append(dict(kind=layout['name'],path=str(dest),SHA256=sha(dest),layout=layout,viewing_lossy=True))
        png=dest.with_suffix('.png');ffmpeg(['-threads','1','-i',str(dest),'-vf','select=eq(n\\,64)','-frames:v','1',str(png)])
        if 'crop' in layout:
            x,y,w,h=layout['crop'];factor=layout['integer_scale'];pw,ph=layout['panel']
            canvas=Image.new('RGB',(3*pw,ph+32),'black');draw=ImageDraw.Draw(canvas)
            for i,(image,model) in enumerate(zip(rgb,['GT','Original','B300'])):
                crop=Image.fromarray(image[y:y+h,x:x+w])
                if factor!=1:crop=crop.resize((pw,ph),Image.Resampling.NEAREST)
                canvas.paste(crop,(i*pw,32));draw.text((i*pw+8,8),f'{model} QP{q} frame64 {layout["name"]}',fill='white')
            exact=out/f'{alias}_QP{q}_{layout["name"]}_frame64_exact_RGB8.png';canvas.save(exact)
            files.append(dict(kind=layout['name']+'_exact_frame64',path=str(exact),SHA256=sha(exact),source_frame=profile['row']['source_frame_indices'][64],viewing_lossy=False,layout=layout))
    result=dict(status='PASS',sequence=profile['sequence'],alias=alias,QP=q,frames=128,source_indices=profile['row']['source_frame_indices'],fps=profile['row']['source_fps'],
        panel_order=['GT','Original GVC-RT','B300 s0.5'],ROI_definition_before_model_metrics=True,files=files,metrics_computed_from_float32_not_previews=True)
    dump(manifest,result);return result

def main():
    env()
    for profile in load(ROOT/'profiles.json'):gt_reference(profile)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(render,tasks()))
    dump(ROOT/'comparison_manifest.json',dict(status='PASS',comparisons=records))

if __name__=='__main__':main()
