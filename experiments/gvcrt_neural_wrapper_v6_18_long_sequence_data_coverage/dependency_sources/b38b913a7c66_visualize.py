"""Viewing-only MP4 triptychs; original MP4s are never written or regenerated."""
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from parallel_utils import *

def main():
    root=A;config=load(root/'config.json');manifest=[]
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),20) if font_path.exists() else ImageFont.load_default()
    for video in config['videos']:
        for q in (0,3,6,9):
            directory=root/'visualizations'/video['name'];directory.mkdir(parents=True,exist_ok=True)
            out=directory/f'qp{q}_source_original_clip8.mp4';tmp=out.with_suffix('.tmp.mp4')
            records={m:load(point(root,'virat',m,video['video_index'],q)) for m in ('original','clip8')}
            command=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-video_size','2562x480','-framerate','20',
                     '-i','pipe:0','-an','-c:v','libx264','-crf','16','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
            with tempfile.TemporaryFile() as errors:
                process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=errors)
                try:
                    for i,source in enumerate(raw_mp4(video)):
                        arrays=[source]
                        for method in ('original','clip8'):
                            path=root/'reconstructions/virat'/method/f"video_{video['video_index']}_qp{q}"/f'frame_{i:06d}.png'
                            with Image.open(path) as image:
                                assert image.size==(854,480);arrays.append(np.array(image))
                        canvas=Image.fromarray(np.concatenate(arrays,axis=1));draw=ImageDraw.Draw(canvas)
                        labels=[f'Source | QP {q} | codec rate: N/A',f"Original | QP {q} | {records['original']['kbps']:.3f} kbps @20fps",f"clip8 | QP {q} | {records['clip8']['kbps']:.3f} kbps @20fps"]
                        for j,label in enumerate(labels):
                            draw.rectangle((j*854,0,(j+1)*854-1,36),fill='black');draw.text((j*854+8,6),label,font=font,fill='white')
                        process.stdin.write(canvas.tobytes())
                    process.stdin.close();assert process.wait()==0
                finally:
                    if process.poll() is None:process.terminate();process.wait()
                    errors.seek(0)
                    if process.returncode:raise RuntimeError(errors.read().decode())
            tmp.replace(out)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,avg_frame_rate','-of','json',str(out)],text=True))['streams'][0]
            assert (probe['width'],probe['height'],int(probe['nb_read_frames']))==(2562,480,video['frames'])
            manifest.append(dict(video=video['name'],QP=q,path=str(out),sha256=sha(out),frames=video['frames'],probe=probe,view_only=True,
                                 source_sha256=video['source_sha256'],command=command))
            dump(root/'visualization_manifest.json',dict(status='RUNNING',videos=manifest));print('VISUALIZATION',video['name'],q,flush=True)
    assert len(manifest)==32
    dump(root/'visualization_manifest.json',dict(status='PASS',videos=manifest,count=32,source_used_as_codec_input='existing manifest path MP4 only'))
if __name__=='__main__':main()
