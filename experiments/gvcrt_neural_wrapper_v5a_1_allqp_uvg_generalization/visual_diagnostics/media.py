import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from diag_utils import *
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def writer(path,width,height,fps):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    command=['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
        '-framerate',str(fps),'-i','pipe:0','-an','-vf','scale=in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p',
        '-c:v','libx264','-crf','16','-preset','fast','-threads','4','-color_range','tv','-colorspace','bt709',
        '-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',str(path)]
    p=subprocess.Popen(command,stdin=subprocess.PIPE)
    return p,command
def finish(p):
    p.stdin.close();assert p.wait()==0,'ffmpeg export failed'
def panel(array,label,width=640):
    h=width*9//16;out=Image.new('RGB',(width,h+60),'#101820')
    out.paste(Image.fromarray(array,'RGB').resize((width,h),Image.Resampling.LANCZOS),(0,60))
    ImageDraw.Draw(out).text((10,5),label,font=ImageFont.truetype(FONT,20),fill='white')
    return out
def grid(arrays,labels,width=640):
    images=[panel(a,l,width) for a,l in zip(arrays,labels)];out=Image.new('RGB',(width*3,(width*9//16+60)*2))
    for i,img in enumerate(images):out.paste(img,((i%3)*width,(i//3)*img.height))
    return out
def probe(path):
    command=['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries',
        'stream=codec_name,width,height,nb_read_frames,pix_fmt,r_frame_rate','-of','json',str(path)]
    return json.loads(subprocess.check_output(command))['streams'][0]
def render(s,q,smoke=False):
    source=rawpath(s,'source',smoke=smoke)
    arrays=[np.load(source,mmap_mode='r'),np.load(rawpath(s,'original',q,smoke),mmap_mode='r'),
        np.load(rawpath(s,'v41_20000',q,smoke),mmap_mode='r'),np.load(rawpath(s,'v41_20000_proxy',smoke=smoke),mmap_mode='r'),
        np.load(rawpath(s,'clip8',q,smoke),mmap_mode='r'),np.load(rawpath(s,'clip8_proxy',smoke=smoke),mmap_mode='r')]
    records={m:load(f'bt709_smoke/records/{m}_qp{q}.json') if smoke else point(s,m,q) for m in METHODS}
    def label(m):return f'{LABELS[m]} reconstruction\nQP={q} | {float(records[m]["kbps"]):.1f} kbps'
    labels=['Source / Ground Truth',label('original'),label('v41_20000'),'V4.1 proxy P(x)',label('clip8'),'clip8 proxy P(x)']
    name='bt709_smoke' if smoke else 'videos'
    dest=ROOT/name/s['dataset']/s['name']/f'qp_{q}_comparison.mp4'
    output,cmd=writer(dest,1920,840,s['video']['fps']);keys=[]
    sheets=ROOT/'contact_sheets'/('BT709' if smoke else 'CURRENT')/s['dataset']/s['name']/f'qp_{q}'
    sheets.mkdir(parents=True,exist_ok=True)
    for i in range(count(s)):
        frame=grid([a[i] for a in arrays],labels)
        output.stdin.write(frame.tobytes())
        if i in keyframes(s):
            frame.save(sheets/f'frame_{i:03d}.png');keys.append(frame.resize((960,420),Image.Resampling.LANCZOS))
    finish(output)
    sheet=Image.new('RGB',(1920,840))
    for i,img in enumerate(keys):sheet.paste(img,((i%2)*960,(i//2)*420))
    sheet.save(sheets/'contact_sheet.png')
    info=probe(dest);assert int(info['nb_read_frames'])==count(s) and info['codec_name']=='h264'
    return dict(path=str(dest),command=cmd,probe=info,contact_sheet_directory=str(sheets),
        dataset=s['dataset'],video_index=s['video_index'],external_qp=q,source_conversion='BT709_EXPLICIT' if smoke else 'CURRENT')
