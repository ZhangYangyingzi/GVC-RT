"""Full-resolution short-segment lossless validation and explicit source/model index."""
from util import *
import concurrent.futures,numpy as np
from evaluate import VideoSink

def assets():
 out={}
 for p in sorted((ROOT/'metrics/points').glob('*.json')):
  proof=load(p)
  for v in proof['videos']:out[v['path']]=v|dict(checkpoint_sha256=proof['checkpoint_sha256'],stream_sha256=proof['stream_sha256'])
 return list(out.values())

def verify(v):
 p=Path(v['path']);assert sha(p)==v['sha256'];receipt=ROOT/'audits/video_verified'/f'{p.parent.name}_{p.stem}.json'
 if receipt.exists():
  old=load(receipt)
  if old['sha256']==v['sha256'] and Path(old['mp4']).exists() and sha(old['mp4'])==old['mp4_sha256']:return old
 pr=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-i',str(p),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);h=hashlib.sha256();count=0
 while True:
  chunk=pr.stdout.read(2**20)
  if not chunk:break
  h.update(chunk);count+=len(chunk)
 err=pr.stderr.read();assert pr.wait()==0,err.decode();assert count==15*v['width']*v['height']*3 and h.hexdigest()==v['RGB8_sha256'],'RGB8_ROUNDTRIP_FAILED'
 probe=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,color_range','-of','json',str(p)],capture_output=True,text=True,check=True).stdout)['streams'][0];assert probe['width']==v['width'] and probe['height']==v['height'] and probe['r_frame_rate']==v['fps'] and probe['color_range']=='pc'
 mp4=p.with_suffix('.mp4');tmp=mp4.with_suffix('.part.mp4');subprocess.run(['ffmpeg','-y','-v','error','-threads','2','-i',str(p),'-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-threads','2',str(tmp)],check=True);tmp.replace(mp4);out=v|dict(RGB8_roundtrip_exact=True,mp4=str(mp4),mp4_sha256=sha(mp4),mp4_bytes=mp4.stat().st_size,container_probe=probe);dump(receipt,out);return out

def panels(videos):
 from PIL import Image,ImageDraw,ImageFont
 allv=[];last=load(ROOT/'completion.json')['committed_updates'];steps=sorted(set(([100] if last>=100 else [])+[last]));font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
 for row in videos_rows():
  seq=row['sequence'];r=row['ROI'];x,y,w,h=[r[k] for k in ['x','y','width','height']]
  for step in steps:
   pick=[];raw=[]
   for label in ['GT','Original','B300','C_ref','G11']:
    vs=[v for v in videos if v['sequence']==seq and v['label']==label and (label!='G11' or v['step']==step)];assert len(vs)==1,(seq,step,label,len(vs));v=vs[0];pick.append(v);res=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',v['path'],'-vf',f'crop={w}:{h}:{x}:{y}','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True).stdout;raw.append(np.frombuffer(res,np.uint8).reshape(15,h,w,3))
   sink=VideoSink(ROOT/'videos'/seq/f'GT_Original_B300_Cref_G11_step{step:04d}_ROI_native.mkv',h+48,w*5,pick[0]['fps'])
   for i in range(15):
    im=Image.new('RGB',(w*5,h+48),(20,20,20));draw=ImageDraw.Draw(im)
    for j,v in enumerate(pick):draw.text((j*w+5,13),v['label']+(f' step{step}' if v['label']=='G11' else ''),font=font,fill='white');im.paste(Image.fromarray(raw[j][i]),(j*w,48))
    a=np.asarray(im);bs=a.tobytes();sink.proc.stdin.write(bs);sink.hash.update(bs);sink.frames+=1
    if i in [0,7,14]:p=ROOT/'figures/native_ROI'/seq/f'step{step:04d}_source_frame{i+49:02d}.png';p.parent.mkdir(exist_ok=True,parents=True);im.save(p)
   allv.append(sink.close()|dict(label='GT|Original|B300|C_ref|G11',step=step,sequence=seq,fps=pick[0]['fps'],width=w*5,height=h+48,frames_source=list(range(49,64)),ROI=r,title_band_pixels=48,ROI_scale=1,range='full RGB8'))
 return allv

def videos_rows():return videos('validation')
def deliver():
 vv=assets();native_count=len(vv);vv+=panels(vv)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:checked=list(pool.map(verify,vv))
 dump(ROOT/'audits/video_delivery.json',dict(status='PASS',native_videos=native_count,native_ROI_panels=len(vv)-native_count,total_videos=len(vv),frames_each=15,videos=checked))
 lines=['# G11 VIDEO_INDEX','',f'实验目录：`{ROOT}`','', '仅6个预先固定validation来源，49..63共15帧，47/48为独立suffix预热。短段不构成长时间稳定性证据；未使用final_test或树叶诊断样例。FFV1是原分辨率full-range RGB8无损文件，逐字节回读一致；MP4仅为BT709 limited-range有损预览。','', 'Original为独立原版I/P与原版实现、FP32重建；B300为旧完整叠加链；C_ref为完整B300+D12-S@900；G11为唯一训练候选。G11的step100和终点按实际达到情况展示。ROI为原尺寸像素、256×256/栏，标题栏48像素位于内容外。','']
 for v in checked:
  row=next(r for r in videos_rows() if r['sequence']==v['sequence']);info=load(ROOT/'cache/frozen'/v['sequence']/'completion.json');lines += [f"- **{v['sequence']} / {v['label']} / step{v['step']}**：frames49..63；FPS={v['fps']}；{v['width']}×{v['height']}；QP5；实际64帧码流bytes={info['real_bytes']}，bpp={info['bpp']}",f"  - FFV1：`{v['path']}`",f"  - MP4：`{v['mp4']}`",f"  - FFV1 SHA256：`{v['sha256']}`",'']
 lines += ['完整模型/checkpoint/stream身份、大小与FFV1/MP4哈希见audits/video_delivery.json及large_artifacts.json。大型媒体未上传普通Git。'];(ROOT/'VIDEO_INDEX.md').write_text('\n'.join(lines)+'\n');return checked
