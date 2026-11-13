"""Lossless validation, previews and five-panel native-pixel ROI delivery."""
from util import *
import numpy as np,concurrent.futures
from evaluate import tag,proofpath,VideoSink

def metadata_assets():
 selected=[devrows()[k] for k in [0,2]];assets=[]
 for p in (ROOT/'evaluation/points').glob('*.json'):
  r=load(p)
  if not r.get('video'):continue
  j=r['job'];row=next(v for v in selected if v['id']==j['id']);v=dict(r['video']);v.update(label=j['label'],step=j['step'],beta=j['beta'],QP=j['qp'],bpp=r['summary']['bpp'],id=j['id'],width=row['width'],height=row['height'],fps=row['source_fps'],source_frames=row['source_frame_indices'],stream_sha256=r['stream_sha256'],checkpoint_sha256=r['checkpoint_sha256']);assets.append(v)
 for row in selected:assets.append(load(ROOT/'audits/video_assets'/f"{row['id']}_GT.json")|dict(step=0,beta=None,QP=None,bpp=None))
 assert len(assets)==30,('NATIVE_VIDEO_COUNT',len(assets));return assets

def verify_and_preview(v):
 path=Path(v['path']);assert path.exists() and sha(path)==v['sha256'],'VIDEO_FILE_HASH_MISMATCH';report=ROOT/'audits/video_verified'/f'{path.parent.name}_{path.stem}.json'
 if report.exists():
  old=load(report)
  if old['sha256']==v['sha256'] and Path(old['mp4']).exists() and sha(old['mp4'])==old['mp4_sha256']:return old
 p=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);h=hashlib.sha256();count=0
 while True:
  chunk=p.stdout.read(2**20)
  if not chunk:break
  h.update(chunk);count+=len(chunk)
 error=p.stderr.read();assert p.wait()==0,error.decode();assert count==64*v['width']*v['height']*3 and h.hexdigest()==v['RGB8_sha256'],'FFV1_RGB8_ROUNDTRIP_MISMATCH'
 probe=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,pix_fmt,color_range,color_space','-of','json',str(path)],capture_output=True,text=True,check=True).stdout)['streams'][0];assert probe['width']==v['width'] and probe['height']==v['height'] and probe['r_frame_rate']==v['fps'];assert probe.get('color_range')=='pc'
 mp4=path.with_suffix('.mp4');tmp=mp4.with_suffix('.part.mp4');subprocess.run(['ffmpeg','-y','-v','error','-threads','2','-i',str(path),'-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-threads','2',str(tmp)],check=True);tmp.replace(mp4)
 out=dict(v,mp4=str(mp4),mp4_sha256=sha(mp4),bytes=path.stat().st_size,RGB8_roundtrip_exact=True,container_probe=probe,preview='lossy H264 BT709 limited-range YUV420; scientific metrics are prior to RGB8 quantization');dump(report,out);event('VIDEO_VALIDATED',path=str(path),frames=64);return out

_rois={}
def roi_frames(v,row):
 key=v['path']
 if key not in _rois:
  x,y,w,h=row['ROI'];out=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',key,'-vf',f'crop={w}:{h}:{x}:{y}','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True).stdout;_rois[key]=np.frombuffer(out,np.uint8).reshape(64,h,w,3).copy()
 return _rois[key]

def panels(assets):
 import torch
 from PIL import Image,ImageDraw,ImageFont
 selected=[devrows()[k] for k in [0,2]];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);out=[];fixed=load(ROOT/'evaluation/fixed_points.json')['fixed_points']
 for row in selected:
  allrow=[v for v in assets if v['id']==row['id']];w,h=row['ROI'][2:]
  for step in fixed:
   for beta in [.25,.5,1.]:
    pick=[]
    for label in ['GT','Original','B300','A_zero_base','B_flow_guided']:
     candidates=[v for v in allrow if v['label']==label and (label in ['GT','Original','B300'] or (v['step']==step and v['beta']==beta))];assert len(candidates)==1,(row['id'],step,beta,label,len(candidates));pick.append(candidates[0])
    dest=ROOT/'videos'/row['id']/f'GT_Original_B300_A_B_s{step:04d}_beta{beta:g}_native_ROI.mkv';sink=VideoSink(dest,h+48,w*5,row['source_fps']);rr=[roi_frames(v,row) for v in pick]
    for i in range(64):
     image=Image.new('RGB',(w*5,h+48),(20,20,20));draw=ImageDraw.Draw(image)
     for k,v in enumerate(pick):
      title=v['label']+(f' s{step} beta{beta:g}' if k>=3 else '');draw.text((k*w+5,13),title,font=font,fill='white');image.paste(Image.fromarray(rr[k][i]),(k*w,48))
     rgb=np.asarray(image);raw=rgb.tobytes();sink.p.stdin.write(raw);sink.hash.update(raw);sink.frames+=1
     if beta==1 and i in [1,32,63]:
      png=ROOT/'figures/fixed_ROI'/row['id']/f's{step:04d}_frame{i:03d}_source{row["source_frame_indices"][i]:06d}.png';png.parent.mkdir(parents=True,exist_ok=True);image.save(png)
    v=sink.close();v.update(label='GT | Original | B300 | A_zero_base | B_flow_guided',id=row['id'],step=step,beta=beta,QP=5,bpp=pick[2]['bpp'],width=w*5,height=h+48,fps=row['source_fps'],source_frames=row['source_frame_indices'],ROI=row['ROI'],ROI_pixel_scale=1,title_band_pixels=48);out.append(v)
 return out

def deliver():
 assets=metadata_assets()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:verified=list(pool.map(verify_and_preview,assets))
 pp=panels(assets)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:verified.extend(pool.map(verify_and_preview,pp))
 assert len(verified)==42 and all(v['RGB8_roundtrip_exact'] for v in verified);dump(ROOT/'audits/video_delivery.json',dict(status='PASS',native_count=30,ROI_panel_count=12,total=42,frames_each=64,videos=verified))
 lines=['# VIDEO_INDEX','',f'实验目录：`{ROOT}`','', '所有视频均为预先固定的两个ULong开发案例（24747b35、de12efc2），连续64帧，原生1920×1080、原帧率30fps。它们是已使用开发源，不是独立最终测试。FFV1是full-range RGB8无损文件，已逐字节回读核验；MP4为BT.709限幅YUV420有损播放副本。','', '图例：GT；独立Original GVC-RT（只读Original I/P与独立代码）；冻结完整B300；A_zero_base（零基础位移，但offset/mask与融合可学习，仍有公共运动证据）；B_flow_guided（独立Original解码参考上的冻结双向RAFT基础引导）。LoRA s=0.5。','', '固定展示点100和300；beta0经64帧精确等价核验后与B300去重。beta0.25/0.5/1均独立reset/rollout，不插值最终RGB。原尺寸ROI为'+str(devrows()[0]['ROI'])+'；每栏256×256，单独48像素标题栏，未遮挡或缩放内容。','']
 for v in verified:
  meta=f"step={v.get('step')}, beta={v.get('beta')}, externalQP={v.get('QP')}, bpp={v.get('bpp')}, source frames={v['source_frames'][0]}..{v['source_frames'][-1]}, FPS={v['fps']}, {v['width']}×{v['height']}"
  lines.extend([f"- **{v['id']} / {v['label']}**：{meta}",f"  - FFV1：`{v['path']}`",f"  - MP4：`{v['mp4']}`",f"  - SHA256：`{v['sha256']}`",''])
 lines+=['完整路径、大小、FFV1/MP4哈希、RGB8回读哈希及对应checkpoint身份见 audits/video_delivery.json 和 large_artifacts.json。大型媒体仅保存在本地项目内，未上传普通Git。',''];(ROOT/'VIDEO_INDEX.md').write_text('\n'.join(lines));return verified
