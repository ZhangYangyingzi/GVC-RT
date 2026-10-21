"""Fixed GT-selected visual comparisons; no metric-driven image selection."""
from common42 import *
from PIL import Image,ImageDraw
import subprocess,html

def banner(images,labels,widths=None):
 w=sum(im.width for im in images);h=max(im.height for im in images)
 canvas=Image.new('RGB',(w,h+30),'white');d=ImageDraw.Draw(canvas);x=0
 for im,label in zip(images,labels):canvas.paste(im,(x,30));d.text((x+5,8),label,fill='black');x+=im.width
 return canvas
def ffmpeg(source,fps,dest,lossless=False):
 args=['ffmpeg','-v','error','-threads','1','-y','-framerate',str(fps),'-i',str(source),'-frames:v','64','-an']
 args+=['-c:v','ffv1','-level','3','-pix_fmt','bgr0'] if lossless else ['-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart']
 subprocess.run(args+[str(dest)],check=True)
def main():
 vm=load(ROOT/'visual_manifest.json');manifest_hash=sha(ROOT/'visual_manifest.json');groups=[];videos=[]
 for v in vm['sequences']:
  for q in vm['QPs']:
   ds,n=v['dataset'],v['sequence'];key=f'{ds}/{n}/q{q}';receipt=load(ROOT/'tasks'/key/'pair.json')
   assert receipt['status']=='PASS' and receipt['visual_manifest_sha256']==manifest_hash
   media=ROOT/'server_media'/key;out=ROOT/'visuals'/key;out.mkdir(parents=True,exist_ok=True)
   for i in vm['frames']:
    originals=[Image.open(media/m/f'{i:03d}.png').convert('RGB') for m in ('GT',*METHODS)]
    overview=[]
    for im in originals:
     c=im.copy();draw=ImageDraw.Draw(c)
     for rix,r in enumerate(v['rois']):
      x,y,w,h=r['x'],r['y'],r['width'],r['height'];draw.rectangle((x,y,x+w-1,y+h-1),outline=('red' if rix==0 else 'cyan'),width=5)
     overview.append(c.resize((640,360),Image.Resampling.LANCZOS))
    p=out/f'f{i:03d}_overview.jpg';banner(overview,[f'{m} | frame {i} | 1/3 overview' for m in ('GT',*METHODS)]).save(p,quality=88,subsampling=0)
    crops=[]
    for rix,r in enumerate(v['rois']):
     box=(r['x'],r['y'],r['x']+256,r['y']+256)
     trip= [im.crop(box) for im in originals]
     rp=out/f'f{i:03d}_roi{rix+1}.png';banner(trip,[f'{m} f{i} ROI{rix+1} ({r["x"]},{r["y"]})' for m in ('GT',*METHODS)]).save(rp)
     crops.append(str(rp.relative_to(ROOT)))
    groups.append(dict(dataset=ds,sequence=n,QP=q,frame=i,overview=str(p.relative_to(ROOT)),rois=crops,
      originals=[dict(model=m,path=str(media/m/f'{i:03d}.png'),sha256=sha(media/m/f'{i:03d}.png')) for m in ('GT',*METHODS)]))
   # These previews use the same saved lossless PNGs as the static views.
   cache=ROOT/'server_previews'/key;cache.mkdir(parents=True,exist_ok=True)
   full=cache/'full_frames';full.mkdir(exist_ok=True)
   roidirs=[cache/'roi1_frames',cache/'roi2_frames']
   for p in roidirs:p.mkdir(exist_ok=True)
   for i in range(64):
    ims=[Image.open(media/m/f'{i:03d}.png').convert('RGB') for m in ('GT',*METHODS)]
    banner([im.resize((640,360),Image.Resampling.LANCZOS) for im in ims],[f'{m} f{i} overview 1/3' for m in ('GT',*METHODS)]).save(full/f'{i:03d}.png',compress_level=1)
    for rix,r in enumerate(v['rois']):
     box=(r['x'],r['y'],r['x']+256,r['y']+256)
     banner([im.crop(box) for im in ims],[f'{m} f{i} ROI{rix+1}' for m in ('GT',*METHODS)]).save(roidirs[rix]/f'{i:03d}.png',compress_level=1)
   previews=[]
   for name,p in [('full',full),('roi1',roidirs[0]),('roi2',roidirs[1])]:
    mp4=cache/f'{name}.mp4';ffmpeg(p/'%03d.png',v['fps'],mp4)
    entry=dict(view=name,preview=str(mp4.relative_to(ROOT)),sha256=sha(mp4),display_encoding='H264 CRF18; detail inspection uses lossless PNGs')
    if name!='full':
     mkv=cache/f'{name}_lossless.mkv';ffmpeg(p/'%03d.png',v['fps'],mkv,True);entry['lossless']=str(mkv.relative_to(ROOT));entry['lossless_sha256']=sha(mkv)
    previews.append(entry)
   videos.append(dict(dataset=ds,sequence=n,QP=q,fps=v['fps'],frames=64,same_verified_decode=True,lossless_original_PNGs=str(media),previews=previews))
   print('VISUAL_PASS',key,flush=True)
 assert len(groups)==32 and len(videos)==8
 dump(ROOT/'visual_index.json',dict(status='PASS',visual_manifest_sha256=manifest_hash,static_groups=groups,video_groups=videos))
 # Server-only assets are enabled only when this HTML is opened from the local filesystem.
 parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>v6.42 fixed content comparisons</title>',
 '<style>body{font:15px Arial;margin:24px;max-width:1800px}img{max-width:100%;height:auto}article{border-top:1px solid #aaa;padding:16px 0}video{max-width:100%}nav a{margin-right:16px}code{overflow-wrap:anywhere}</style>',
 '<h1>v6.42: GT / O_original / G_basic</h1><p>Static frames: 16,32,48,63. Fixed GT-only ROIs. ROI panels are 1:1 pixels; overviews uniformly resized to 1/3. H264 videos are display previews; inspect original PNGs and FFV1 ROI videos for lossless details. Full media remain on the server and are not uploaded to GitHub.</p>',
 '<nav>'+''.join(f'<a href="#{v["sequence"]}">{v["sequence"]}</a>' for v in vm['sequences'])+'</nav>']
 for v in vm['sequences']:
  parts.append(f'<h2 id="{v["sequence"]}">{v["dataset"]} / {v["sequence"]}</h2><p>Timeline: {v["fps"]} fps. ROIs: {html.escape(json.dumps(v["rois"]))}</p>')
  for q in (0,9):
   parts.append(f'<h3>QP {q}</h3>')
   vid=next(x for x in videos if x['sequence']==v['sequence'] and x['QP']==q)
   for p in vid['previews']:
    parts.append(f'<p>{p["view"]} video, 64 synchronized frames. <span class="local" data-path="{p["preview"]}">Server-only: {html.escape(str(ROOT/p["preview"]))}</span></p>')
   parts.append(f'<p>Lossless full-size PNG source: <span class="local" data-path="{str(Path(vid["lossless_original_PNGs"]).relative_to(ROOT))}/">{html.escape(vid["lossless_original_PNGs"])}</span></p>')
   for g in [g for g in groups if g['sequence']==v['sequence'] and g['QP']==q]:
    parts.append(f'<article><h4>Frame {g["frame"]}</h4><img loading="lazy" src="{g["overview"]}" alt="GT original basic overview">')
    parts.extend(f'<p>ROI {i+1}, original 256×256 pixels per model</p><img loading="lazy" src="{p}" alt="fixed ROI comparison">' for i,p in enumerate(g['rois']))
    for a in g['originals']:
     rel=str(Path(a['path']).relative_to(ROOT));parts.append(f'<p>{a["model"]}: <span class="local" data-path="{rel}">{html.escape(a["path"])}</span><br>SHA256 <code>{a["sha256"]}</code></p>')
    parts.append('</article>')
 parts.append('<script>if(location.protocol==="file:"){document.querySelectorAll(".local").forEach(e=>{let a=document.createElement("a");a.href=e.dataset.path;a.textContent=e.textContent;e.replaceWith(a);if(a.href.endsWith(".mp4")){let v=document.createElement("video");v.controls=true;v.preload="none";v.src=a.href;a.after(v);}})}</script></html>')
 textfile(ROOT/'index.html','\n'.join(parts))
if __name__=='__main__':main()
