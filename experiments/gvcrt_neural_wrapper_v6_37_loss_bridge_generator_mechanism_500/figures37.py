from io32 import *
import numpy as np

def make_figures(points,pooled,summary,latents):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image,ImageDraw,ImageFont
 import textwrap
 datasets=('ulong','uvg_holdout','hevc_b');metrics=('LPIPS','DISTS','FloLPIPS','FID');figs=[];colors=plt.cm.tab10(np.linspace(0,.9,7))
 def save(fig,name,caption):
  path=ROOT/'figures'/name;fig.savefig(path,dpi=140,bbox_inches='tight');plt.close(fig);figs.append(dict(path=str(path.relative_to(ROOT)),caption=caption,sha256=sha(path)))
 for ds in datasets:
  fig,axs=plt.subplots(2,2,figsize=(14,9))
  for ax,k in zip(axs.flat,metrics):
   for ci,m in enumerate(METHODS):
    rr=[]
    for q in QPS:
     if k=='FID':r=next((r for r in pooled if (r['dataset'],r['method'],r['QP'])==(ds,m,q)),None);x=r['mean_actual_bpp'] if r else None;y=r['FID'] if r else None
     else:
      rs=[r for r in points if (r['dataset'],r['method'],r['QP'])==(ds,m,q)];x=float(np.mean([r['actual_bpp'] for r in rs])) if rs else None;y=float(np.mean([r[k] for r in rs])) if rs else None
     if x is not None:rr.append((x,y))
    if rr:ax.plot([r[0] for r in rr],[r[1] for r in rr],'-o',ms=3,color=colors[ci],label=m)
   ax.set_xlabel('Mean actual complete RANS bpp');ax.set_ylabel(k+' (lower is better)');ax.grid(alpha=.2)
  n=len(load(ROOT/'eval_manifest.json')[ds]);fig.suptitle(f'{ds}: {n} videos ×64 frames; FID samples={n*64}; QP0/5/9\nAnchor curves; equal-rate attribution uses per-video intervals')
  h,l=axs.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,-.04));fig.tight_layout(rect=[0,.06,1,.94]);save(fig,f'{ds}_four_metric_RD.png','Seven full method names; actual total RANS bpp,64 native frames; lower metric distance is favorable.')
 for name,comps in [('loss_comparisons',['DISTS_conditional','remaining_statistics_scheme','B_vs_original','D_vs_original','F_vs_original']),('module_swaps',['bridge_with_originalG','bridge_with_fullG','generator_with_originalBridge','generator_with_fullBridge'])]:
  fig,axs=plt.subplots(3,4,figsize=(18,10))
  for ri,ds in enumerate(datasets):
   for ki,k in enumerate(metrics):
    ax=axs[ri,ki];rs=[next((r for r in summary if (r['dataset'],r['comparison'],r['metric'])==(ds,c,k)),None) for c in comps];vals=[r['delta'] if r else np.nan for r in rs];ax.bar(range(len(comps)),vals,color=['#4183b0' if x<=0 else '#c06a58' for x in vals]);ax.set_xticks(range(len(comps)),comps,rotation=40,ha='right',fontsize=7);ax.axhline(0,color='black',lw=.5);ax.set_title(ds+' '+k+f'; {len(load(ROOT/"eval_manifest.json")[ds])} videos ×64 frames');ax.set_ylabel('Equal actual-rate delta; negative is lower')
  mapping=[]
  for c in comps:
   r=next((r for r in summary if r['comparison']==c),None)
   if r:mapping.append(c+': '+r['method']+' minus '+r['reference'])
  fig.suptitle('Shared measured actual RANS bpp intervals; negative is lower distance; FID pools384/128/320 samples; no grid-based significance claim');fig.text(.5,.004,'\n'.join(mapping),ha='center',va='bottom',fontsize=8);fig.tight_layout(rect=[0,.14,1,.96]);save(fig,name+'.png','Fixed loss/module comparisons. Full method names shown in mapping; shared actual-rate bounds in results/common_rate_bounds.csv; negative is lower distance, not bitrate savings.')
 Y='Y_originalBridge_fullG';I='I_halfLatent_fullG';F='F_fullLoss_noP_500'
 fig,axs=plt.subplots(3,4,figsize=(17,9))
 for ri,ds in enumerate(datasets):
  for ki,k in enumerate(metrics):
   ax=axs[ri,ki]
   for q in QPS:
    yy=[];bb=[]
    for m in (Y,I,F):
     if k=='FID':r=next((r for r in pooled if (r['dataset'],r['method'],r['QP'])==(ds,m,q)),None);y=r['FID'] if r else np.nan;b=r['mean_actual_bpp'] if r else np.nan
     else:
      rr=[r for r in points if (r['dataset'],r['method'],r['QP'])==(ds,m,q)];y=float(np.mean([r[k] for r in rr])) if rr else np.nan;b=float(np.mean([r['actual_bpp'] for r in rr])) if rr else np.nan
     yy.append(y);bb.append(b)
    ax.plot([0,.5,1],yy,'-o',label=f'QP{q}, actual bpp={bb[0]:.5g}')
   ax.set_xticks([0,.5,1],['0: Y','0.5: I','1: F']);ax.set_ylabel(k+' lower is better');ax.set_title(ds);ax.legend(fontsize=6);ax.grid(alpha=.2)
 fig.suptitle('Fixed generator_F: Y_originalBridge_fullG(alpha0), I_halfLatent_fullG(alpha0.5), F_fullLoss_noP_500(alpha1)\n13 videos ×64 frames; only this prescribed latent direction; no optimal-alpha selection',fontsize=11);fig.tight_layout(rect=[0,0,1,.96]);save(fig,'three_point_latent_interpolation.png','Y_originalBridge_fullG / I_halfLatent_fullG / F_fullLoss_noP_500; actual equal streams at each anchor.')
 comparisons=[('O_original','F_fullLoss_noP_500'),('B_basicLoss_noP_500','D_basicDISTS_noP_500'),('D_basicDISTS_noP_500','F_fullLoss_noP_500')]
 fig,axs=plt.subplots(2,3,figsize=(16,8))
 for col,(a,b) in enumerate(comparisons):
  aa=next((r for r in latents if r['level']=='overall' and r['method']==a),None);bb=next((r for r in latents if r['level']=='overall' and r['method']==b),None)
  if aa and bb:
   axs[0,col].plot(np.array(bb['mean'])-np.array(aa['mean']),label='Mean change');axs[0,col].plot(np.array(bb['variance'])-np.array(aa['variance']),label='Variance change');axs[0,col].axhline(0,color='black',lw=.4);axs[0,col].set_xlabel('Actual18-channel generator input');axs[0,col].set_title(b+' minus\n'+a,fontsize=8);axs[0,col].legend();delta=np.array(bb['covariance18x18'])-np.array(aa['covariance18x18']);im=axs[1,col].imshow(delta,cmap='coolwarm');fig.colorbar(im,ax=axs[1,col],shrink=.7);axs[1,col].set_title('FP64 space/time covariance change; ddof0')
 fig.suptitle('Descriptive18-channel codeword changes:13 videos ×3 QPs ×63 P frames; actual padded latent extent\nActual-rate anchors are recorded in all_rd_points.csv; not Inception features or a causal quality claim',fontsize=11);fig.tight_layout(rect=[0,0,1,.95]);save(fig,'latent_statistics_changes.png','Space/time samples pooled by exact cross-moments; all datasets/QPs and P1–P63. Per-video/QP/frame data in results/.')
 # Fixed examples: full frame and identical crop; no quality-dependent selection.
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',11);large=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
 methods=('GT',*METHODS);crop=(720,300,1200,660);panels=[]
 for ds in datasets:
  v=load(ROOT/'eval_manifest.json')[ds][0]
  for q in QPS:
   key=f'{ds}_{v["video_index"]}_q{q}'
   for frame in (16,32,48,63):
    paths={m:ROOT/'artifacts/examples'/key/m/f'{frame}.png' for m in methods}
    if not all(p.exists() for p in paths.values()):continue
    canvas=Image.new('RGB',(256*8,64+52+144+192),(245,245,245));draw=ImageDraw.Draw(canvas);rate=next(r['actual_bpp'] for r in points if (r['dataset'],r['video'],r['QP'])==(ds,v['video_index'],q))
    draw.text((10,8),f'{ds} / {v["name"]} / QP{q} / zero-based frame{frame}; complete actual bpp={rate:.8g};64-frame stream',font=large,fill='black');draw.text((10,32),'All panels equal size. Top: full frame. Bottom: fixed source crop x720 y300 width480 height360. No selected-best images.',font=large,fill='black')
    for col,m in enumerate(methods):
     image=Image.open(paths[m]).convert('RGB');assert image.size==(1920,1080);full=image.copy();ImageDraw.Draw(full).rectangle(crop,outline=(0,220,220),width=4)
     draw.multiline_text((col*256+4,66),'\n'.join(textwrap.wrap(m,32)),font=font,fill='black');canvas.paste(full.resize((256,144),Image.Resampling.LANCZOS),(col*256,116));canvas.paste(image.crop(crop).resize((256,192),Image.Resampling.LANCZOS),(col*256,260))
    path=ROOT/'figures'/f'fixed_{key}_frame{frame}.png';canvas.save(path,optimize=True);figs.append(dict(path=str(path.relative_to(ROOT)),caption=f'Fixed manifest-first video {ds}, QP{q}, frame{frame}; crop={crop}; all8 panels GT/O/B/D/F/X/Y/I.',sha256=sha(path)));panels.append(dict(dataset=ds,video=v['video_index'],QP=q,frame=frame,crop=list(crop),source_paths={m:str(p) for m,p in paths.items()},source_sha256={m:sha(p) for m,p in paths.items()},figure=str(path)))
 dump(ROOT/'audits/fixed_example_manifest.json',dict(selection_rule='first manifest video per dataset ×QP0/5/9 ×zero-based frames16/32/48/63',planned=36,complete=len(panels),panels=panels))
 textfile(ROOT/'figures/FIGURE_INDEX.md','# Figure index\n\n'+'\n'.join(f'- [{r["path"].split("/")[-1]}]({r["path"].split("/")[-1]}): {r["caption"]}' for r in figs)+'\n')
 dump(ROOT/'figures/index.json',figs)
 textfile(ROOT/'FIGURE_INDEX.md','# Figure index\n\nSee [the complete figure index](figures/FIGURE_INDEX.md) for RD curves, fixed loss/module contrasts, the three-point latent path, latent statistics and all36 prespecified montage panels.\n')
