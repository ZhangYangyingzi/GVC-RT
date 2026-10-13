from io21 import *
import argparse,io

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
 import torch,numpy as np
 from PIL import Image,ImageDraw
 from adapter import engine,frames_for
 torch.set_num_threads(2);eng=engine();cfg=evalcfg();plan=load(ROOT/'plans/visuals.json');records=[]
 for entry in plan['samples']:
  d=entry['dataset'];v=next(v for v in sources(d) if v['video_index']==entry['video_index']);frames=frames_for(v);reference=load(point(d,'B1_initial',v['video_index'],5));target_bpp=reference['bpp'];panels={};rates={};points={}
  for m in METHODS[:3]:
   rs=[load(point(d,m,v['video_index'],q)) for q in range(10)];r=min(rs,key=lambda x:abs(__import__('math').log(x['bpp']/target_bpp)));rates[m]=r['bpp'];points[m]=r['QP']
   runtime=eng.Runtime(dict(experiment='B',IP=-1,methods={m:[None,None] if m=='original' else [m,m]},checkpoints={} if m=='original' else {m:cfg['checkpoints'][m]},force_zero_thres=.12),m,torch.device('cuda'))
   ia,pa=runtime.models(1088,1920);pa.clear_dpb();pa.set_curr_poc(0);stream=io.BytesIO(Path(r['bitstream_path']).read_bytes());helper=eng.eco.SPSHelper();panels[m]={};hs=[]
   with torch.inference_mode():
    for i in range(64):
     header=eng.eco.read_header(stream)
     while header['nal_type']==eng.eco.NalType.NAL_SPS:helper.add_sps_by_id(eng.eco.read_sps_remaining(stream,header['sps_id']));header=eng.eco.read_header(stream)
     sps=helper.get_sps_by_id(header['sps_id']);q,bits=eng.eco.read_ip_remaining(stream)
     if i==0:dec=ia.decompress(bits,sps,q);pa.clear_dpb();pa.add_ref_frame(None,dec['x_hat'])
     else:dec=pa.decompress(bits,sps,q)
     hs.append(eng.eco.tensor_sha(dec['x_hat']))
     if i in entry['frames']:
      rgb=eng.core.unit(dec['x_hat'],1080,1920).float().cpu()[0].permute(1,2,0).numpy();x,y,w,h=plan['ROI'];panels[m][i]=Image.fromarray(np.round(rgb[y:y+h,x:x+w]*255).clip(0,255).astype('uint8'))
   assert stream.tell()==len(stream.getvalue()) and eng.eco.sha256_bytes(''.join(hs).encode())==r['reconstruction_sha256'];runtime.assert_frozen(ia,pa);del runtime,ia,pa;torch.cuda.empty_cache()
  for i in entry['frames']:
   x,y,w,h=plan['ROI'];gt=frames[i][0].permute(1,2,0).numpy();canvas=Image.new('RGB',(w*4,h+55),'white');draw=ImageDraw.Draw(canvas);canvas.paste(Image.fromarray(np.round(gt[y:y+h,x:x+w]*255).astype('uint8')),(0,55));draw.text((5,5),'GT',fill='black')
   for j,m in enumerate(METHODS[:3],1):canvas.paste(panels[m][i],(j*w,55));draw.text((j*w+5,5),f'{m}\nQP={points[m]} bpp={rates[m]:.7f}',fill='black')
   p=ROOT/'plots/visual_baselines'/d/f'{v["name"]}_frame{i}.png';p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p);records.append(dict(dataset=d,sequence=v['name'],frame=i,ROI=plan['ROI'],target_B1_QP=5,target_bpp=target_bpp,rates=rates,rate_differences_vs_B1={m:rates[m]-target_bpp for m in METHODS[:3]},nearest_real_rate=True,image_interpolation=False,path=str(p),sha256=sha(p)))
 dump(ROOT/'audits/visual_baselines_integrity.json',dict(status='PASS',pre_registered_plan_sha256=sha(ROOT/'plans/visuals.json'),records=records,real_independent_decode=True,no_image_interpolation=True))
if __name__=='__main__':main()
