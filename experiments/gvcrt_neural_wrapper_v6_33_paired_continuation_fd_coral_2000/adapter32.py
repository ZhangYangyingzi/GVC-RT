"""Historical real-RANS protocol with independent v6.32 configuration."""
from io32 import *
from training32 import backend,official_model
def frames_for(v):
 import torch,numpy as np
 from PIL import Image
 class Frames:
  def __len__(self):return 64
  def __getitem__(self,i):
   if i>=64:raise IndexError(i)
   p=Path(v['input_dir'])/(f'im{i+1:05d}.png' if v['dataset']=='hevc_b' else f'frame_{i:06d}.png')
   assert sha(p)==v['frame_png_sha256' if v['dataset']=='hevc_b' else 'frame_file_sha256'][i]
   with Image.open(p) as im:
    assert im.mode=='RGB' and im.size==(1920,1080);a=np.asarray(im,dtype=np.uint8).copy()
   assert hashlib.sha256(a.tobytes()).hexdigest()==v['frame_rgb_sha256'][i]
   return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255
 return Frames()
def eval_config(t):
 cfg=load(ROOT/'evaluation_config.json');m=t['method']
 if t['stage']!='baseline':
  ix=load(ROOT/'branches'/t['branch']/'checkpoint_index.json')[str(t['step'])]
  cp=ix['inference'];assert sha(cp['path'])==cp['sha256']
  cfg['checkpoints'][m]=cp;cfg['deployment_receiver_hashes'][m]=ix['deployment_receiver_hashes'];cfg['methods'][m]=[m,m]
 cfg.update(experiment='B',IP=-1)
 return cfg
def runtime(t):
 import torch
 cfg=eval_config(t);ns=dict(legacy_io().__dict__);ns.update(ROOT=ROOT,dump=dump,write=write,evalcfg=lambda:cfg)
 source=(V30/'adapter.py').read_text().replace('from io21 import *','')
 exec(compile(source,str(ROOT/'adapter32.py'),'exec'),ns)
 eng=ns['engine']();rt=eng.Runtime(cfg,t['method'],torch.device('cuda:0'))
 rt.metric_models=(*rt.metric_models[:2],official_model())
 return eng,rt
def decode(eng,rt,path,old):
 import io,torch
 ia,pa=rt.models(1088,1920);pa.clear_dpb();pa.set_curr_poc(0);before=[tensor_hash(m.state_dict()) for m in (ia,pa)]
 stream=io.BytesIO(Path(path).read_bytes());helper=eng.eco.SPSHelper();hashes=[];qps=[]
 with torch.inference_mode():
  for i in range(64):
   pos=stream.tell();h=eng.eco.read_header(stream)
   while h['nal_type']==eng.eco.NalType.NAL_SPS:
    helper.add_sps_by_id(eng.eco.read_sps_remaining(stream,h['sps_id']));h=eng.eco.read_header(stream)
   sp=helper.get_sps_by_id(h['sps_id']);q,bits=eng.eco.read_ip_remaining(stream);qps.append(q)
   assert sp['height']==1088 and sp['width']==1920 and (h['nal_type']==eng.eco.NalType.NAL_I)==(i==0);assert q==old['actual_qps'][i]
   if i==0:dec=ia.decompress(bits,sp,q);pa.clear_dpb();pa.set_curr_poc(0);pa.add_ref_frame(None,dec['x_hat'])
   else:dec=pa.decompress(bits,sp,q)
   hashes.append(eng.eco.tensor_sha(dec['x_hat']));rgb=eng.core.unit(dec['x_hat'],1080,1920).float();assert rgb.shape==(1,3,1080,1920) and rgb.isfinite().all()
   yield i,rgb,dict(actual_qp=q,real_bits=(stream.tell()-pos)*8,payload_bits=len(bits)*8)
 assert stream.tell()==Path(path).stat().st_size
 h=eng.eco.sha256_bytes(''.join(hashes).encode());assert h==old['reconstruction_sha256'];rt.assert_frozen(ia,pa);assert before==[tensor_hash(m.state_dict()) for m in (ia,pa)]
 rt.decode_audit=dict(status='PASS',bytes_consumed=stream.tell(),actual_qps=qps,reconstruction_sha256=h,model_hashes=before)
