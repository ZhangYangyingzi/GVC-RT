"""Read-only historical codec with explicit, independently audited cross cells."""
from audit31 import *
import types
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
def official_model():
 import torch
 sys.path.insert(0,str(V29/'vendor/pytorch-fid/src'))
 import pytorch_fid.inception as src
 p=Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')
 assert sha(p)=='6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
 src.load_state_dict_from_url=lambda *a,**kw:torch.load(p,map_location='cpu',weights_only=True)
 net=src.InceptionV3([3],resize_input=True,normalize_input=True,requires_grad=False,use_fid_inception=True).cuda().eval().requires_grad_(False)
 class Flat(torch.nn.Module):
  def __init__(self):super().__init__();self.net=net
  def forward(self,x):return self.net(x.float())[0].flatten(1)
 return Flat().eval().requires_grad_(False)
def engine():
 oldio=module('v631_historical_adapter_io',V30/'io21.py');ns=dict(oldio.__dict__)
 cfg=load(ROOT/'evaluation_config.json');cps=cfg['checkpoints'];b=cps['B1_initial']
 cfg['methods']={c:([None if c in ('R00','R01') else 'B1_initial',None if c in ('R00','R10') else 'B1_initial']) for c in CELLS}
 cfg['methods'].update(original=[None,None],B1_initial=['B1_initial','B1_initial'])
 for c in CELLS:
  cfg['checkpoints'][c]=b
  cfg['deployment_receiver_hashes'][c]=cfg['original_receiver_hashes'] if c in ('R00','R10') else cfg['deployment_receiver_hashes']['B1_initial']
 cfg.update(experiment='B',IP=-1)
 ns.update(ROOT=ROOT,evalcfg=lambda:cfg,dump=dump,write=write)
 src=(V30/'adapter.py').read_text().replace('from io21 import *','')
 src=src.replace("if self.method!='original':assert audit['wrapper_hash']==cfg['checkpoints'][self.method]['module_hashes']['wrapper']","if self.wrapper is not None:assert audit['wrapper_hash']==cfg['checkpoints'][self.method]['module_hashes']['wrapper']")
 exec(compile(src,str(ROOT/'adapter31.py'),'exec'),ns)
 eng=ns['engine']();return eng,cfg
def runtime(cell):
 import torch
 eng,cfg=engine();rt=eng.Runtime(cfg,cell,torch.device('cuda:0'))
 if cell=='R11_center':
  def forward(self,x):
   r=self.output(self.blocks(self.input(x)));return (x+r-r.mean((2,3),keepdim=True)).clamp(0,1)
  rt.wrapper.forward=types.MethodType(forward,rt.wrapper)
 # Hash historical metric models before replacing only FID with the audited official extractor.
 rt.metric_models=(*rt.metric_models[:2],official_model())
 return eng,rt
def backend():
 import torch
 sys.path.insert(0,str(REPO));sys.path.insert(0,str(ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
 from src.utils.common import set_torch_env
 torch.set_num_threads(2);set_torch_env()
 return dict(setup_source=str(REPO/'src/utils/common.py'),source_sha256=sha(REPO/'src/utils/common.py'),deterministic=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
def decode(eng,rt,path,old=None):
 import io,torch
 ia,pa=rt.models(1088,1920);pa.clear_dpb();pa.set_curr_poc(0)
 before=[tensor_hash(m.state_dict()) for m in (ia,pa)];s=io.BytesIO(Path(path).read_bytes());helper=eng.eco.SPSHelper();hashes=[];qps=[]
 with torch.inference_mode():
  for i in range(64):
   start=s.tell();h=eng.eco.read_header(s)
   while h['nal_type']==eng.eco.NalType.NAL_SPS:
    helper.add_sps_by_id(eng.eco.read_sps_remaining(s,h['sps_id']));h=eng.eco.read_header(s)
   sp=helper.get_sps_by_id(h['sps_id']);q,bits=eng.eco.read_ip_remaining(s);qps.append(q)
   assert sp['height']==1088 and sp['width']==1920 and (h['nal_type']==eng.eco.NalType.NAL_I)==(i==0)
   if old is not None:assert q==old['actual_qps'][i]
   if i==0:
    dec=ia.decompress(bits,sp,q);pa.clear_dpb();pa.set_curr_poc(0);pa.add_ref_frame(None,dec['x_hat'])
   else:dec=pa.decompress(bits,sp,q)
   hashes.append(eng.eco.tensor_sha(dec['x_hat']));x=eng.core.unit(dec['x_hat'],1080,1920).float()
   assert x.shape==(1,3,1080,1920) and torch.isfinite(x).all()
   yield i,x,dict(actual_qp=q,real_bits=(s.tell()-start)*8,payload_bits=len(bits)*8)
 assert s.tell()==Path(path).stat().st_size
 h=eng.eco.sha256_bytes(''.join(hashes).encode())
 if old is not None:assert h==old['reconstruction_sha256']
 rt.assert_frozen(ia,pa);assert before==[tensor_hash(m.state_dict()) for m in (ia,pa)]
 rt.decode_audit=dict(status='PASS',bytes_consumed=s.tell(),reconstruction_sha256=h,actual_qps=qps,model_hashes_before=before,model_hashes_after=before)
