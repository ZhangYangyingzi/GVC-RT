"""Both extractors receive the exact same FP32 RGB tensor from a real decoder."""
from io21 import *
import argparse,io

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);ap.add_argument('--dataset',choices=('ulong','hevc_b','mcljcv'),required=True);a=ap.parse_args()
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
 import torch,numpy as np
 torch.set_num_threads(2)
 from adapter import engine,frames_for
 from audit_statistics import historical_point
 vendor=ROOT/'vendor/pytorch-fid';sys.path.insert(0,str(vendor/'src'))
 from pytorch_fid.inception import InceptionV3
 import pytorch_fid.inception as official_source
 weight=Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')
 official_source.load_state_dict_from_url=lambda *a,**k:torch.load(weight,map_location='cpu',weights_only=True)
 official=InceptionV3([3],resize_input=True,normalize_input=True,requires_grad=False,use_fid_inception=True).to('cuda').eval().requires_grad_(False)
 old=module('v629_historical_fid',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py').InceptionPool3().to('cuda').eval().requires_grad_(False)
 before=[tensor_hash(m.state_dict()) for m in (official,old)]
 eng=engine();cfg=evalcfg();rows=[];source=[]
 for m in ('original','B1_initial'):
  runtime=eng.Runtime(dict(experiment='B',IP=-1,methods={m:[None,None] if m=='original' else [m,m]},checkpoints={} if m=='original' else {m:cfg['checkpoints'][m]},force_zero_thres=.12),m,torch.device('cuda:0'))
  for q in (0,5,9):
   target=ROOT/'audits/features'/a.dataset/m/f'qp{q}.json'
   if target.exists():continue
   records=[];f0=[];f1=[]
   for v in sources(a.dataset):
    p=historical_point(a.dataset,m,v['video_index'],q);r=load(p);path=Path(r['bitstream_path']);assert sha(path)==r['bitstream_sha256']
    frames=frames_for(v);h,w=frames[0].shape[-2:];ia,pa=runtime.models((h+63)//64*64,(w+63)//64*64);pa.clear_dpb();pa.set_curr_poc(0);stream=io.BytesIO(path.read_bytes());helper=eng.eco.SPSHelper()
    for i,cpu in enumerate(frames):
     header=eng.eco.read_header(stream)
     while header['nal_type']==eng.eco.NalType.NAL_SPS:
      helper.add_sps_by_id(eng.eco.read_sps_remaining(stream,header['sps_id']));header=eng.eco.read_header(stream)
     sps=helper.get_sps_by_id(header['sps_id']);actual,bits=eng.eco.read_ip_remaining(stream)
     with torch.inference_mode():
      if i==0:dec=ia.decompress(bits,sps,actual);pa.clear_dpb();pa.set_curr_poc(0);pa.add_ref_frame(None,dec['x_hat'])
      else:dec=pa.decompress(bits,sps,actual)
      rec=eng.core.unit(dec['x_hat'],h,w)
      for kind,x in (('GT',cpu.to('cuda')),('reconstruction',rec)):
       x=x.float();assert x.shape==(1,3,1080,1920)
       z0=old(x).cpu().numpy();z1=official(x)[0].flatten(1).cpu().numpy();delta=z1-z0;relative=float(np.linalg.norm(delta)/max(np.linalg.norm(z0),1e-12));ma=float(np.abs(delta).max());tol=1e-4+1e-4*float(np.abs(z0).max())
       records.append(dict(sequence=v['name'],frame=i,source_frame=v['source_frame_indices'][i],kind=kind,input_tensor_sha256=tensor_hash({'RGB':x}),relative_L2=relative,max_abs=ma,max_abs_tolerance=tol,PASS=relative<=1e-4 and ma<=tol))
       f0.append(z0);f1.append(z1)
    assert stream.tell()==path.stat().st_size
    runtime.assert_frozen(ia,pa);del ia,pa;torch.cuda.empty_cache()
   assert before==[tensor_hash(x.state_dict()) for x in (official,old)]
   dump(target,dict(status='PASS' if all(x['PASS'] for x in records) else 'FAIL',dataset=a.dataset,method=m,QP=q,frames=len(records)//2,records=records,exact_same_float_RGB=True,quantized_PNG_used=False,source_code_sha256=sha(vendor/'src/pytorch_fid/inception.py'),weight_path=str(weight),weight_sha256=sha(weight),official_commit=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip(),precision='FP32',resize='299 bilinear align_corners=False',normalization='2*RGB-1',physical_GPU=a.gpu,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
   print('FEATURE AUDIT',a.dataset,m,q,len(records),'PASS' if all(x['PASS'] for x in records) else 'FAIL',flush=True)
  del runtime;torch.cuda.empty_cache()
if __name__=='__main__':main()
