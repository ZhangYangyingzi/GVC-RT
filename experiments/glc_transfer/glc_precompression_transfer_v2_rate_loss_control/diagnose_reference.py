"""First-difference diagnostic on a fixed training canary, zero updates."""
from preflight import *
from data import Frames,load
from bootstrap import V1
def main():
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']='7'
 import torch
 from runtime import models,Sender,step
 from src.entropy_coder import GaussianRansCodec
 im,pm=models('cuda:0');p=Sender('C_latent_P').cuda().eval();p.load_state_dict(torch.load(V1/'final_P/C_latent_P_1000.pt',map_location='cpu',weights_only=False)['P'],strict=True);f=Frames(load(R/'plans/rate_audit.json')['rows'][0],True);coder=GaussianRansCodec();dpb=None;capture={};records=[]
 def record(m,args,result):capture['test_z']=result[0].detach().clone();capture['test_indices']=result[2]['min_encoding_indices'].detach().clone()
 h=pm.z_vq.register_forward_hook(record);original=pm.z_vq.get_quan_feat
 def get(indices,shape):
  result=original(indices,shape);capture['compress_z']=result.detach().clone();capture['compress_indices']=indices.detach().clone();return result
 pm.z_vq.get_quan_feat=get
 with torch.inference_mode():
  for pos in range(64):
   capture.clear();x=f.tensor(pos,'cuda:0');a,rec,rd,_=step(im,pm,x,dpb,0,pos,p);b,out,dpb,_=step(im,pm,x,dpb,0,pos,p,mode='compress',coder=coder)
   r=dict(frame=pos,reconstruction_max_difference=float((rec-out).abs().max()),reconstruction_exact=torch.equal(rec,out))
   if pos:r.update(categorical_indices_equal=torch.equal(capture['test_indices'],capture['compress_indices']),categorical_value_max_difference=float((capture['test_z']-capture['compress_z']).abs().max()),categorical_value_exact=torch.equal(capture['test_z'],capture['compress_z']))
   records.append(r)
   if not r['reconstruction_exact']:break
 h.remove();f.close();save('audits/reference_first_difference.json',dict(status='RECORDED',source='installed GLC test VQ forward vs unchanged compress get_quan_feat',records=records,optimizer_updates=0,independent_decode_checks_not_relaxed=True));print(json.dumps(records),flush=True)
if __name__=='__main__':main()
