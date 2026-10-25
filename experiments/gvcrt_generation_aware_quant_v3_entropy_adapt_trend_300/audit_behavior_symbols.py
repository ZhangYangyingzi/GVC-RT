"""Read existing three preflight streams to retain their exact decoded symbol records."""
from util import *
import io,gc

def main():
 gpu_env(4);import train;train.setup_path()
 import torch
 from model_io import load_checkpoint,assert_cdf
 from src.utils import stream_helper as sh
 torch.set_num_threads(2);rows=[];records=[]
 for path in sorted((P/'server_artifacts/fixed_multiplier_check').glob('*.json')):
  e=load(path);assert e['status']=='PASS';m=e['fixed_multiplier_check'];stream=Path(e['stream_path']);assert sha(stream)==e['stream_sha256']
  im,pm,ck=load_checkpoint('Q1',0)
  class Constant(torch.nn.Module):
   def forward(self,common,qp):return torch.full((common.shape[0],1,*common.shape[-2:]),m,device=common.device,dtype=torch.float32)
  pm.quant_adjust=Constant();assert assert_cdf(pm)==e['probability_decoder']
  pm.clear_dpb();pm.set_curr_poc(0);f=io.BytesIO(stream.read_bytes());helper=sh.SPSHelper()
  with torch.inference_mode():
   for i in range(64):
    header=sh.read_header(f)
    while header['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(f,header['sps_id']));header=sh.read_header(f)
    sps=helper.get_sps_by_id(header['sps_id']);qp,packet=sh.read_ip_remaining(f)
    if i==0:
     d=im.decompress(packet,sps,qp);pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,d['x_hat'])
    else:
     d=pm.decompress(packet,sps,qp);t=pm.decoding_trace
     r=dict(multiplier=m,frame=i,actual_QP=qp,actual_P_payload_bytes=len(packet),stream_sha256=e['stream_sha256'])
     for j in (0,1):
      v=t[f'q{j}_read'].to(torch.int16).cpu();r.update({f'pass{j}_symbol_sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest(),f'pass{j}_symbol_count':v.numel(),f'pass{j}_nonzero_symbols':int((v!=0).sum()),f'pass{j}_symbol_min':int(v.min()),f'pass{j}_symbol_max':int(v.max())})
     r.update(qmap_sha256=state_hash({'qmap':t['qmap']}),rho_mean=float(t['rho'].float().mean()))
     rows.append(r)
  assert f.tell()==stream.stat().st_size
  records.append(dict(multiplier=m,stream_sha256=e['stream_sha256'],checkpoint_sha256=ck,frames=64,P_symbol_records=63,new_training_updates=0))
  del im,pm,d;gc.collect();torch.cuda.empty_cache()
 write(P/'behavior_symbols.csv',rows);dump(P/'audits/behavior_symbols.json',dict(status='PASS',records=records,source='existing preflight multiplier files; no new encoding or optimization',symbol_hash_dtype='int16'))
 print('BEHAVIOR_SYMBOLS_PASS',len(rows),flush=True)
if __name__=='__main__':main()
