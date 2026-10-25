"""Readonly paired old/new decoding for reused N0 points lacking latent/qmap logs."""
from util import *
import io,importlib.util,gc

def main():
 gpu_env(6);sys.path.insert(0,str(P));import train
 train.setup_path()
 import torch
 from gvc_hooks import load_models as original_load
 from model_io import load_checkpoint,cdf_hashes
 from src.utils import stream_helper as sh
 torch.set_num_threads(2)
 spec=importlib.util.spec_from_file_location('readonly_v2_h11_runtime',V2/'runtime.py');old_runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_runtime)
 entries=[x for x in load(P/'reuse_provenance.json')['evaluation_records'] if x['method']=='N0' and x['legacy_P_latent_qmap_flag']=='not_recorded'];results=[]
 for e in entries:
  original=load(e['source_path']);path=Path(original['stream_path']);assert sha(path)==e['stream_sha256']
  oi,op=original_load(torch.device('cuda:0'),force_zero_thres=.12)
  init=torch.load(train.INIT,map_location='cpu',weights_only=True)
  op.recon_generation_net.mlp.load_state_dict(init['bridge'],strict=True);op.recon_generation_net.decoder.load_state_dict(init['generator'],strict=True)
  op=old_runtime.upgrade(op,'A');op.recon_generation_net.mlp.float();op.recon_generation_net.decoder.float()
  cp=legacy_checkpoint('N0',e['step']);st=torch.load(cp,map_location='cpu',weights_only=False);op.recon_generation_net.decoder.load_state_dict(st['G'],strict=True)
  oi.eval().requires_grad_(False);op.eval().requires_grad_(False)
  ni,np,ck=load_checkpoint('N0',e['step']);assert ck==e['checkpoint_sha256']
  assert cdf_hashes(op)==cdf_hashes(np)
  for m in (op,np):m.clear_dpb();m.set_curr_poc(0)
  f=io.BytesIO(path.read_bytes());helper=sh.SPSHelper();rows=[]
  with torch.inference_mode():
   for i in range(64):
    h=sh.read_header(f)
    while h['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(f,h['sps_id']));h=sh.read_header(f)
    s=helper.get_sps_by_id(h['sps_id']);q,b=sh.read_ip_remaining(f)
    for a,p in [(oi,op),(ni,np)]:
     if i==0:
      d=a.decompress(b,s,q);p.clear_dpb();p.set_curr_poc(0);p.add_ref_frame(None,d['x_hat'])
     else:d=p.decompress(b,s,q)
     if p is op:old_x=d['x_hat'].clone()
     else:assert torch.equal(old_x,d['x_hat'])
    if i:
     assert torch.equal(op.decoding_trace['y_hat'],np.decoding_trace['y_hat']) and torch.equal(op.q_actual,np.q_actual)
     assert torch.equal(op.dpb[0].feature,np.dpb[0].feature)
    rows.append(dict(frame=i,actual_qp=q,reconstruction_exact=True,reference_exact=True,latent_qmap_exact=i>0))
  assert f.tell()==path.stat().st_size
  results.append(dict(method='N0',source=e['source_path'],source_sha256=e['source_sha256'],checkpoint_sha256=ck,stream_sha256=sha(path),frames=rows,status='PASS',new_optimization_updates=0,read_only_extended_audit=True))
  dump(P/'audits/reused_N0_extended_decode.json',dict(status='RUNNING',complete=len(results),planned=len(entries),records=results))
  del oi,op,ni,np,st,init;gc.collect();torch.cuda.empty_cache();print('READONLY_REUSE_DECODE_PASS',e['step'],e['sequence'],e['QP'],flush=True)
 dump(P/'audits/reused_N0_extended_decode.json',dict(status='PASS',complete=len(results),planned=len(entries),records=results,old_table_values_unchanged=True))
if __name__=='__main__':main()
