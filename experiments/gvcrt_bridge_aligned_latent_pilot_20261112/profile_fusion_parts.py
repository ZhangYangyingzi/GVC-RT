"""Measure adapter and reliability separately on the actual frozen native input shape."""
from util import *
import torch,time,statistics
from model import AlignedLatent,bridge
from restore import restore_b300

def run():
 setup();c=restore_b300();stats=readtensor(ROOT/'cache/statistics.pt');m=AlignedLatent(stats,'T_aligned').cuda();m.load_state_dict(readtensor(ROOT/'checkpoints/T_aligned/step_0300.pt')['model']);m.eval();d=waitcache(manifest()['dev'][0]['id'],'profile_parts',5)
 with torch.no_grad():
  for i in [7,8]:
   f=d['features'][i].cuda().float();l=bridge(c,f);_,a=m(l,f,d['actual_qps'][i],d['id'],i)
  ln=l/m.fusion.l_rms;an=a['a']/m.fusion.l_rms;z=torch.cat((ln,an,(ln-an).abs()),1);b,_,h,w=l.shape
  qp=l.new_tensor([d['actual_qps'][8],d['actual_qps'][7]]).view(1,2,1,1).expand(b,-1,h,w);qp=(qp-m.fusion.qp_mean)/m.fusion.qp_std;zz=torch.cat((z,a['e'],qp,a['v']),1)
  def measure(fn):
   for _ in range(3):fn()
   vals=[]
   for _ in range(30):
    torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();vals.append((time.perf_counter()-t)*1000)
   return dict(mean_ms=statistics.mean(vals),median_ms=statistics.median(vals),samples_ms=vals)
  results=dict(adapter_only=measure(lambda:m.fusion.adapter(z)*m.fusion.l_rms),reliability_only=measure(lambda:a['v']*torch.sigmoid(m.fusion.reliability(zz))))
 r=load(ROOT/'audits/runtime_profile.json');r['component_timings'].update(results);r['separate_parts_scope']='Actual frozen B300 current and predicted-aligned original previous latent, full native padded shape; FP32; GPU0; 30 synchronized repeats; no training updates';dump(ROOT/'audits/runtime_profile.json',r);dump(ROOT/'audits/fusion_parts_timing.json',dict(status='PASS',physical_gpu=0,results=results,updates=0));event('FUSION_PARTS_PROFILE_COMPLETE',**{k:v['mean_ms'] for k,v in results.items()})
if __name__=='__main__':run()
