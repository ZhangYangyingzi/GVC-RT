"""No optimizer updates: parent initialization and three native reader/flow/VJP audits."""
from ab_common import *
from ab_model_context import build,module,tensor_hash
def main():
 import torch
 torch.cuda.set_device(0)
 c=build(0);tr=c['tr'];pm=c['pm'];gen=c['models']['generator']
 loss=module('ab_objective_check',ROOT/'loss.py').basic_crop
 from ab_runtime import Pipeline,optimized_loss,raw
 from ab_data_reader import PlanFrames,source_file
 frames32=module('ab_reference_frames32',REPO/'experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100/frames32.py')
 oldrows=module('ab_source_audit',ROOT/'audit.py').rows()
 early='--old-only' in sys.argv
 if early:brows=[dict(r,width=1920,height=1080) for r in oldrows]
 else:
  bindex=load(ROOT/'plans/B_expanded_ulong/index.json');brows=[r for s in bindex['shards'] for r in load(ROOT/s['file'])]
 oldids={r['source_sha256'] for r in oldrows}
 samples=[('UVG',next(r for r in brows if r['domain']=='uvg')),('old_ULong',next(r for r in brows if r['domain']=='ulong' and r['source_sha256'] in oldids))]
 if not early:samples.append(('new_ULong',next(r for r in brows if r['domain']=='ulong' and r['source_sha256'] not in oldids)))
 reports=[]
 for label,row in samples:
  rng=tr.rng_state();outputs=[]
  class LegacyFrames:
   def __init__(self):
    if row['domain']=='uvg':
     frames32._SOURCES['uvg']={v['name']:v for v in load(P/'plans/train_manifest_uvg.local.json')['videos']}
     source=frames32.PlanFrames(row,torch.device('cuda',0))
     try:self.frames=[source[i] for i in range(64)]
     finally:source.close()
    else:
     with source_file(row) as path:
      rr=dict(row,source_path=str(path));frames32._SOURCES['ulong']={rr['video']:dict(path=str(path),sha256=rr['source_sha256'],width=rr['width'],height=rr['height'])}
      source=frames32.PlanFrames(rr,torch.device('cuda',0))
      try:self.frames=[source[i] for i in range(64)]
      finally:source.close()
   def __getitem__(self,i):return self.frames[i]
   def close(self):self.frames=None
  def run(frames,fn):
   tr.restore_rng(rng);hashes=[];hook=gen.register_forward_hook(lambda m,args,out:hashes.append(tensor_hash([('RGB',out)])))
   try:gs,summary=fn(c['ctx'],frames,row['external_qp'],dists_weight=.5,raft=c['raft'],raft_transform=c['transform'])
   finally:hook.remove();frames.close();pm.clear_dpb()
   return torch.cat([g.flatten() for g in gs]),summary,hashes
  # Same parent model evaluated through the original scalar RAFT loss path.
  original,stat,hashes=run(LegacyFrames(),loss)
  repeated,stat2,hashes2=run(LegacyFrames(),loss)
  pipeline=Pipeline([row]*4,0,0)
  frames=pipeline.next(row)
  # Check every GPU input exactly against the inherited PlanFrames tensor path.
  reference=LegacyFrames();inputs_exact=all(torch.equal(frames[i],reference[i]) for i in range(64));reference.close();assert inputs_exact
  targets=[]
  with torch.no_grad():
   previous=frames[48]
   for pos in range(49,64):
    current=frames[pos];a,b=c['transform'](previous,current);targets.append(c['raft'](a,b)[-1]);previous=current
  flows_exact=all(torch.equal(f,frames.target_flows[i]) for i,f in enumerate(targets));assert flows_exact
  optimized,faststat,fasthashes=run(frames,optimized_loss(loss));pipeline.close()
  repeat_error=float((original-repeated).double().norm()/original.double().norm())
  relative=float((original-optimized).double().norm()/original.double().norm())
  assert relative<1e-3 and repeat_error<1e-3
  assert stat['losses']==faststat['losses'] and hashes==fasthashes and len(hashes)==63
  reports.append(dict(domain=label,status='PASS',inputs_bit_exact=inputs_exact,target_flows_bit_exact=flows_exact,losses_bit_exact=True,parent_same_input_output_hashes=hashes,parent_same_input_output_match=True,gradient_relative_L2=relative,original_repeat_gradient_relative_L2=repeat_error,tolerance=1e-3,no_optimizer_update=True,source_sha256=row['source_sha256']))
 assert all(torch.equal(gen.state_dict()[n].detach().cpu(),v) for n,v in c['parent']['tcg_model_state'].items())
 dump(ROOT/('audits/early_runtime_old.json' if early else 'audits/runtime_equivalence.json'),dict(status='PASS',domains=reports,parent_TCG_tensors_exact=True,optimizer_step=5000,no_optimizer_update=True,loss_source_sha256=sha(ROOT/'loss.py'),parent_loss_source_sha256=sha(P/'loss.py')))
if __name__=='__main__':main()
