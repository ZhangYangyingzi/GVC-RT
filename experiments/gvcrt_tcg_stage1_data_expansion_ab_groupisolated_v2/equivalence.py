"""No optimizer updates: parent initialization and three native reader/flow/VJP audits."""
from ab_common import *
from ab_model_context import build,module,tensor_hash
def main():
 import torch
 torch.cuda.set_device(0)
 c=build(0);tr=c['tr'];pm=c['pm'];gen=c['models']['generator']
 from ab_model_context import frozen_items
 frozen=module('ab_equivalence_frozen',P/'runtime_acceleration.py')
 frozen_hash=frozen.frozen_hash(frozen_items(c));assert frozen_hash==c['parent']['frozen_state_hash']
 loss=module('ab_objective_check',ROOT/'loss.py').basic_crop
 from ab_runtime import Pipeline,optimized_loss,raw
 from ab_data_reader import PlanFrames,source_file
 frames32=module('ab_reference_frames32',REPO/'experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100/frames32.py')
 from reuse import parent_rows
 oldrows=parent_rows()
 early='--old-only' in sys.argv
 if early:brows=[dict(r,width=1920,height=1080) for r in oldrows]
 else:
  bindex=load(ROOT/'plans/B_expanded_ulong/index.json');brows=[r for s in bindex['shards'] for r in load(ROOT/s['file'])]
 oldids={r['source_sha256'] for r in oldrows}
 samples=[('UVG',next(r for r in brows if r['domain']=='uvg')),('old_ULong',next(r for r in brows if r['domain']=='ulong' and r['source_sha256'] in oldids))]
 if not early:samples.append(('new_ULong',next(r for r in brows if r['domain']=='ulong' and r['source_sha256'] not in oldids)))
 if not early:
  transition=load(ROOT/'repairs/native_geometry_v1/transition.json')
  assert sha(ROOT/'repairs/native_geometry_v1/transition.json')=='9b5c1529bbedeb4e73dc92e43d2cc6bd211c8c75e35481d56e758090ae3c9a0f'
  windows=ROOT/transition['affected_unique_windows_file'];assert sha(windows)==transition['affected_unique_windows_sha256']
  samples.append(('old_ULong_geometry_erratum',next(r for r in load(windows) if r['source_sha256']=='7886b72d96a7b36db8e8a6812d3dc453a7100e47fca82e5aad889366f0fb8183')))
 parent_source_metadata={v['sha256']:v for v in load(P/'plans/train_manifest_ulong.local.json')['videos']}
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
      rr=dict(row,source_path=str(path));native=parent_source_metadata.get(rr['source_sha256'],rr)
      frames32._SOURCES['ulong']={rr['video']:dict(path=str(path),sha256=rr['source_sha256'],width=native['width'],height=native['height'])}
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
 assert frozen.frozen_hash(frozen_items(c))==frozen_hash;frozen.full_frozen_hash_audit()
 if not early:
  assert subprocess.run([PYTHON,'-B','-u',str(ROOT/'engineering_rans.py')],cwd=REPO).returncode==0
  assert load(ROOT/'audits/rans_engineering.json')['status']=='PASS'
 dump(ROOT/('audits/early_runtime_old.json' if early else 'audits/runtime_equivalence.json'),dict(status='PASS',domains=reports,parent_TCG_tensors_exact=True,parent_frozen_state_sha256=frozen_hash,parent_frozen_state_exact=True,optimizer_step=5000,no_optimizer_update=True,loss_source_sha256=sha(ROOT/'loss.py'),parent_loss_source_sha256=sha(P/'loss.py'),RANS_instrumentation_audit=None if early else 'audits/rans_engineering.json'))
if __name__=='__main__':
 with lock('equivalence.lock'):main()
