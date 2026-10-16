from pathlib import Path
import shutil,json,hashlib,subprocess,time
repo=Path('/Huang_group/zyyz/Projects/GVC-RT');old=repo/'experiments/gvcrt_neural_wrapper_v6_32_learning_rate_fd_coral_control';new=repo/'experiments/gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion'
assert not new.exists();new.mkdir()
for name in ['io32.py','frames32.py','training32.py','distribution32.py','evaluate32.py','adapter32.py','plot32.py','selftest32.py','report32.py','publish32.py','evaluation_config.json','eval_manifest.json','task_manifest.json','qp_semantics_audit.json','train_manifest_ulong.json','train_manifest_uvg.json','source_hashes.json','checkpoint_audit.json','objective_spec.json']:
 p=old/name
 if p.exists():shutil.copy2(p,new/name)
for d in ['plans','audits','branches','logs','tasks','reference_stats/receipts','calibration','evaluation','features','results']:(new/d).mkdir(parents=True,exist_ok=True)
shutil.copytree(old/'plans',new/'plans',dirs_exist_ok=True)
for b in ['A_C1_lr1','B_C1_lr2']:
 (new/'branches'/b).mkdir();shutil.copy2(old/'branches'/b/'checkpoint_index.json',new/'branches'/b/'checkpoint_index.json')
for name in ['FD_source.json','common.json']:
 if (old/'audits'/name).exists():shutil.copy2(old/'audits'/name,new/'audits'/('historical_'+name))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
c=json.loads((old/'config.json').read_text());c.update(schema='v632b',experiment=new.name,expected_points=3312,checkpoint_steps=[0,1,250,500,1000],reference_commit='6bd337cff6b15a20891636003cfbcef65456b935',actual_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),native_auxiliary_geometry=True,historical_source=str(old),expected=dict(committed_updates=4000,new_committed_updates=2000,prefill=1000,calibration=40,logical_points=3312,new_encodes=1196,historical_encodes=2116,baseline=920),distribution_backend='float64_cholesky_cross_svd',latest_every=1)
dump(new/'config.json',c)
p=json.loads((old/'protocol.json').read_text());p.update(experiment=new.name,schema='v632b',updates=1000,new_branches=c['branches'][2:],historical_branches=c['branches'][:2],methods=c['methods'],expected_points=3312,auxiliary_view='native valid RGB HxW, official codec alignment padding, unpad before Inception; crop unchanged',full_frame_restriction_1920x1080=False,formal_evaluation_view='unchanged 1920x1080 RGB / 1920x1088 canvas',new_optimizer_updates=2000,discriminator_optimizer_updates=0,seed_model=c['source_checkpoint'],statistics_ddof=0,evaluation_ddof=1,EMA_beta=.95,EMA_commit='exactly once with atomic successful model+optimizer checkpoint',calibration='40 fixed training windows; no optimizer step; shared detached seed per window',latest_every=1,latest_reason='save every successful step for atomic optimizer/EMA transaction, stronger than required 25-step recovery interval',no_training_or_test_selection=True)
dump(new/'protocol.json',p)
assert sha(new/'plans/shared.json')=='e83011570849e027883fc9e34e7bf04907e79dabbcfb89382a50571f774902db'
assert sha(c['source_checkpoint']['path'])==c['source_checkpoint']['sha256']
s=(new/'io32.py').read_text().replace("V29=ROOT.parent", "OLD32=ROOT.parent/'gvcrt_neural_wrapper_v6_32_learning_rate_fd_coral_control'\nV29=ROOT.parent",1)
s=s.replace("if kind=='train':names+=['train.py']", "if kind=='train':names+=['train.py','calibration32b.py']")
s=s.replace("if kind in ('prefill','calibrate','selftest'):names+=['statistics32.py','selftest32.py']", "if kind in ('prefill','calibrate','selftest'):names+=['statistics32b.py','calibration32b.py']")
(new/'io32.py').write_text(s)
s=(new/'frames32.py').read_text().replace('self.cap=None;self.next=None','self.cap=None;self.next=None;self.geometry=None')
s=s.replace("if self.full:assert a.shape==(1080,1920,3),('FULL_RESOLUTION_MISMATCH',r['video'],a.shape)","if self.geometry is None:self.geometry=a.shape\n  assert a.shape==self.geometry and a.shape==(self.v.get('height',1080),self.v.get('width',1920),3),('NATIVE_GEOMETRY_MISMATCH',r['video'],a.shape,self.v.get('height'))")
a=s.index('def full_eligible(row):');s=s[:a]+'''def full_eligible(row):
 v=metadata(row)
 return v.get('width',1920)>0 and v.get('height',1080)>0
'''
(new/'frames32.py').write_text(s)
s=(new/'training32.py').read_text().replace('gt=frames[0];assert gt.shape==(1,3,1080,1920)',"gt=frames[0];h,w=gt.shape[-2:];right,bottom=pm.get_padding_size(h,w,16);padding=(0,right,0,bottom)")
s=s.replace("F.pad(gt,(0,0,0,8),mode='replicate')","F.pad(gt,padding,mode='replicate')").replace("F.pad(proxy,(0,0,0,8),mode='replicate')","F.pad(proxy,padding,mode='replicate')").replace('rgb=rec[:,:,:1080,:1920]','rgb=rec[:,:,:h,:w];assert gt.shape==(1,3,h,w)')
# Batched objective VJPs share the same replay graph and never apply an extra /15.
s=s.replace('feature_grad=None,expected=None):','feature_grad=None,expected=None):')
s=s.replace('hook=pm.dec.register_forward_hook(lambda _m,_a,o:capt.append(o))\n try:',"multi=isinstance(feature_grad,dict)\n if multi:gs={name:zero(params) for name in feature_grad}\n hook=pm.dec.register_forward_hook(lambda _m,_a,o:capt.append(o))\n try:")
s=s.replace("g=feature_grad[pos-49].to(z.device,dtype=z.dtype).reshape_as(z)\n      gg=torch.autograd.grad(z,params,grad_outputs=g*4096,allow_unused=True);add(gs,gg,4096)","items=list(feature_grad.items()) if multi else [('single',feature_grad)]\n      for j,(name,values) in enumerate(items):\n       g=values[pos-49].to(z.device,dtype=z.dtype).reshape_as(z)\n       gg=torch.autograd.grad(z,[rgb]+params,grad_outputs=g*4096,allow_unused=True,retain_graph=j<len(items)-1)\n       assert gg[0] is not None and torch.isfinite(gg[0]).all() and gg[0].float().norm()>0,('RGB_VJP',name,pos)\n       replay.append(dict(position=pos,objective=name,RGB_gradient_norm=float(gg[0].float().norm()/4096),RGB_gradient_finite=True))\n       add(gs[name] if multi else gs,gg[1:],4096)")
(new/'training32.py').write_text(s)
(new/'.gitignore').write_text('*.pt\n*.npz\n*.bin\n*.lock\n*.pid\n__pycache__/\nfeatures/\ncache/\nlogs/\nreference_stats/*.pt\nreference_stats/strata/\nbranches/*/receipts/\n*.tmp\npublication/\n')
deps={str(p):dict(sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in old.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md')}
dump(new/'audits/historical_before.json',deps)
dump(new/'audits/preflight.json',dict(status='PASS',actual_HEAD=c['actual_HEAD'],source_checkpoint=c['source_checkpoint'],shared_plan_sha256=sha(new/'plans/shared.json'),calibration_plan_sha256=sha(new/'plans/calibration.json'),old_source=str(old),timestamp=time.time(),new_optimizer_updates_budget=2000))
print(new)
