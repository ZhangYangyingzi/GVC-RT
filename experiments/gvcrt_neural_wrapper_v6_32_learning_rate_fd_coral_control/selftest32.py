"""No landed optimizer step; common C1 and fixed distribution math gates."""
from io32 import *
import argparse,ast,textwrap,traceback,resource,fcntl,math,random
def historical_reference(gpu):
 import torch
 src=(V31/'probe_worker.py').read_text();tree=ast.parse(src);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 stop=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='opt' and n.value.func.attr=='step')
 fn.body=fn.body[:stop]+ast.parse(textwrap.dedent('''
 hook.remove()
 assert initial=={k:v4.module_hash(m) for k,m in models.items()}
 assert not opt.state
 return total,before,initial,deep
 ''')).body
 tree=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
 cfg=load(ROOT/'config.json');cfg['seed_probe']=load(ROOT/'plans/calibration.json')['seed']
 cfg['optimizer_effective'].update(source='historical effective AdamW')
 oldload=load
 ns=dict(globals());ns.update(uid=lambda t:'common_C1',receipt=lambda t:ROOT/'audits/common_C1.json',valid=lambda t:False,LAM=STAT,load=lambda p:cfg if Path(p)==ROOT/'config.json' else oldload(p),legacy_code=lambda n,f,overrides=None:globals()['legacy_code'](f,overrides))
 sys.path.insert(0,str(V31))
 exec(compile(tree,str(V31/'probe_worker.py'),'exec'),ns)
 row=load(ROOT/'plans/calibration.json')['updates'][0]
 return ns['main'](dict(method='C1',video=0,qp=row['external_qp']),gpu)
def math_checks():
 import torch
 from distribution32 import moments,reference,losses,commit
 torch.manual_seed(20261032);z=torch.randn(31,8,dtype=torch.float64);ref=reference(z);mu,M,_=moments(z);old=dict(mu=mu,M=M,counter=0)
 same,_=losses(z,old,ref);assert abs(float(same['FD']))<1e-9 and float(same['CORAL'])<1e-20
 x=(z[:15]+.3).requires_grad_();out,next_state=losses(x,old,ref);rows=[]
 for n in ('FD','mean','CORAL'):
  g=torch.autograd.grad(out[n],x,retain_graph=True)[0];assert torch.isfinite(g).all() and g.norm()>0
  v=torch.randn_like(x);v=v/v.norm();h=1e-5
  plus=losses(x.detach()+h*v,old,ref)[0][n];minus=losses(x.detach()-h*v,old,ref)[0][n];fd=(plus-minus)/(2*h);ad=(g*v).sum();rel=float((fd-ad).abs()/ad.abs().clamp_min(1e-8));assert rel<1e-4
  rows.append(dict(loss=n,value=float(out[n]),directional_finite_difference=float(fd),autograd=float(ad),relative_error=rel))
 assert old['counter']==0 and commit(old,next_state)['counter']==1
 # Small FP32 network, feature leaf VJP, same exact parameter collection.
 layer=torch.nn.Linear(5,8).float();inp=torch.randn(15,5);direct=layer(inp);dout,_=losses(direct,old,ref);checks=[]
 for name in ('FD','mean','CORAL'):
  dg=torch.autograd.grad(dout[name],list(layer.parameters()),retain_graph=True)
  leaf=direct.detach().clone().requires_grad_();value=losses(leaf,old,ref)[0][name];gz=torch.autograd.grad(value,leaf)[0].detach();replayed=layer(inp);vg=torch.autograd.grad(replayed,list(layer.parameters()),grad_outputs=gz)
  num=sum(float((a-b).double().square().sum()) for a,b in zip(dg,vg))**.5;den=sum(float(a.double().square().sum()) for a in dg)**.5;rel=num/max(den,1e-12);assert rel<=1e-6;checks.append(dict(loss=name,parameter_relative_L2=rel))
 dump(ROOT/'audits/distribution_math.json',dict(status='PASS',identity={k:float(v) for k,v in same.items()},directional_checks=rows,two_pass_toy=checks,optimizer_updates=0,EMA_commit_once=True))
def main(gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch
 from training32 import setup,seed,crop_c1,norm,groups,check_frozen,official_model,backend
 from frames32 import PlanFrames,audit_resolution
 settings=backend();math_checks();bad=audit_resolution()
 row=dict(load(ROOT/'plans/calibration.json')['updates'][0],calibration_index=0)
 dump(ROOT/'probe_manifest.json',dict(updates=[row]))
 torch.cuda.reset_peak_memory_stats();reference,before,initial,deep=historical_reference(gpu)
 ctx=setup();seed(load(ROOT/'plans/calibration.json')['seed']);frames=PlanFrames(row,'cuda:0');grad,sums,trace=crop_c1(ctx,frames,row['external_qp']);frames.close()
 rel=norm([a-b for a,b in zip(grad,reference)])/max(norm(reference),1e-12);assert rel<=1e-6,('C1 historical fused gradient mismatch',rel)
 for a,b in zip(trace,before):
  assert a['actual_qp']==b['actual_qp']
  for n in ('LPIPS','DISTS','reconstruction_L1'):assert abs(a[n]-b[n])<=1e-7
 assert all(x['finite'] and x['norm']>0 for x in groups(ctx[3],grad).values())
 check_frozen(ctx);assert initial=={k:ctx[0].module_hash(m) for k,m in ctx[3].items()}
 # Analytic first AdamW increment, no extra model update or optimizer.step.
 oc=load(ROOT/'config.json')['optimizer_effective'];lr_checks=[]
 for lr in (3e-6,5e-7):
  g=torch.tensor([.001,-.02,.07],dtype=torch.float64);p=torch.tensor([.2,-.3,.7],dtype=torch.float64)
  delta=lambda r: -r*1e-4*p-r*g/(g.abs()+oc['eps'])
  err=float((delta(2*lr)-2*delta(lr)).abs().max());assert err<1e-15;lr_checks.append(dict(lr=lr,twice_lr_delta_error=err))
 official=official_model();x=torch.rand(1,3,128,128,device='cuda');x.requires_grad_();z=official(x)
 with torch.no_grad():z0=official(x)
 assert torch.equal(z,z0);gx=torch.autograd.grad(z.square().mean(),x)[0];assert gx.isfinite().all() and gx.norm()>0
 assert all(not p.requires_grad and p.grad is None for p in official.parameters())
 dump(ROOT/'audits/common.json',dict(status='PASS',historical_C1_fused_gradient_relative_L2=rel,analytic_learning_rate_checks=lr_checks,source_and_frozen_hashes=ctx[-1],deep_DISTS=deep,Inception_input_gradient_norm=float(gx.norm()),Inception_grad_forward_identical=True,optimizer_updates=0,D_updates=0,backend=settings,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,full_resolution_conflicts=len(bad),A_B_crop_training_gate='PASS',C_D_gate='BLOCKED' if bad else 'PENDING_FULL_REPLAY'))
 print('COMMON PASS; distribution gate', 'BLOCKED' if bad else 'PENDING',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True,choices=(4,5,6,7));a=ap.parse_args()
 try:main(a.gpu)
 except Exception:error(dict(stage='selftest',error=traceback.format_exc()));raise
