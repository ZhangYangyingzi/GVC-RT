"""No-update step0 codec replay, branch loss/gradient, and frozen-state checks."""
from preflight import *
from data import load,Frames
from bootstrap import V1,BRANCHES,SOURCE_SHA
import argparse,importlib.util
from contextlib import contextmanager
@contextmanager
def full_fp32_backward_check():
 import torch
 flags=dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
 try:
  torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
  yield flags
 finally:
  torch.backends.cudnn.allow_tf32=flags['cudnn_allow_tf32'];torch.backends.cuda.matmul.allow_tf32=flags['matmul_allow_tf32']
def run(gpu):
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 import torch
 from runtime import Sender,models,quality,step,losses,ExactCategoricalSTE
 from train import seed,rng,restore,norm
 from rate_audit import refhash
 from rate_backend import install
 from src.entropy_coder import GaussianRansCodec,make_gop_stream
 seed(20261016);random_state=rng();source=V1/'final_P/C_latent_P_1000.pt';assert sha(source)==SOURCE_SHA
 toy=torch.randn(2,3,requires_grad=True,device='cuda:0');hard=torch.randn_like(toy);exact=ExactCategoricalSTE.apply(toy,hard);assert torch.equal(exact,hard);assert torch.equal(torch.autograd.grad(exact.sum(),toy)[0],torch.ones_like(toy))
 spec=importlib.util.spec_from_file_location('readonly_v1_runtime_replay',V1/'runtime.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 p=Sender('C_latent_P').cuda().eval();p.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['P'],strict=True)
 previous=old.Sender('C_latent_P').cuda().eval();previous.load_state_dict(p.state_dict(),strict=True);original=state_hash(p.state_dict());im,pm=models('cuda:0');ql=quality('cuda:0');fixed=[state_hash(m.state_dict()) for m in (im,pm,*ql)];coder=GaussianRansCodec();replays=[];probe=None
 for index in (0,18):
  row=load(R/'plans/rate_audit.json')['rows'][index];f=Frames(row,True);a=None;b=None;packets=[]
  with torch.no_grad():
   for pos in range(64):
    x=f.tensor(pos,'cuda:0')
    if pos==49:probe=(x,dict(b),row['q'])
    e,rec,a,_=old.step(im,pm,x,a,row['q'],pos,previous,mode='compress',coder=coder);n,out,b,_=step(im,pm,x,b,row['q'],pos,p,mode='compress',coder=coder)
    assert torch.equal(rec,out) and e['bit_stream']==n['bit_stream'] and refhash(a)==refhash(b)
    packets.append(n['bit_stream'])
  stream=make_gop_stream(256,256,packets);h=hashlib.sha256(stream).hexdigest();paired=load(R/f'audits/rate_pairs/{index:02d}_C_latent_P.json');assert h==paired['stream_sha256'] and paired['independent_decode']
  replays.append(dict(index=index,q=row['q'],frames=64,stream_bytes=len(stream),stream_sha256=h,paired_independent_decode_reference=str(R/f'audits/rate_pairs/{index:02d}_C_latent_P.json'),strict_byte_reconstruction_and_reference_equality=True));f.close()
 backend=load(R/'audits/rate_backend_decision.json')['backend'];install(pm,backend,coder)
 x,reference,q=probe;e,out,_,_=step(im,pm,x,reference,q,49,p);terms=losses(out,x,ql);terms['rate']=e['bpp'].mean();params=list(p.parameters());grad={}
 # Audit only: reduced-mantissa convolution backwards need not be additive at
 # 1e-4. Check the VJP identity in full FP32 on the SAME inherited forward graph,
 # then restore flags; deployed/training precision and decoder checks are intact.
 with full_fp32_backward_check() as inherited_flags:
  for k,v in terms.items():
   gg=torch.autograd.grad(v,params,retain_graph=True,allow_unused=True);grad[k]=[g if g is not None else torch.zeros_like(t) for g,t in zip(gg,params)]
   assert norm(grad[k])>0 and all(torch.isfinite(g).all() for g in grad[k])
  beta=load(R/'calibration/final.json')['beta'][str(q)];checks=[]
  for branch,cfg in BRANCHES.items():
   weights=dict(LPIPS=1.,DISTS=cfg['dists'],L1=.1383589248954108,rate=beta*cfg['rate_multiplier']);loss=sum(weights[k]*v for k,v in terms.items())
   coefficient_grad=torch.autograd.grad(loss,tuple(terms.values()),retain_graph=True)
   assert all(torch.equal(g,torch.as_tensor(weights[k],dtype=g.dtype,device=g.device)) for k,g in zip(terms,coefficient_grad))
   actual=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
   expected=[sum(weights[k]*grad[k][j] for k in terms) for j in range(len(params))];actual=[g if g is not None else torch.zeros_like(t) for g,t in zip(actual,params)];err=norm([a-b for a,b in zip(actual,expected)]);assert err<=1e-4*max(1,norm(expected)),(branch,err,norm(expected))
   checks.append(dict(branch=branch,source_P_sha256=SOURCE_SHA,source_P_module_sha256=original,weights=weights,loss=float(loss),unweighted={k:float(v) for k,v in terms.items()},gradient_norm=norm(actual),weighted_linearity_relative_error=err/max(norm(expected),1e-12),loss_boundary_coefficients_verified=True))
 assert torch.backends.cudnn.allow_tf32==inherited_flags['cudnn_allow_tf32'] and torch.backends.cuda.matmul.allow_tf32==inherited_flags['matmul_allow_tf32']
 save('audits/recovery/gradient_linearity_check_precision.json',dict(status='PASS',inherited_flags=inherited_flags,audit_backward_only='TF32 disabled for VJP additivity check; same original forward graph',flags_restored=True,training_and_evaluation_precision_unchanged=True,independent_decode_tolerance_unchanged=True,original_gradient_tolerance_unchanged=True,diagnostic='audits/recovery/gradient_linearity_diagnostic.json',optimizer_updates=0))
 assert state_hash(p.state_dict())==original and fixed==[state_hash(m.state_dict()) for m in (im,pm,*ql)]
 assert all(not t.requires_grad for m in (im,pm,*ql) for t in m.parameters()) and all(not m.training for m in (im,pm,*ql))
 restore(random_state)
 import itertools
 cosine={f'{a}__{b}':float(sum((u.double()*v.double()).sum() for u,v in zip(grad[a],grad[b])))/(norm(grad[a])*norm(grad[b])) for a,b in itertools.combinations(grad,2)}
 save('audits/pretraining_checks.json',dict(status='PASS',source_step=1000,adaptation_step=0,optimizer_updates=0,formal_plan_updates_consumed=0,replays=replays,branches=checks,unweighted_gradients={k:dict(norm=norm(g),finite=True,nonzero_parameters=sum(bool(t.count_nonzero()) for t in g)) for k,g in grad.items()},component_cosines=cosine,weighted_cosines='identical to unweighted under all fixed positive branch coefficients',categorical_hard_forward_exact_and_STE_identity=True,DISTS_require_grad=True,GLC_and_quality_frozen_eval=True,parameters_unchanged=True,RNG_restored=True,common_rate_backend=backend,peak_MiB=torch.cuda.max_memory_reserved()/2**20))
 fix=load(R/'audits/reference_alignment_fix.json');fix.update(status='PASS',training_checks_pending=False,full_paired_audits_completed=120,full_paired_audits_expected=120,pretraining_checks_sha256=sha(R/'audits/pretraining_checks.json'),identity_STE_backward_verified=True);save('audits/reference_alignment_fix.json',fix)
 print('PRETRAINING CHECKS PASS',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);run(a.parse_args().gpu)
