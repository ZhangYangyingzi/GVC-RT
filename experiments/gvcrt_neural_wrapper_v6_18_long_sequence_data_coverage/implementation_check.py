"""No-update loss/gradient equivalence, deep DISTS and 16-frame budget probe."""
import argparse,traceback
from io18 import *
from window_training import *
from gradient_tools import stats,dists_details
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args();os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261010);frozen();command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    v4,im,pm,models,quality=load_training(torch.device('cuda:0'));before={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);qh=[v4.module_hash(m) for m in quality];params=[p for m in models.values() for p in m.parameters()]
    record=replay_rows('A')[0];frames=replay(record,'A',torch.device('cuda:0'));legacy=objective(v4,im,pm,models,quality,lambda_dists=.5);new=make_runner(v4,im,pm,models,quality);q=record['external_qp']
    for p in params:p.grad=None
    oldloss,oldactual=legacy(frames[:4],q);oldgrad=[p.grad.detach().cpu().clone() if p.grad is not None else None for p in params];four=[]
    for branch in ('A','B'):
        for p in params:p.grad=None
        loss,actual,trace=new(frames[:4],q,branch,True);assert actual==oldactual
        values={k:dict(historical=oldloss[k],new=loss[k],abs_difference=abs(oldloss[k]-loss[k])) for k in oldloss};assert all(abs(oldloss[k]-loss[k])<=2e-6+1e-5*abs(oldloss[k]) for k in oldloss)
        maximum=0.;relative=0.
        for p,g in zip(params,oldgrad):
            assert (p.grad is None)==(g is None)
            if g is not None:
                h=p.grad.detach().cpu();assert torch.allclose(h,g,atol=2e-6,rtol=2e-4),('4frame gradient mismatch',branch);maximum=max(maximum,float((h-g).abs().max()));relative=max(relative,float((h-g).norm()/(g.norm()+1e-12)))
        four.append(dict(branch=branch,status='PASS',losses=values,max_gradient_absolute_difference=maximum,max_tensor_gradient_relative_L2=relative,actual_qps=actual))
    del oldgrad
    x=torch.rand(1,3,256,256,device='cuda:0',requires_grad=True);y=torch.rand_like(x);score,layers,terms=dists_details(quality,x,y,True);assert all(r['output_gradient']['finite'] and r['output_gradient']['effective_parameters']>0 for r in layers)
    del score,terms,x,y
    windows=[]
    for b in ('A','B','C'):
        for p in params:p.grad=None
        r=replay_rows(b)[0];fs=replay(r,b,torch.device('cuda:0'));loss,actual,trace=new(fs,r['external_qp'],b,True)
        assert actual==r['actual_qps'] and len(trace)==15 and all(t['normalization_denominator']==15 and t['reference_detached'] for t in trace)
        assert [t['window_position']-1 for t in trace if t['reference_reset_before']]==([0,3,6,9,12] if b=='A' else [0])
        gs={k:stats([p.grad for p in m.parameters()],list(m.parameters())) for k,m in models.items()};assert all(s['finite'] and s['effective_parameters'] for s in gs.values())
        windows.append(dict(branch=b,status='PASS',sample=r,loss=loss,actual_qps=actual,trace=trace,gradients=gs,optimizer_updates=0,supervised_P_frames=15));print('WINDOW CHECK PASS',b,flush=True)
    assert before=={k:v4.module_hash(m) for k,m in models.items()} and qh==[v4.module_hash(m) for m in quality] and core==v4.compression_hash(im,pm)==evalcfg()['compression_hash']
    dump(ROOT/'audits/implementation_check.json',dict(status='PASS',optimizer_updates=0,initial_module_hashes=before,compression_hash=core,quality_hashes=qh,standard4_equivalence=four,DISTS_layers=layers,windows=windows,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,models_unchanged=True));print('IMPLEMENTATION CHECK PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'audits/implementation_check.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
