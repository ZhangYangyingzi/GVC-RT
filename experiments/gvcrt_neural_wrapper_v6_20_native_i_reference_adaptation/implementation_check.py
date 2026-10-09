"""No optimizer update: exact wrapped16 equivalence and both branch gradients."""
import argparse,traceback,types
from io20 import *
from window_training import *
from gradient_tools import stats
if __name__=='__main__':
    try:
        ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=ap.parse_args();os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
        import torch
        torch.set_num_threads(2);torch.manual_seed(20261010);frozen();v4,im,pm,models,quality=load_training(torch.device('cuda:0'));before={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);ih=v4.module_hash(im);qh=[v4.module_hash(m) for m in quality];params=[p for m in models.values() for p in m.parameters()];r=replay_rows(BRANCHES[0])[0];frames=replay(r,BRANCHES[0],torch.device('cuda:0'))
        ns=dict(globals());exec((V18/'window_training.py').read_text().replace('from io18 import *',''),ns);old=ns['make_runner'](v4,im,pm,models,quality);new=make_runner(v4,im,pm,models,quality)
        for p in params:p.grad=None
        oldloss,qa,oldtrace=old(frames,r['external_qp'],'B',True);grad=[p.grad.detach().cpu().clone() if p.grad is not None else None for p in params];records=[]
        for b in BRANCHES:
            for p in params:p.grad=None
            loss,actual,trace=new(frames,r['external_qp'],b,True);assert actual==r['actual_qps'] and len(trace)==15 and all(x['reference_detached'] and x['normalization_denominator']==15 for x in trace);assert [x['window_position']-1 for x in trace if x['reference_reset_before']]==[0]
            if b=='wrapped_i_control':
                assert all(abs(loss[k]-oldloss[k])<=2e-6+1e-5*abs(oldloss[k]) for k in oldloss)
                for p,g in zip(params,grad):
                    assert (p.grad is None)==(g is None)
                    if g is not None:assert torch.allclose(p.grad.cpu(),g,atol=2e-6,rtol=2e-4)
            gs={k:stats([p.grad for p in m.parameters()],list(m.parameters())) for k,m in models.items()};assert all(g['finite'] and g['effective_parameters']>0 for g in gs.values());records.append(dict(branch=b,loss=loss,actual_qps=actual,trace=trace,gradients=gs,resets=[0],supervised_P_frames=15,optimizer_updates=0))
        assert before=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm) and ih==v4.module_hash(im) and qh==[v4.module_hash(m) for m in quality]
        dump(ROOT/'audits/implementation_check.json',dict(status='PASS',windows=records,initial_module_hashes=before,compression_hash=core,I_hash=ih,wrapped16_equivalent_to_v618_B=True,optimizer_updates=0,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20));print('IMPLEMENTATION PASS',flush=True)
    except Exception:dump(ROOT/'audits/implementation_check.json',dict(status='FAIL',error=traceback.format_exc()));raise
