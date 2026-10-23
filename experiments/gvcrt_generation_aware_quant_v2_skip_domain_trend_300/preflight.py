"""Fixed tolerances and adversarial coordinate/gradient/coding checks; gates training."""
import os, sys, json, hashlib, importlib.util, time
from pathlib import Path
P=Path(__file__).resolve().parent
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='7'
sys.path.insert(0,str(P))
import train
train.setup_path()
import torch
from core import codec_input, module_hash
from runtime import BlockQuantAdjust
V1=P.parent/'gvcrt_generation_aware_quant_visual_condition_v1_trend_300'
R={'status':'RUNNING','physical_gpu':7,'tolerances':{'identity':1e-5,'symbols':0,'q':0,'hard_latent':.01,'hard_feature':.01,'decode_reference':0,'cpu_gradient':1e-9},'checks':{},'code_sha256':{f.name:train.sha(f) for f in P.glob('*.py')},'config_sha256':train.sha(P/'effective_config.json'),'initialization_sha256':train.sha(train.INIT)}

def near(name,a,b,tol):
    e=float((a.detach().double()-b.detach().double()).abs().max())
    R['checks'][name]={'max_abs_error':e,'tolerance':tol}
    assert e<=tol,(name,e,tol)

def seed(im,pm,frame):
    pm.clear_dpb();pm.set_curr_poc(0)
    with torch.no_grad():pm.add_ref_frame(None,im.compress(codec_input(frame),2)['x_hat'])

def cpu_check():
    results={}
    for fix in (False,True):
        q=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
        y=torch.tensor(2.,dtype=torch.float64);mu=torch.tensor(.5,dtype=torch.float64)
        residual=y/q-mu;hard=residual.round().clamp(-128,127)*False
        sym=residual+(hard-residual).detach()
        if fix:sym=sym*False
        hat=(sym+mu)*q
        grad=float(torch.autograd.grad(hat,q)[0])
        eps=1e-5;fd=(.5*(1+eps)-.5*(1-eps))/(2*eps)
        assert float(hard)==0 and float(hat)==.5
        assert abs(grad-(.5 if fix else -1.5))<1e-9
        results['fixed' if fix else 'original']={'symbol':float(hard),'reconstruction':float(hat),'dy_hat_dq_autograd':grad,'local_finite_difference':fd,'epsilon':eps}
    R['checks']['cpu_skip']=results
    # Jacobian residual->symbol, and retained mean/dequantization paths, on both active states.
    for fix in (False,True):
        for active in (False,True):
            r=torch.tensor(2.2,requires_grad=True);mu=torch.tensor(.5,requires_grad=True);q=torch.tensor(1.,requires_grad=True)
            hard=r.round().clamp(-128,127)*active
            ste=r+(hard-r).detach()
            if fix:ste=ste*active
            out=(ste+mu)*q
            grads=torch.autograd.grad(out,(r,mu,q))
            assert float(grads[0])==(int(active) if fix else 1)
            assert float(grads[1])==1 and float(grads[2])==float(hard+.5)
    R['checks']['branch_jacobians']='PASS'


def synthetic(pm, branch):
    class Map(torch.nn.Module):
        def forward(self, common, qp):
            a=common.new_full((1,1,4,4),.6).float()
            a[:,:,:,2:]=1.5
            return a
    class Spatial(torch.nn.Module):
        def forward(self, cat):
            out=cat.new_full((1,256,4,4),.25)
            out[:,:128,:,:2]=.05;out[:,:128,:,2:]=1.
            return out
    saved=pm.quant_adjust;pm.quant_adjust=Map();pm.current_qp=2
    common=torch.ones((1,384,4,4),device='cuda',dtype=torch.float16)
    common[:,:128,:,:2]=.5
    common[:,128:256,:,:2]=.05
    common[:,256:]=.25
    q,rho,scale,mean=pm.coordinates(common)
    multiplier=pm.quant_adjust(common,2)
    assert bool(((common[:,:128].float()*multiplier)<.5).any())
    assert bool((rho.float()!=multiplier).any())
    near(branch+'_synthetic_actual_rho',rho,q/common[:,:128].clamp_min(.5),0)
    y=torch.full((1,128,4,4),.25,device='cuda',dtype=torch.float16)
    y[:,0]=-300;y[:,1]=300
    y=y*q
    spatial=Spatial();u=y*torch.reciprocal(q)
    masks=pm.get_mask_2x(*u.shape,u.dtype,u.device)
    sym0,u0,s0,a0,_=train.symbol_pass(u,scale,mean,masks[0],pm.fix_skip)
    sc,mu=pm.second_prior(u0,common,spatial,rho)
    sym1,u1,s1,a1,_=train.symbol_pass(u,sc,mu,masks[1],pm.fix_skip)
    result=pm.compress_prior_2x(y,common,spatial)
    for k,t in [('q0',sym0),('q1',sym1)]:near(branch+'_synthetic_'+k,t,pm.coding_trace[k],0)
    near(branch+'_synthetic_yhat',(u0+u1)*q,result[-1],.01)
    for i,(sym,active,mask) in enumerate([(sym0,a0,masks[0]),(sym1,a1,masks[1])]):
        assert bool((mask.bool() & ~active).any()) and bool(active.any())
        assert bool(((sym==-128)|(sym==127)).any())
        R['checks'][branch+f'_synthetic_coverage_pass{i}']={'skip':int((mask.bool() & ~active).sum()),'active':int(active.sum()),'saturated':int(((sym==-128)|(sym==127)).sum())}
    # Persist exact synthetic two-pass stream and consume both decoded passes.
    pm.entropy_coder.reset()
    pm.gaussian_encoder.encode_y(result[0],result[2]);pm.gaussian_encoder.encode_y(result[1],result[3]);pm.entropy_coder.flush()
    stream=pm.entropy_coder.get_encoded_stream()
    path=P/'server_artifacts/preflight'/f'{branch}_synthetic.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(stream)
    pm.entropy_coder.set_stream(path.read_bytes())
    info=pm.decompress_prior_2x_part1(common)
    decoded=pm.decompress_prior_2x_part2(common,spatial,info)
    near(branch+'_synthetic_independent_path',result[-1],decoded,0)
    pm.quant_adjust=saved

def main():
    torch.set_num_threads(2);cpu_check()
    assert R['initialization_sha256']=='89c1a5baa195c8ffc9695af495d43389e00021b926eae13a0337d3b2eb117fd4'
    plan=json.loads((P/'train_plan.json').read_text())
    assert plan['rows']==train.frame_plan() and plan['seed']==20261023
    canonical=hashlib.sha256(json.dumps(plan['rows'],sort_keys=True).encode()).hexdigest()
    assert canonical==plan['canonical_plan_sha256']
    for f in ('data_manifest.json','train_plan.json','evaluation_sources.json'):
        assert train.sha(P/f)==train.sha(V1/f)
    sources={r['path']:r['source_sha256'] for r in plan['rows']}
    for path,digest in sources.items():assert train.sha(path)==digest
    R['checks']['protocol']={'sources':len(sources),'updates':len(plan['rows']),'plan_sha256':canonical,'manifest_sha256':train.sha(P/'data_manifest.json'),'source_hashes_verified':True}
    spec=importlib.util.spec_from_file_location('v1_train',V1/'train.py');v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
    frames=train.load_frames(plan['rows'][0],torch.device('cuda:0'))[:3]
    old=torch.load(V1/'server_artifacts/B/step_0.pt',map_location='cpu',weights_only=False)
    init_audit={}
    identity=None
    for branch in ('H00','H10','H01','H11'):
        im,pm=train.load_models(branch)
        assert Path(sys.modules['runtime'].__file__).resolve()==P/'runtime.py'
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(20261023);pm.quant_adjust.load_state_dict(BlockQuantAdjust().state_dict())
        for key,val in pm.quant_adjust.state_dict().items():assert torch.equal(val.cpu(),old['H'][key])
        for key,val in pm.recon_generation_net.decoder.state_dict().items():assert torch.equal(val.cpu(),old['G'][key])
        init_audit[branch]={'G':module_hash(pm.recon_generation_net.decoder),'H':module_hash(pm.quant_adjust),'v1_B_step0_sha256':train.sha(V1/'server_artifacts/B/step_0.pt')}
        with torch.no_grad():
            synthetic(pm,branch)
            seed(im,pm,frames[0]);base=v1.causal_forward(pm,frames[1],2)
            actual=train.causal_forward(pm,frames[1],2)
            for label,a,b in [('rec',base[0],actual[0]),('feature',base[1],actual[1]),('q0',base[3]['q0'],actual[3]['q0']),('q1',base[3]['q1'],actual[3]['q1'])]:near(branch+'_identity_'+label,a,b,1e-5)
            if identity is None:identity=actual[0].clone()
            else:near(branch+'_identity_cross',identity,actual[0],1e-5)
            # Nonidentity H; lower bound is exercised here and independently by synthetic test.
            pm.quant_adjust.net[-1].bias.fill_(-.7)
            seed(im,pm,frames[0]);coded=[]
            for pos in (1,2):
                rec,feature,rate,trace=train.causal_forward(pm,frames[pos],2)
                capture={};hook=pm.dec.register_forward_hook(lambda m,a,out:capture.update(feature=out.clone()))
                packet=pm.compress(codec_input(frames[pos]),2)['bit_stream'];hook.remove()
                for key,tol in [('q0',0),('q1',0),('qmap',0),('y_hat',.01)]:near(branch+f'_p{pos}_hard_'+key,trace[key],pm.coding_trace[key],tol)
                near(branch+f'_p{pos}_hard_feature',feature,capture['feature'],.01)
                path=P/'server_artifacts/preflight'/f'{branch}_P{pos}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(packet)
                coded.append((path,pm.coding_trace['y_hat'].clone(),pm.q_actual.clone(),pm.dpb[0].feature.clone()))
            # H10 must retain original v1 hard forward under nonidentity weights.
            if branch=='H10':
                seed(im,pm,frames[0]);a=v1.causal_forward(pm,frames[1],2);b=train.causal_forward(pm,frames[1],2)
                for key in ('q0','q1','qmap','y_hat'):near('H10_nonidentity_v1_'+key,a[3][key],b[3][key],0)
            del im,pm
            torch.cuda.empty_cache()
            im,pm=train.load_models(branch);pm.quant_adjust.net[-1].bias.fill_(-.7)
            seed(im,pm,frames[0])
            for pos,(path,y,q,ref) in enumerate(coded,1):
                pm.decompress(path.read_bytes(),{'height':256,'width':256,'ec_part':1},2)
                near(branch+f'_p{pos}_decode_y',y,pm.decoding_trace['y_hat'],0)
                near(branch+f'_p{pos}_decode_q',q,pm.q_actual,0)
                near(branch+f'_p{pos}_decode_reference',ref,pm.dpb[0].feature,0)
        del im,pm
        torch.cuda.empty_cache()
    R['checks']['import_path']=str(Path(sys.modules['runtime'].__file__).resolve())
    (P/'initialization_audit.json').write_text(json.dumps(init_audit,indent=2)+'\n')
    R['status']='PASS';R['time_unix']=time.time()

if __name__=='__main__':
    try:main()
    except BaseException as e:
        R['status']='FAIL';R['error']=repr(e);raise
    finally:(P/'preflight.json').write_text(json.dumps(R,indent=2)+'\n')
