"""Four matched 300-update continuations. Run one process per physical GPU."""
import argparse, csv, hashlib, json, os, random, sys, time
from pathlib import Path

P = Path(__file__).resolve().parent
REPO = Path("/Huang_group/zyyz/Projects/GVC-RT")
ORIGINAL = Path('/Huang_group/zyyz/Projects/GVC-RT')
INIT = ORIGINAL / 'experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100/evaluation/checkpoints/G_basic_100.pt'


def setup_path():
    sys.path[:0] = [str(P), str(REPO),
                    str(REPO/'experiments/gvcrt_neural_wrapper_v2_joint'),
                    str(REPO/'expericent_generation_input/expericent_interface_causal_controls_v9/src'),
                    str(REPO/'experiments/gvcrt_neural_wrapper_v6_1_large_diverse_qp09'),
                    str(REPO/'experiments/gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies')]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_save(torch, path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def load_models(branch):
    from model_io import build_models
    return build_models(branch)


def rng_state(torch, np):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(torch, np, state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])


def frame_plan():
    manifest = json.loads((P/'data_manifest.json').read_text())
    windows = [(v['source'], a, b) for v in manifest['train'] for a, b in v['windows']]
    plan = []
    for i in range(300):
        rng = random.Random(20261023 + i)
        source, first, last = windows[i % len(windows)]
        h, w = int(source['height']), int(source['width'])
        plan.append({'source_sha256': source['sha256'], 'path': source['path'],
                     'indices': list(range(first, last + 1)),
                     'x': rng.randrange(w - 256 + 1), 'y': rng.randrange(h - 256 + 1),
                     'external_qp': (2, 7)[i % 2]})
    return plan


def load_frames(row, device):
    import cv2, torch
    cap = cv2.VideoCapture(row['path'])
    assert cap.isOpened(), row['path']
    out = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, row['indices'][0])
        for index in row['indices']:
            ok, frame = cap.read()
            assert ok, (row['path'], index)
            crop = cv2.cvtColor(frame[row['y']:row['y']+256,
                                            row['x']:row['x']+256], cv2.COLOR_BGR2RGB)
            assert crop.shape == (256, 256, 3)
            out.append(torch.from_numpy(crop.copy()).permute(2, 0, 1)
                       .unsqueeze(0).to(device).float()/255)
    finally:
        cap.release()
    return out


def symbol_pass(value, scale, mean, mask, fix_skip=False):
    import torch
    s, mu = scale*mask, mean*mask
    residual = (value - mu)*mask
    symbol = torch.round(residual).clamp(-128, 127)
    active = mask.bool() & (s > .12)
    symbol = symbol*active
    ste = residual + (symbol-residual).detach()
    if fix_skip:
        ste = ste * active.to(residual.dtype)
    return symbol, ste+mu, s, active, ste

def causal_forward(pm, frame, qp):
    """STE reconstruction and continuous CDF y+z rate in the native prior domain."""
    import torch
    from core import codec_input, unit, prepare_p
    ctx, temporal, qe, qd, qr = prepare_p(pm, qp)
    y = pm.enc(codec_input(frame), ctx, qe)
    z = pm.hyper_enc(pm.pad_for_y(y))
    z_hard = torch.clamp(torch.round(z), -128, 127)
    z_hat = z + (z_hard - z).detach()
    pm.current_qp = qp
    common = pm.res_prior_param_decoder(z_hat, temporal)
    quant, rho, scale0, mean0 = pm.coordinates(common)
    pm.q_actual = quant
    scaled = y * torch.reciprocal(quant)
    b, c, h, w = scaled.shape
    mask0, mask1 = pm.get_mask_2x(b, c, h, w, scaled.dtype, scaled.device)


    q0, u0, s0, a0, ste0 = symbol_pass(scaled, scale0, mean0, mask0, pm.fix_skip)
    s1, mu1 = pm.second_prior(u0, common, pm.y_spatial_prior, rho)
    q1, u1, s1, a1, ste1 = symbol_pass(scaled, s1, mu1, mask1, pm.fix_skip)
    y_hat = (u0+u1)*quant
    x_hat, feature = pm.get_recon_and_feature(y_hat, ctx, qd, qr)
    rec = unit(x_hat, *frame.shape[-2:])
    upper = pm.bit_estimator_z.get_cdf(z_hat.float()+.5, qp)
    lower = pm.bit_estimator_z.get_cdf(z_hat.float()-.5, qp)
    z_bits = -torch.log2((upper-lower).clamp_min(1e-9)).sum()

    def y_bits(ste, sigma, active):
        sym = ste.float()[active]
        scale = sigma.float()[active].clamp(.11, 16)
        normal = torch.distributions.Normal(torch.zeros_like(scale), scale)
        probability = (normal.cdf(sym+.5)-normal.cdf(sym-.5)).clamp_min(1e-9)
        return -torch.log2(probability).sum()

    rate = z_bits + y_bits(ste0, s0, a0) + y_bits(ste1, s1, a1)
    return rec, feature, rate, {'y_hat': y_hat, 'q0': q0, 'q1': q1,
                               'qmap': quant, 'rho': rho, 'common':common, 'z_bits': z_bits, 'y_bits': rate-z_bits, **diagnostics(pm, common, quant, rho, q0, q1, a0, a1, mask0, mask1)}



def diagnostics(pm, common, quant, rho, q0, q1, a0, a1, m0, m1):
    import torch
    out = {}
    with torch.no_grad():
        mult = pm.quant_adjust(common, pm.current_qp) if pm.use_quant else torch.ones_like(quant)
        for name, tensor in [('H_multiplier', mult), ('actual_rho', rho), ('q_base',common[:,:128].clamp_min(.5)), ('q_new',quant), ('raw_quant_step',common[:,:128])]:
            flat = tensor.detach().float().flatten()
            out[name+'_mean'] = float(flat.mean())
            out[name+'_std'] = float(flat.std(unbiased=False))
            for label, level in [('p01',.01),('p05',.05),('p50',.5),('p95',.95),('p99',.99)]:
                out[name+'_'+label] = float(torch.quantile(flat,level))
        base = common[:,:128].clamp_min(.5)
        out['q_base_lower_clamp_fraction']=float((common[:,:128]<.5).float().mean())
        out['H_multiplier_saturation_fraction']=float(((mult<=.5001)|(mult>=1.9999)).float().mean())
        out['q_lower_clamp_fraction'] = float(((base.float()*mult)<.5).float().mean())
        for i,(sym,active,mask) in enumerate([(q0,a0,m0),(q1,a1,m1)]):
            out[f'pass{i}_skip_fraction'] = float((mask.bool() & ~active).sum()/mask.sum())
            out[f'pass{i}_saturation_fraction'] = float((((sym == -128)|(sym == 127)) & active).sum()/mask.sum())
    return out



def parameter_groups(pm,branch):
    from runtime import PHI_SCOPE
    groups=[dict(params=list(pm.recon_generation_net.decoder.parameters()),lr=1e-6,name='G'),
            dict(params=[p for name in PHI_SCOPE for p in getattr(pm,name).parameters()],lr=1e-6,name='Phi')]
    if branch=='Q1':groups.append(dict(params=list(pm.quant_adjust.parameters()),lr=1e-4,name='H'))
    return groups

def norm(values):
    return sum(float(v.detach().double().square().sum()) for v in values if v is not None)**.5

def frozen_hash(im,pm):
    from util import state_hash
    return state_hash({**{'I.'+n:v for n,v in im.named_parameters()},**{'P.'+n:v for n,v in pm.named_parameters() if not v.requires_grad}})

def train_branch(branch,gpu,stop=300,smoke=False):
    from util import P as ROOT,gpu_env,load,dump,write,append,signature,model_hash
    from model_io import checkpoint_state,restore_model,phi_hash
    import fcntl
    assert branch in ('N1','Q1') and stop in (2,300)
    gpu_env(gpu);setup_path()
    import torch,numpy as np,cv2,torch.nn.functional as F
    from core import codec_input
    from gvc_hooks import INDEX_MAP
    assert INDEX_MAP==[0,1,0,2,0,2,0,2]
    from runtime import PHI_SCOPE
    random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if not smoke:
        pf=load(P/'preflight.json');assert pf['status']=='PASS' and pf['signature']==signature()
    out=P/'server_artifacts'/branch;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    im,pm=load_models(branch);groups=parameter_groups(pm,branch);params=[p for g in groups for p in g['params']]
    assert all(p.dtype==torch.float32 for p in params)
    assert {id(p) for p in params}=={id(p) for p in pm.parameters() if p.requires_grad}
    assert not hasattr(pm,'visual_condition')
    frozen=frozen_hash(im,pm);initial=dict(G=model_hash(pm.recon_generation_net.decoder),Phi=phi_hash(pm),H=model_hash(pm.quant_adjust) if pm.use_quant else None)
    opt=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    common_path=REPO/'experiments/gvcrt_neural_wrapper_v6_2_objective_calibration/fullqp_common.py'
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('quant_v3_qp_common',common_path);common=module_from_spec(spec);spec.loader.exec_module(common)
    plan=load(P/'train_plan.json')['rows'];assert plan==frame_plan()
    latest=out/'latest.pt';step=0
    if latest.exists():
        st=torch.load(latest,map_location='cpu',weights_only=False);assert st['signature']==signature() and st['branch']==branch and st['plan_sha256']==sha(P/'train_plan.json')
        restore_model(pm,st);opt.load_state_dict(st['optimizer']);restore_rng(torch,np,st['rng']);step=st['step'];assert st['cursor']==step
        if step and not (out/'receipts'/f'{step:04d}.json').exists():dump(out/'receipts'/f'{step:04d}.json',st['last_receipt'])
        if step in (100,300) and not (out/f'step_{step}.pt').exists():atomic_save(torch,out/f'step_{step}.pt',st)
        initial=st['initial_hashes'];assert frozen==st['frozen_hash'];del st
        events=[json.loads(x) for x in (out/'optimizer_events.jsonl').read_text().splitlines()] if (out/'optimizer_events.jsonl').exists() else []
        for event in events:
            if event['event']=='APPLIED' and event['step']>step:append(out/'optimizer_events.jsonl',dict(event='UNCOMMITTED_DISCARDED',step=event['step'],old_PID=event['PID'],resume_cursor=step))
    def save(n,receipt=None):
        assert frozen_hash(im,pm)==frozen
        st=dict(branch=branch,step=n,cursor=n,signature=signature(),plan_sha256=sha(P/'train_plan.json'),**checkpoint_state(pm),optimizer=opt.state_dict(),rng=rng_state(torch,np),initial_hashes=initial,frozen_hash=frozen,last_receipt=receipt,CDF_must_be_regenerated_after_Phi_load=True)
        atomic_save(torch,latest,st)
        if n in (0,100,300):atomic_save(torch,out/f'step_{n}.pt',st)
    if not latest.exists():save(0)
    torch.cuda.reset_peak_memory_stats()
    for index in range(step,stop):
        started=time.time();row=plan[index];frames=load_frames(row,torch.device('cuda:0'));before={id(p):p.detach().clone() for p in params};opt.zero_grad(set_to_none=True)
        pm.clear_dpb();pm.set_curr_poc(0)
        with torch.no_grad():pm.add_ref_frame(None,im.compress(codec_input(frames[0]),row['external_qp'])['x_hat'])
        losses=[];extra=[];actual=[row['external_qp']]
        for pos in range(1,16):
            q=pm.shift_qp(row['external_qp'],INDEX_MAP[pos%8]);actual.append(q)
            with torch.enable_grad() if pos>=12 else torch.no_grad():
                rec,feature,rate,trace=causal_forward(pm,frames[pos],q)
                if pos>=12:
                    lpi=lp(rec,frames[pos],normalize=True).mean();l1=F.l1_loss(rec,frames[pos]);bpp=rate/65536
                    loss=lpi+.1383589248954108*l1+common.beta_q('schedule_s1p0',row['external_qp'])*bpp
                    (loss/4).backward()
                    losses.append(tuple(float(x.detach()) for x in (loss,lpi,l1,bpp)))
                    extra.append({k:float(v.detach()) if hasattr(v,'detach') else float(v) for k,v in trace.items() if k not in ('y_hat','q0','q1','qmap','rho','common')})
            with torch.no_grad():pm.add_ref_frame(feature.detach(),(rec.detach()*2-1).half())
            assert not pm.dpb[0].feature.requires_grad
        assert len(losses)==4
        assert all(p.grad is None for p in im.parameters())
        assert all(p.grad is None for p in pm.parameters() if not p.requires_grad)
        assert all(p.grad is None or (p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all()) for p in params)
        grad={g['name']+'_grad_norm':norm(p.grad for p in g['params']) for g in groups}
        grad.update(Phi_y_grad_norm=norm(p.grad for name in PHI_SCOPE[:-1] for p in getattr(pm,name).parameters()),Phi_z_grad_norm=norm(p.grad for p in pm.bit_estimator_z.parameters()))
        assert all(v>0 for v in grad.values()),grad
        pre=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));post=norm(p.grad for p in params)
        append(out/'optimizer_events.jsonl',dict(event='STARTED',step=index+1,PID=os.getpid(),time=time.time()))
        opt.step();append(out/'optimizer_events.jsonl',dict(event='APPLIED',step=index+1,PID=os.getpid(),time=time.time()))
        assert {int(s['step']) for s in opt.state.values()}=={index+1}
        delta={g['name']+'_parameter_delta':norm(p.detach()-before[id(p)] for p in g['params']) for g in groups}
        delta.update(Phi_y_parameter_delta=norm(p.detach()-before[id(p)] for name in PHI_SCOPE[:-1] for p in getattr(pm,name).parameters()),Phi_z_parameter_delta=norm(p.detach()-before[id(p)] for p in pm.bit_estimator_z.parameters()))
        assert all(v>0 for v in delta.values()),delta
        values=dict(branch=branch,step=index+1,source_sha256=row['source_sha256'],first_frame=row['indices'][0],crop_x=row['x'],crop_y=row['y'],external_qp=row['external_qp'],actual_frame_QPs=actual,loss=sum(x[0] for x in losses)/4,LPIPS=sum(x[1] for x in losses)/4,L1=sum(x[2] for x in losses)/4,estimated_bpp=sum(x[3] for x in losses)/4,preclip_norm=pre,postclip_norm=post,clip_coefficient=min(1.,1/(pre+1e-6)),optimizer_internal_step=index+1,seconds=time.time()-started,peak_MiB=torch.cuda.max_memory_reserved()/2**20,frozen_gradients_absent=True,**grad,**delta)
        values.update({('estimated_'+k if k in ('y_bits','z_bits') else k):sum(e[k] for e in extra)/4 for k in extra[0]})
        save(index+1,values);dump(out/'receipts'/f'{index+1:04d}.json',values)
        append(out/'optimizer_events.jsonl',dict(event='COMMITTED',step=index+1,PID=os.getpid(),time=time.time()))
        dump(P/f'training_status_{branch}.json',dict(status='COMPLETE' if index==299 else 'SMOKE' if stop==2 else 'RUNNING',committed_updates=index+1,physical_GPU=gpu,signature=signature()))
        if (index+1)%10==0 or index==step:print('UPDATE',branch,index+1,round(values['seconds'],2),flush=True)
    assert frozen_hash(im,pm)==frozen
    receipts=[load(out/'receipts'/f'{i:04d}.json') for i in range(1,stop+1)];write(P/f'training_{branch}.csv',receipts)
    dump(P/f'parameter_audit_{branch}.json',dict(status='PASS',branch=branch,committed_updates=stop,optimizer_internal_step=stop,initial=initial,final=dict(G=model_hash(pm.recon_generation_net.decoder),Phi=phi_hash(pm),H=model_hash(pm.quant_adjust) if pm.use_quant else None),frozen_hash=frozen,frozen_parameters_unchanged=True,signature=signature()))
    print('TRAIN_DONE',branch,stop,flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['N1','Q1'],required=True);ap.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True);ap.add_argument('--stop',type=int,choices=[2,300],default=300);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();train_branch(a.branch,a.gpu,a.stop,a.smoke)
