"""Four matched 300-update continuations. Run one process per physical GPU."""
import argparse, csv, hashlib, json, os, random, sys, time
from pathlib import Path

P = Path(__file__).resolve().parent
REPO = P.parents[1]
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
    import torch
    from gvc_hooks import load_models as base_load
    from runtime import upgrade
    im, pm = base_load(torch.device('cuda:0'), force_zero_thres=.12)
    init = torch.load(INIT, map_location='cpu', weights_only=True)
    assert not init['wrapper']
    pm.recon_generation_net.mlp.load_state_dict(init['bridge'], strict=True)
    pm.recon_generation_net.decoder.load_state_dict(init['generator'], strict=True)
    quant, cond = branch in 'BD', branch in 'CD'
    pm = upgrade(pm, quant, cond)
    im.requires_grad_(False)
    pm.requires_grad_(False)
    pm.recon_generation_net.decoder.float().requires_grad_(True)
    pm.recon_generation_net.mlp.float().requires_grad_(False)
    if quant:
        pm.quant_adjust.float().requires_grad_(True)
    if cond:
        pm.visual_condition.float().requires_grad_(True)
    return im, pm


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
    quant = pm.quant_step(common)
    pm.q_actual = quant
    scaled = y * torch.reciprocal(quant)
    b, c, h, w = scaled.shape
    mask0, mask1 = pm.get_mask_2x(b, c, h, w, scaled.dtype, scaled.device)

    def pass_one(value, scale, mean, mask):
        s, mu = scale*mask, mean*mask
        residual = (value - mu)*mask
        symbol = torch.round(residual).clamp(-128, 127)
        active = mask.bool() & (s > .12)
        symbol = symbol*active
        ste = residual + (symbol-residual).detach()
        return symbol, ste+mu, s, active, ste

    q0, u0, s0, a0, ste0 = pass_one(scaled, common[:,128:256], common[:,256:384], mask0)
    s1, mu1 = pm.y_spatial_prior(torch.cat((u0, common), 1)).chunk(2, 1)
    q1, u1, s1, a1, ste1 = pass_one(scaled, s1, mu1, mask1)
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
                               'qmap': quant, 'z_bits': z_bits}


def main(branch, gpu, stop):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    setup_path()
    import cv2, numpy as np, torch, torch.nn.functional as F
    from core import codec_input
    from gvc_hooks import INDEX_MAP
    from runtime import BlockQuantAdjust, SpatialVisualCondition
    from importlib.util import spec_from_file_location, module_from_spec
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    random.seed(20261023); np.random.seed(20261023)
    torch.manual_seed(20261023); torch.cuda.manual_seed_all(20261023)
    assert sha(INIT) == '89c1a5baa195c8ffc9695af495d43389e00021b926eae13a0337d3b2eb117fd4'
    im, pm = load_models(branch)
    decoder = pm.recon_generation_net.decoder
    groups = [{'params': list(decoder.parameters()), 'lr': 1e-6, 'name': 'G'}]
    if branch in 'BD':
        # Identical weights in B and D independent of model construction RNG.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(20261023)
            pm.quant_adjust.load_state_dict(BlockQuantAdjust().state_dict())
        groups.append({'params': list(pm.quant_adjust.parameters()), 'lr': 1e-4, 'name': 'H'})
    if branch in 'CD':
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(20261024)
            pm.visual_condition.load_state_dict(SpatialVisualCondition().state_dict())
        groups.append({'params': list(pm.visual_condition.parameters()), 'lr': 1e-4, 'name': 'V'})
    opt = torch.optim.AdamW(groups, betas=(.9,.999), eps=1e-8, weight_decay=1e-4)
    params = [p for g in groups for p in g['params']]
    assert all(p.dtype == torch.float32 for p in params)
    import lpips
    lp = lpips.LPIPS(net='alex', verbose=False).cuda().eval().requires_grad_(False)
    common_path = REPO/'experiments/gvcrt_neural_wrapper_v6_2_objective_calibration/fullqp_common.py'
    spec = spec_from_file_location('fullqp_common', common_path)
    common = module_from_spec(spec); spec.loader.exec_module(common)
    plan = frame_plan()
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    out = P/'server_artifacts'/branch
    out.mkdir(parents=True, exist_ok=True)
    latest = out/'latest.pt'
    step = 0
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        assert state['plan_hash'] == plan_hash and state['branch'] == branch
        decoder.load_state_dict(state['G'])
        if branch in 'BD': pm.quant_adjust.load_state_dict(state['H'])
        if branch in 'CD': pm.visual_condition.load_state_dict(state['V'])
        opt.load_state_dict(state['optimizer'])
        restore_rng(torch, np, state['rng'])
        step = state['step']
        log_path=P/f'training_{branch}.csv'
        logged=list(csv.DictReader(log_path.open())) if log_path.exists() else []
        last_logged=int(logged[-1]['step']) if logged else 0
        assert last_logged<=step
        if last_logged<step:
            receipt=state.get('last_receipt')
            assert last_logged==step-1 and receipt is not None and int(receipt['step'])==step
            with log_path.open('a',newline='') as f:
                writer=csv.DictWriter(f,list(receipt))
                if not logged:writer.writeheader()
                writer.writerow(receipt);f.flush();os.fsync(f.fileno())

    def save(number,receipt=None):
        payload = {'branch':branch, 'step':number, 'cursor':number,
                   'plan_hash':plan_hash, 'G':decoder.state_dict(),
                   'H':pm.quant_adjust.state_dict() if branch in 'BD' else None,
                   'V':pm.visual_condition.state_dict() if branch in 'CD' else None,
                   'optimizer':opt.state_dict(), 'rng':rng_state(torch,np),
                   'last_receipt':receipt}
        atomic_save(torch, latest, payload)
        if number in (0,100,300): atomic_save(torch,out/f'step_{number}.pt',payload)
    if step == 0 and not latest.exists(): save(0)
    log = P/f'training_{branch}.csv'
    fields = ['branch','step','source_sha256','first_frame','crop_x','crop_y','external_qp',
              'loss','LPIPS','L1','estimated_bpp','grad_norm','parameter_delta',
              'optimizer_internal_step','seconds','peak_MiB']
    for index in range(step, min(stop,300)):
        start = time.time()
        row = plan[index]
        frames = load_frames(row, torch.device('cuda:0'))
        before = [p.detach().clone() for p in params]
        opt.zero_grad(set_to_none=True)
        pm.clear_dpb(); pm.set_curr_poc(0)
        with torch.no_grad():
            imode = im.compress(codec_input(frames[0]), row['external_qp'])
            pm.add_ref_frame(None, imode['x_hat'])
        losses = []
        for pos in range(1,16):
            qp = pm.shift_qp(row['external_qp'], INDEX_MAP[pos % len(INDEX_MAP)])
            with torch.enable_grad() if pos >= 12 else torch.no_grad():
                rec, feature, rate, _ = causal_forward(pm, frames[pos], qp)
                if pos >= 12:
                    lpi = lp(rec, frames[pos], normalize=True).mean()
                    l1 = F.l1_loss(rec, frames[pos])
                    bpp = rate/(256*256)
                    total = lpi + .1383589248954108*l1 + common.beta_q('schedule_s1p0',row['external_qp'])*bpp
                    (total/4).backward()
                    losses.append((float(total.detach()),float(lpi.detach()),
                                   float(l1.detach()),float(bpp.detach())))
            with torch.no_grad():
                pm.add_ref_frame(feature.detach(),(rec.detach()*2-1).half())
            del rec, feature, rate
        assert len(losses) == 4
        grad_norm = torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        assert float(grad_norm) > 0
        opt.step()
        delta = sum(float((p.detach()-old).double().square().sum()) for p,old in zip(params,before))**.5
        assert delta > 0
        internal = {int(s['step']) for s in opt.state.values()}
        assert internal == {index+1}, internal
        values = {'branch':branch,'step':index+1,'source_sha256':row['source_sha256'],
                  'first_frame':row['indices'][0],'crop_x':row['x'],'crop_y':row['y'],
                  'external_qp':row['external_qp'],
                  'loss':sum(x[0] for x in losses)/4,'LPIPS':sum(x[1] for x in losses)/4,
                  'L1':sum(x[2] for x in losses)/4,
                  'estimated_bpp':sum(x[3] for x in losses)/4,
                  'grad_norm':float(grad_norm),'parameter_delta':delta,
                  'optimizer_internal_step':index+1,'seconds':time.time()-start,
                  'peak_MiB':torch.cuda.max_memory_reserved()/2**20}
        save(index+1,values)
        with log.open('a', newline='') as f:
            writer = csv.DictWriter(f,fields)
            if f.tell() == 0: writer.writeheader()
            writer.writerow(values)
            f.flush(); os.fsync(f.fileno())
        if (index+1)%10==0 or index == step: print(branch,index+1,'/',stop,values['seconds'],flush=True)
    print('DONE',branch,min(stop,300),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--branch',choices=list('ABCD'),required=True)
    ap.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True)
    ap.add_argument('--stop',type=int,default=300)
    a = ap.parse_args()
    main(a.branch,a.gpu,a.stop)
