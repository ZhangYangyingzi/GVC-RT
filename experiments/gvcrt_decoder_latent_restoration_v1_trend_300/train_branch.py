"""Three matched 300-update continuations over the shared native decoder cache."""
import argparse
import math
import time

from common_run import (HERE, dump, load, load_base, restore_rng, rng, save_pt,
                        setup, sha, state_hash)
from restoration import Adapter, condition


def run(branch, gpu, stop=300):
    assert branch in ('G', 'R', 'RQ') and stop in (2, 300)
    assert load(HERE / 'preflight.json')['status'] == 'PASS'
    torch, np = setup(gpu)
    from torch.nn import functional as F
    import lpips
    from cache_training import load_frames
    _, pm = load_base(torch)
    generator = pm.recon_generation_net.decoder
    adapter = None
    if branch == 'G':
        generator.train().requires_grad_(True)
        params = list(generator.parameters())
        learning_rate = 1e-6
    else:
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(20261027); torch.cuda.manual_seed_all(20261027)
            adapter = Adapter().cuda().float()
        params = list(adapter.parameters())
        learning_rate = 1e-4
    assert all(p.dtype == torch.float32 for p in params)
    component_hashes = dict(codec=state_hash({k: v for k, v in pm.state_dict().items()
                                               if not k.startswith('recon_generation_net.')}),
                            bridge=state_hash(pm.recon_generation_net.mlp.state_dict()),
                            generator_A=state_hash(generator.state_dict()))
    initial = state_hash((adapter if adapter is not None else generator).state_dict())
    opt = torch.optim.AdamW(params, lr=learning_rate, betas=(.9, .999), eps=1e-8,
                            weight_decay=1e-4)
    root = HERE / 'local_artifacts/checkpoints' / branch
    root.mkdir(parents=True, exist_ok=True)
    latest = root / 'latest.pt'
    step = 0
    rows = []
    # Loss construction must not change the saved training RNG on resume.
    with torch.random.fork_rng(devices=[0]):
        lp = lpips.LPIPS(net='alex', verbose=False).cuda().eval().requires_grad_(False)

    def checkpoint(n):
        payload = dict(branch=branch, step=n, cursor=n, effective_updates=n,
                       model=(adapter if adapter is not None else generator).state_dict(),
                       optimizer=opt.state_dict(), rng=rng(torch, np),
                       plan_sha256=sha(HERE / 'train_plan.json'),
                       initial_model_hash=initial, frozen_components=component_hashes, training_rows=rows)
        save_pt(torch, latest, payload)
        if n in (0, 100, 300):
            save_pt(torch, root / f'step{n}.pt', payload)

    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        assert state['branch'] == branch and state['plan_sha256'] == sha(HERE / 'train_plan.json')
        assert state['frozen_components'] == component_hashes
        assert state['cursor'] == state['step'] == state['effective_updates']
        (adapter if adapter is not None else generator).load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], torch, np)
        step = state['step']; initial = state['initial_model_hash']
    else:
        checkpoint(0)
    plan = load(HERE / 'train_plan.json')['rows']
    log = HERE / f'training_{branch}.csv'
    if step and 'training_rows' in state:
        rows = state['training_rows']
        from common_run import table
        table(log, rows)
        assert len(rows) == step
    elif log.exists():
        import csv
        with log.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == step
    torch.cuda.reset_peak_memory_stats()
    for index in range(step, stop):
        receipt = load(HERE / 'local_artifacts/training_cache' / f'{index:04d}.json')
        path = HERE / 'local_artifacts/training_cache' / f'{index:04d}.pt'
        assert sha(path) == receipt['cache_sha256']
        cache = torch.load(path, map_location='cpu', weights_only=False)
        assert cache['plan_sha256'] == sha(HERE / 'train_plan.json') and cache['row'] == plan[index]
        assert cache['weight_manifest_sha256'] == sha(HERE / 'weight_manifest.json')
        assert cache['code_sha256'] == {name: sha(HERE / name) for name in
                                        ('cache_training.py', 'common_run.py', 'restoration.py')}
        began = time.time(); opt.zero_grad(set_to_none=True)
        gt = load_frames(plan[index], torch.device('cuda:0'))
        components = []
        ratios = []
        for j, record in enumerate(cache['records']):
            assert record['frame'] == 12 + j and record['source_frame_index'] == plan[index]['indices'][12 + j]
            u = record['u'].cuda().float()
            q = record['q_recon'].cuda().float()
            if adapter is None:
                delta = torch.zeros_like(u)
            else:
                delta = adapter(record['f'].cuda(), u,
                                condition(record['q_actual'].cuda(), branch))
            output = generator(u + delta, q).float().clamp(-1, 1)
            output = (output + 1) / 2
            target = gt[12 + j]
            lpi = lp(output, target, normalize=True).mean()
            l1 = F.l1_loss(output, target)
            loss = lpi + .1383589248954108 * l1
            (loss / 4).backward()
            with torch.no_grad():
                dnorm = torch.linalg.vector_norm(delta.float())
                unorm = torch.linalg.vector_norm(u.float())
                ratios.append((float(delta.square().mean().sqrt()), float(dnorm / unorm.clamp_min(1e-12))))
                components.append((float(loss), float(lpi), float(l1)))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
        assert grad_norm > 0 and math.isfinite(grad_norm)
        active_parameter_count = (sum(p.numel() for p in params
                                      if p.grad is not None and bool(torch.count_nonzero(p.grad)))
                                  if index + 1 in (1, 2, 100, 300) else None)
        if branch in ('R', 'RQ') and index == 0:
            assert adapter.output.weight.grad is not None and adapter.output.weight.grad.abs().sum() > 0
        opt.step()
        if branch in ('R', 'RQ') and index == 1:
            assert adapter.feature[0].weight.grad is not None
            assert adapter.feature[0].weight.grad.abs().sum() > 0
        row = dict(branch=branch, step=index + 1, source_sha256=plan[index]['source_sha256'],
                   pair_index=plan[index]['pair_index'], external_qp=plan[index]['external_qp'],
                   stream_sha256=cache['stream_sha256'],
                   loss=sum(x[0] for x in components) / 4,
                   LPIPS=sum(x[1] for x in components) / 4,
                   RGB_L1=sum(x[2] for x in components) / 4,
                   gradient_norm_preclip=grad_norm,
                   parameters_with_nonzero_gradient=active_parameter_count,
                   delta_u_RMS=sum(x[0] for x in ratios) / 4,
                   relative_delta_u_norm=sum(x[1] for x in ratios) / 4,
                   seconds=time.time() - began,
                   peak_memory_MiB=torch.cuda.max_memory_reserved() / 2**20)
        rows.append(row)
        checkpoint(index + 1)
        if index + 1 in (2, 100, 300):
            assert state_hash({k: v for k, v in pm.state_dict().items()
                               if not k.startswith('recon_generation_net.')}) == component_hashes['codec']
            assert state_hash(pm.recon_generation_net.mlp.state_dict()) == component_hashes['bridge']
            if adapter is not None:
                assert state_hash(generator.state_dict()) == component_hashes['generator_A']
        from common_run import table
        table(log, rows)
        dump(HERE / f'training_status_{branch}.json',
             dict(status='COMPLETE' if index == 299 else 'RUNNING', updates=index + 1,
                  physical_gpu=gpu, checkpoint=str(latest), initial_model_hash=initial,
                  trainable_parameters=sum(p.numel() for p in params),
                  frozen_components=component_hashes, precision='FP32 G and Adapter; FP16 base codec'))
        if index < 2 or (index + 1) % 10 == 0:
            print('UPDATE', branch, index + 1, round(row['seconds'], 2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', choices=['G', 'R', 'RQ'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--stop', type=int, choices=[2, 300], default=300)
    args = parser.parse_args()
    run(args.branch, args.gpu, args.stop)
