"""No optimizer updates. Required identity and frozen-generator gradient gates."""
import argparse

from common_run import HERE, dump, load, load_base, setup, sha, state_hash
from restoration import Adapter, condition


def run(gpu):
    torch, _ = setup(gpu)
    path = HERE / 'local_artifacts/training_cache/0000.pt'
    receipt = load(path.with_suffix('.json'))
    assert sha(path) == receipt['cache_sha256']
    cache = torch.load(path, map_location='cpu', weights_only=False)
    assert cache['plan_sha256'] == sha(HERE / 'train_plan.json')
    im, pm = load_base(torch)
    g = pm.recon_generation_net.decoder
    ghash = state_hash(g.state_dict())
    dump(HERE / 'weight_manifest_effective.json',
         dict(input_files=load(HERE / 'weight_manifest.json'),
              loaded_component_state_hashes=dict(
                  I=state_hash(im.state_dict()),
                  P_codec=state_hash({k: v for k, v in pm.state_dict().items()
                                      if not k.startswith('recon_generation_net.')}),
                  bridge=state_hash(pm.recon_generation_net.mlp.state_dict()),
                  G_A=ghash)))
    r = cache['records'][0]
    assert r['frame'] == 12 and r['source_frame_index'] == cache['row']['indices'][12]
    assert r['actual_qp'] in (cache['row']['external_qp'], cache['row']['external_qp'] + 1,
                              cache['row']['external_qp'] + 2)
    f, u, q, qr = (r[k].cuda() for k in ('f', 'u', 'q_actual', 'q_recon'))
    assert q.shape[1] == 128 and bool((q >= .5).all())
    assert q.shape[-2:] == u.shape[-2:]
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(20261027); torch.cuda.manual_seed_all(20261027)
        adapter = Adapter().cuda().float()
    assert bool((adapter.output.weight == 0).all()) and bool((adapter.output.bias == 0).all())
    assert bool((adapter.feature[0].weight != 0).any())
    results = {}
    with torch.no_grad():
        a = g(u.float(), qr.float())
        assert torch.equal(a, r['base_rgb'].cuda()), 'A generator path mismatch'
        for branch in ('R', 'RQ'):
            d = adapter(f, u, condition(q, branch))
            assert torch.count_nonzero(d) == 0
            rec = g(u.float() + d, qr.float())
            assert torch.equal(rec, a)
            results[branch + '_zero_identity'] = True
        assert torch.equal(g(u.float(), qr.float()), a)
    g.requires_grad_(False)
    for branch in ('R', 'RQ'):
        adapter.zero_grad(set_to_none=True)
        delta = adapter(f, u, condition(q, branch))
        loss = g(u.float() + delta, qr.float()).float().mean()
        loss.backward()
        assert adapter.output.weight.grad is not None
        assert float(adapter.output.weight.grad.abs().sum()) > 0
        results[branch + '_gradient_through_frozen_G'] = True
    with torch.no_grad():
        adapter.output.weight.fill_(1e-4)
    adapter.zero_grad(set_to_none=True)
    g(u.float() + adapter(f, u, condition(q, 'RQ')), qr.float()).float().mean().backward()
    assert adapter.feature[0].weight.grad is not None
    assert float(adapter.feature[0].weight.grad.abs().sum()) > 0
    results['earlier_gradient_after_output_update'] = True
    assert state_hash(g.state_dict()) == ghash
    results.update(status='PASS', cache_sha256=sha(path), A_generator_sha256=ghash,
                   R_RQ_nominal_parameters=sum(x.numel() for x in adapter.parameters()),
                   frozen_generator_unchanged=True,
                   precision='FP16 codec, FP32 bridge/G/Adapter', physical_gpu=gpu)
    dump(HERE / 'preflight.json', results)
    print('PREFLIGHT_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True, type=int)
    args = parser.parse_args()
    run(args.gpu)
