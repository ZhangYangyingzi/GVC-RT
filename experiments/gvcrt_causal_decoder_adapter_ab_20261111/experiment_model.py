"""Full audited B300 load, then freeze every inherited component."""
from common_s2 import *
import torch
from model_context import build, assert_reset
from adapter import CausalAdapter, attach


def construct(group='B300', train=False, rank=0, with_adapter=True):
    assert group in ('B300', *GROUPS)
    c = build(rank, train=train, losses=False, group='A_control', fresh_optimizer=True)
    c['optimizer'] = None
    assert sha(B300) == B300_SHA
    b = torch.load(B300, map_location='cpu', weights_only=True)
    assert b['experiment_group'] == 'B' and b['committed_updates'] == 300
    assert b['initial_A2000_sha256'] == INIT_SHA
    required = {n for n, p in c['pm'].named_parameters()
                if '.lora.' in n or n.startswith(tuple('recon_generation_net.decoder.'+p for p in TCG_PREFIXES))
                or n.startswith('recon_generation_net.mlp.3.')}
    assert set(b['trainable_state']) == required and len(required) == 52
    full = dict(c['pm'].state_dict()); full.update(b['trainable_state'])
    c['pm'].load_state_dict(full, strict=True)
    for n, t in b['trainable_state'].items():
        actual = c['pm'].state_dict()[n]
        assert torch.equal(actual.cpu(), t.to(actual.dtype)), n
    for model in (c['im'], c['pm']):
        model.eval().requires_grad_(False)
    gen = c['models']['generator']
    gen.set_temporal_cascade_grad(False)
    for n in ('tcg_low', 'tcg_mid', 'tcg_out'):
        getattr(gen, n).alignment = None
    c['provenance'].update(B300_path=str(B300), B300_SHA256=B300_SHA,
        B300_overlay_last=True, B300_keys=sorted(required), full_recovery_chain=[
            'Original I/P', 'pilot full Bridge/generator/persistent buffers',
            'B1500 FP32 TCG', 'A2000 LoRA', 'B300 52-key LAST overlay'],
        all_inherited_parameters_frozen=True, s_train=.5, s_eval=.5,
        no_E_F_H_trained_weights=True, no_original_TCG_reinitialization=True)
    c['experiment_group'] = group
    c['params'] = []; c['named'] = []; c['trainable_parameter_names'] = set()
    if with_adapter:
        with torch.random.fork_rng(devices=[rank]):
            torch.manual_seed(TRAIN_SEED)
            branch = CausalAdapter().to(c['device'])
        attach(gen, branch)
        branch.requires_grad_(train)
        c['adapter'] = branch
        c['named'] = [('recon_generation_net.decoder.causal_adapter.'+n,p) for n,p in branch.named_parameters()]
        c['params'] = list(branch.parameters())
        c['trainable_parameter_names'] = {n for n,p in c['named']}
    assert all(not p.requires_grad for n,p in c['pm'].named_parameters() if n not in c['trainable_parameter_names'])
    assert_reset(c)
    return c


def reconstruct(c, feature, qp):
    dtype = next(c['models']['bridge'].parameters()).dtype
    return c['pm'].recon_generation_net(feature.detach().to(c['device'], dtype),
        c['pm'].q_scale_recon[qp:qp+1].to(dtype), use_tcg=True, update_buffer=True)


def restore(c, saved):
    assert saved['B300_SHA256'] == B300_SHA
    c['adapter'].load_state_dict(saved['adapter'], strict=True)
    assert_reset(c)
