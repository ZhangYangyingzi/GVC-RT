"""Read-only checkpoint comparison and a small CPU receiver equivalence probe."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import types

OUT = Path(__file__).resolve().parent
EXPERIMENT = OUT.parents[1]
WORKTREE = EXPERIMENT.parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--expanded-step0', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(20261105)
    paths = {
        'original_P': args.project_root / 'checkpoints/GVC-RT_P.pt',
        'original_I': args.project_root / 'checkpoints/GVC-RT_I.pt',
        'v634_step0': args.project_root / 'experiments/gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/branches/N_noP_BG_FD/checkpoints/adaptation_step_0.pt',
        'pilot_step0': args.project_root / 'experiments/gvcrt_tcg_stage1_pilot/checkpoints/adaptation_step_0.pt',
        'expanded_step0': args.expanded_step0,
    }
    fingerprints = {k: {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for k, p in paths.items()}
    measurements = list(csv.DictReader((EXPERIMENT/'metrics_v3.csv').open(newline='')))
    parent0_hashes = {r['checkpoint_sha256'] for r in measurements if r['model_state'] == 'parent0'}
    assert parent0_hashes == {fingerprints['expanded_step0']['sha256']}
    original = torch.load(paths['original_P'], map_location='cpu', weights_only=True)['state_dict']
    source = torch.load(paths['v634_step0'], map_location='cpu', weights_only=True)
    pilot = torch.load(paths['pilot_step0'], map_location='cpu', weights_only=True)
    expanded = torch.load(paths['expanded_step0'], map_location='cpu', weights_only=True)
    comparisons = {}
    for name, prefix in [('bridge', 'recon_generation_net.mlp.'), ('generator', 'recon_generation_net.decoder.')]:
        reference = {k[len(prefix):]: v for k, v in original.items() if k.startswith(prefix)}
        assert set(reference) == set(source[name])
        stats = dict(tensors=len(reference), elements=sum(t.numel() for t in reference.values()),
                     equal_raw=0, equal_after_loader_FP16_to_FP32=0, equal_pilot=0,
                     max_raw_abs_difference=0.0, max_after_cast_abs_difference=0.0)
        for key, value in source[name].items():
            raw = reference[key]
            cast = raw.half().float()
            stats['equal_raw'] += int(torch.equal(value, raw))
            stats['equal_after_loader_FP16_to_FP32'] += int(torch.equal(value, cast))
            stats['equal_pilot'] += int(torch.equal(value, pilot['models'][name][key]))
            stats['max_raw_abs_difference'] = max(stats['max_raw_abs_difference'], float((value.double()-raw.double()).abs().max()))
            stats['max_after_cast_abs_difference'] = max(stats['max_after_cast_abs_difference'], float((value.double()-cast.double()).abs().max()))
        assert stats['equal_after_loader_FP16_to_FP32'] == stats['equal_pilot'] == stats['tensors']
        comparisons[name] = stats
    tcg = expanded['tcg_model_state']
    assert set(tcg) == {k for k in pilot['models']['generator'] if k.startswith('tcg_')}
    assert all(torch.equal(v, pilot['models']['generator'][k]) for k, v in tcg.items())
    terminal = {k: int(torch.count_nonzero(v)) for k, v in tcg.items() if '.fusion.4.' in k}
    assert len(terminal) == 6 and all(v == 0 for v in terminal.values())
    assert not source['wrapper'] and not pilot['models']['wrapper']
    assert source['adaptation_step'] == source['committed_optimizer_updates'] == 0
    assert not source['optimizer']['state']

    # Load just the two receiver modules from the original tracked release.
    # These files run on CPU using shared layer code, whose hashes are recorded.
    sys.path.insert(0, str(WORKTREE))
    from src.models.video_model_gvcrt import PretrainedReconWrapper
    commit = subprocess.check_output(['git', 'rev-parse', 'b2a0895c0'], cwd=args.project_root, text=True).strip()
    sources = {}
    for short, filename in [('decoder', 'improved_model_gvcrt.py'), ('video', 'video_model_gvcrt.py')]:
        code = subprocess.check_output(['git', 'show', f'{commit}:src/models/{filename}'], cwd=args.project_root)
        sources[short] = dict(git_commit=commit, git_path=f'src/models/{filename}', sha256=hashlib.sha256(code).hexdigest())
        mod = types.ModuleType(f'src.models._baseline_audit_{short}')
        mod.__package__ = 'src.models'
        exec(compile(code, f'<{commit}:{filename}>', 'exec'), mod.__dict__)
        if short == 'decoder':
            original_decoder = mod.Decoder
        else:
            mod.PretrainedDecoder = original_decoder
            original_wrapper = mod.PretrainedReconWrapper
    old = original_wrapper().half().eval()
    prefix = 'recon_generation_net.'
    old.load_state_dict({k[len(prefix):]: v for k, v in original.items() if k.startswith(prefix)}, strict=True)
    current = PretrainedReconWrapper().half().eval()
    current.mlp.load_state_dict(pilot['models']['bridge'], strict=True)
    current.decoder.load_state_dict(pilot['models']['generator'], strict=True)
    current.decoder.load_state_dict(tcg, strict=False)
    probes = []
    with torch.inference_mode():
        for qp in (0, 5, 9):
            current.reset_generator_buffer()
            for frame in range(3):
                x = torch.randn(1, 256, 8, 12).half()
                quant = original['q_scale_recon'][qp:qp+1].half()
                a, b = old(x, quant), current(x, quant)
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                result = dict(qp=qp, frame=frame, input_shape=list(x.shape), output_shape=list(a.shape),
                              exact_equal=torch.equal(a, b), max_abs_difference=float((a.float()-b.float()).abs().max()))
                assert result['exact_equal'], result
                probes.append(result)
    shared_sources = {name: sha(WORKTREE/name) for name in [
        'src/layers/layers.py', 'src/layers/cuda_inference.py',
        'src/models/improved_model_gvcrt.py', 'src/models/video_model_gvcrt.py',
        'expericent_generation_input/expericent_interface_causal_controls_v9/src/gvc_hooks.py',
        'experiments/gvcrt_tcg_stage1_data_expansion_ab_groupisolated_v2/v3_evaluate.py',
    ]}
    result = dict(status='PASS', user_confirmed_original_checkpoint_directory=str(args.project_root/'checkpoints'),
                  files=fingerprints, comparisons=comparisons,
                  parent0_metric_checkpoint_hash_matches=True,
                  v634_adaptation_step=source['adaptation_step'], v634_optimizer_state_entries=len(source['optimizer']['state']),
                  pilot_to_expanded_TCG_state_exact=True, terminal_TCG_nonzero_counts=terminal,
                  original_release_sources=sources, current_source_sha256=shared_sources,
                  receiver_probe=dict(device='CPU', dtype='FP16', cases=probes, shared_current_layer_implementation=True),
                  conclusion='v6_34 step0 and the non-TCG pilot receiver tensors exactly match user-confirmed original P weights after the existing loader precision conversion; expanded step0 has zero TCG residual.',
                  scope='Weight identity and small synthetic receiver probes only. No independent full-dataset original-code RANS evaluation was performed. No training or metrics changes.')
    (OUT/'baseline_identity_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], comparisons=comparisons, receiver_probe_cases=len(probes)), indent=2))


if __name__ == '__main__':
    main()
