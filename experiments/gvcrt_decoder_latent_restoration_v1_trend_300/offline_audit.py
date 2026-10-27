"""CPU-only provenance and structural checks; never substitutes for GPU preflight."""
import ast
import io
import sys
from pathlib import Path

from common_run import HERE, ORIGINAL, REPO, dump, load, sha


def run():
    import torch
    from restoration import Adapter, condition
    sys.path.insert(0, str(REPO))
    from src.utils import stream_helper as stream
    torch.set_num_threads(2)
    torch.manual_seed(20261027)
    for path in HERE.rglob('*.py'):
        if 'local_artifacts' not in path.parts:
            ast.parse(path.read_text(), filename=str(path))
    for path, record in load(HERE / 'weight_manifest.json').items():
        assert sha(path) == record['sha256'] and Path(path).stat().st_size == record['bytes']
    cfg = load(HERE / 'effective_config.json')
    assert sha(HERE / 'train_plan.json') == cfg['plan_sha256']
    assert sha(HERE / 'evaluation_sources.json') == cfg['evaluation_sources_sha256']
    v5 = Path('/Huang_group/zyyz/home_dir/tmp/gvcrt-quant-v5-objective/experiments/gvcrt_generation_aware_quant_v5_objective_matched_trend_300')
    assert sha(v5 / 'train_plan.json') == sha(HERE / 'train_plan.json')
    assert sha(v5 / 'evaluation_sources.json') == sha(HERE / 'evaluation_sources.json')
    plan = load(HERE / 'train_plan.json')['rows']
    assert len(plan) == 300
    for row in plan:
        assert len(row['indices']) == 16
        assert row['indices'] == list(range(row['indices'][0], row['indices'][0] + 16))
    for row in load(HERE / 'data_source_check.json')['sources']:
        assert sha(row['path']) == row['sha256']
    for name, record in load(HERE / 'dependency_manifest.json').items():
        assert sha(HERE / 'dependency_snapshot' / name) == record['snapshot_sha256']
    streams = []
    for record in load(HERE / 'stream_manifest.json'):
        path = Path(record['stream_path'])
        assert sha(path) == record['stream_sha256']
        buffer = io.BytesIO(path.read_bytes()); helper = stream.SPSHelper()
        frame_rows = []
        for frame in range(64):
            header = stream.read_header(buffer)
            while header['nal_type'] == stream.NalType.NAL_SPS:
                helper.add_sps_by_id(stream.read_sps_remaining(buffer, header['sps_id']))
                header = stream.read_header(buffer)
            sps = helper.get_sps_by_id(header['sps_id'])
            qp, payload = stream.read_ip_remaining(buffer)
            assert (header['nal_type'] == stream.NalType.NAL_I) == (frame == 0)
            expected = record['external_qp'] + (0 if frame == 0 else [0, 2, 0, 1, 0, 1, 0, 1][frame % 8])
            assert qp == expected, (frame, qp, expected)
            frame_rows.append(dict(frame=frame, actual_qp=qp, payload_bytes=len(payload)))
        assert buffer.tell() == path.stat().st_size == record['total_native_bytes']
        streams.append(dict(sequence=record['sequence'], external_qp=record['external_qp'],
                            frames=64, bytes=buffer.tell(), stream_sha256=sha(path), frame_headers=frame_rows))
    adapter = Adapter()
    f = torch.randn(1, 256, 8, 8); u = torch.randn(1, 18, 4, 4)
    q = torch.rand(1, 128, 4, 4) + .5
    for branch in ('R', 'RQ'):
        delta = adapter(f, u, condition(q, branch))
        assert torch.equal(delta, torch.zeros_like(u))
    # A frozen differentiable surrogate checks Adapter connectivity only.
    # The actual A generator is validated by preflight.py on decoded frames.
    frozen = torch.nn.Conv2d(18, 3, 1).requires_grad_(False)
    frozen(u + adapter(f, u, condition(q, 'RQ'))).square().mean().backward()
    assert adapter.output.weight.grad.abs().sum() > 0
    assert adapter.feature[0].weight.grad.count_nonzero() == 0
    with torch.no_grad():
        adapter.output.weight.add_(-1e-4 * adapter.output.weight.grad)
    adapter.zero_grad(set_to_none=True)
    frozen(u + adapter(f, u, condition(q, 'RQ'))).square().mean().backward()
    assert adapter.feature[0].weight.grad.abs().sum() > 0
    # This temporary diagnostic Adapter is discarded, with no optimizer or training update.
    dependencies = []
    for root in (REPO / 'src', REPO / 'experiments/gvcrt_neural_wrapper_v2_joint'):
        for path in sorted(root.rglob('*.py')):
            rel = path.relative_to(REPO)
            dependencies.append(dict(path=str(rel), sha256=sha(path),
                                     original_matches=(ORIGINAL / rel).exists() and sha(path) == sha(ORIGINAL / rel)))
    for path in (ORIGINAL / 'src').rglob('*.so'):
        dependencies.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                                 disposition='local compiled dependency; not published'))
    dump(HERE / 'repository_dependency_manifest.json', dependencies)
    dump(HERE / 'offline_audit.json', dict(status='PASS_CPU_ONLY', weights_verified=True,
         data_sources_verified=True, plan_rows=300, streams=streams,
         adapter_parameters=sum(p.numel() for p in adapter.parameters()),
         adapter_zero_output=True, adapter_gradient_connectivity_surrogate_only=True,
         training_optimizer_updates=0, cuda_available=torch.cuda.is_available(),
         limitation='Does not validate actual GPU generator, entropy decode, or reference trajectory. GPU preflight remains required.'))
    print('OFFLINE_AUDIT_PASS; GPU preflight remains required', flush=True)


if __name__ == '__main__':
    run()
