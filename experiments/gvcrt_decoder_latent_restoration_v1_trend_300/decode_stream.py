"""Standalone branch decoder. Reads only native stream, weights and public dimensions."""
import argparse
import copy
import io
import time

from common_run import (HERE, dump, load, load_base, save_pt, setup, sha, state_hash,
                        tensor_hash)
from restoration import Adapter, condition


def run(stream_path, source_id, branch, gpu):
    assert branch in ('A', 'G', 'R', 'RQ')
    torch, _ = setup(gpu)
    from src.utils import stream_helper as stream
    cfg = load(HERE / 'public_decode_config.json')[source_id]
    manifest = next(x for x in load(HERE / 'stream_manifest.json')
                    if x['sequence'] == source_id and x['stream_path'] == str(stream_path))
    assert sha(stream_path) == manifest['stream_sha256']
    im, pm = load_base(torch)
    external = None
    if branch != 'A':
        state_path = HERE / 'local_artifacts/checkpoints' / branch / 'step300.pt'
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        assert state['branch'] == branch and state['step'] == 300
        if branch == 'G':
            external = copy.deepcopy(pm.recon_generation_net.decoder)
            external.load_state_dict(state['model'], strict=True)
        else:
            external = Adapter().cuda().float().eval()
            external.load_state_dict(state['model'], strict=True)
        external.eval().requires_grad_(False)
    for model in (im, pm):
        model.set_use_two_entropy_coders(cfg['height'] * cfg['width'] > 1280 * 720)
    pm.clear_dpb(); pm.set_curr_poc(0)
    raw = open(stream_path, 'rb').read()
    buffer = io.BytesIO(raw); helper = stream.SPSHelper()
    dest = HERE / 'local_artifacts/decoded' / f"{source_id}_q{manifest['external_qp']}" / branch
    dest.mkdir(parents=True, exist_ok=True)
    states = []
    times = []
    with torch.inference_mode():
        for i in range(cfg['frames']):
            begin = buffer.tell()
            head = stream.read_header(buffer)
            while head['nal_type'] == stream.NalType.NAL_SPS:
                helper.add_sps_by_id(stream.read_sps_remaining(buffer, head['sps_id']))
                head = stream.read_header(buffer)
            sp = helper.get_sps_by_id(head['sps_id'])
            qp, bits = stream.read_ip_remaining(buffer)
            assert (head['nal_type'] == stream.NalType.NAL_I) == (i == 0)
            assert sp['height'] == (cfg['height'] + 63) // 64 * 64
            assert sp['width'] == (cfg['width'] + 63) // 64 * 64
            if i == 0:
                rec = im.decompress(bits, sp, qp)
                pm.add_ref_frame(None, rec['x_hat'])
                output = rec['x_hat']
                state = dict(reference=tensor_hash(rec['x_hat']), y_hat=None, f=None,
                             u=None, q_actual=None, q_recon=None,
                             reference_rgb=tensor_hash(pm.dpb[0].frame), reference_poc=pm.dpb[0].poc,
                             output_rgb=tensor_hash(output))
            else:
                capture = {}
                hooks = [pm.recon_generation_net.register_forward_pre_hook(
                    lambda _, args: capture.update(f=args[0].detach())),
                    pm.recon_generation_net.decoder.register_forward_pre_hook(
                    lambda _, args: capture.update(u=args[0].detach(), q_recon=args[1].detach()))]
                try:
                    rec = pm.decompress(bits, sp, qp)
                finally:
                    for hook in hooks:
                        hook.remove()
                state = dict(reference=tensor_hash(pm.dpb[0].feature),
                             y_hat=tensor_hash(pm.decoding_trace['y_hat']),
                             f=tensor_hash(capture['f']), u=tensor_hash(capture['u']),
                             q_actual=tensor_hash(pm.q_actual),
                             q_recon=tensor_hash(capture['q_recon']),
                             reference_rgb=tensor_hash(pm.dpb[0].frame),
                             reference_poc=pm.dpb[0].poc)
                if branch == 'A':
                    output = rec['x_hat']
                else:
                    u, qr, f = capture['u'].float(), capture['q_recon'].float(), capture['f']
                    torch.cuda.synchronize(); tick = time.perf_counter()
                    if branch == 'G':
                        output = external(u, qr)
                        delta = None
                    else:
                        delta = external(f, u, condition(pm.q_actual, branch))
                        output = pm.recon_generation_net.decoder(u + delta, qr)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - tick)
                    if i in (15, 31, 63):
                        save_pt(torch, dest / f'diagnostics_{i:02d}.pt',
                                dict(q_actual=pm.q_actual.detach().cpu(),
                                     delta_u=None if delta is None else delta.detach().cpu()))
                # The native decoder has already installed A's f and RGB as its reference.
                assert state['reference'] == tensor_hash(pm.dpb[0].feature)
                assert state['reference_rgb'] == tensor_hash(pm.dpb[0].frame)
                state['output_rgb'] = tensor_hash(output)
            state.update(frame=i, actual_qp=qp, payload_bytes=len(bits),
                         consumed_bytes=buffer.tell() - begin)
            states.append(state)
            save_pt(torch, dest / f'{i:02d}.pt', output.detach().cpu())
    assert buffer.tell() == len(raw) and len(states) == 64
    assert sum(x['consumed_bytes'] for x in states) == len(raw)
    result = dict(branch=branch, sequence=source_id, external_qp=manifest['external_qp'],
                  stream_path=str(stream_path), stream_sha256=sha(stream_path),
                  consumed_bytes=len(raw), real_total_bytes=len(raw),
                  bpp=manifest['bpp'], decoded_path=str(dest), frames=64,
                  states=states, output_seconds=times,
                  base_G_state_sha256=state_hash(pm.recon_generation_net.decoder.state_dict()),
                  branch_model_state_sha256=None if external is None else state_hash(external.state_dict()),
                  branch_checkpoint_sha256=None if branch == 'A' else sha(state_path),
                  input_contract='native stream, branch weights, fixed public dimensions only',
                  precision='FP16 codec; FP32 bridge/G/Adapter')
    dump(dest / 'receipt.json', result)
    print('DECODED', source_id, manifest['external_qp'], branch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--branch', choices=['A', 'G', 'R', 'RQ'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    run(args.stream, args.source, args.branch, args.gpu)
