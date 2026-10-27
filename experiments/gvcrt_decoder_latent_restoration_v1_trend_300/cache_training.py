"""One native A stream and one independent A decode for each planned window."""
import argparse
import io
import time

from common_run import HERE, dump, load, load_base, save_pt, setup, sha, tensor_hash


def load_frames(row, device):
    import cv2
    import torch
    cap = cv2.VideoCapture(row['path'])
    assert cap.isOpened(), row['path']
    frames = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, row['indices'][0])
        for index in row['indices']:
            ok, bgr = cap.read()
            assert ok, (row['path'], index)
            crop = cv2.cvtColor(bgr[row['y']:row['y'] + 256,
                                        row['x']:row['x'] + 256], cv2.COLOR_BGR2RGB)
            assert crop.shape == (256, 256, 3)
            frames.append(torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].to(device).float() / 255)
    finally:
        cap.release()
    return frames


def run(gpu, start=0, stop=300):
    torch, _ = setup(gpu)
    from core import codec_input
    from gvc_hooks import INDEX_MAP
    from src.utils import stream_helper as stream
    plan = load(HERE / 'train_plan.json')['rows']
    assert len(plan) == 300 and 0 <= start <= stop <= 300
    enc_i, enc_p = load_base(torch)
    dec_i, dec_p = load_base(torch)
    sps = dict(sps_id=0, height=256, width=256, ec_part=0, use_ada_i=0)
    for model in (enc_i, dec_i, enc_p, dec_p):
        model.set_use_two_entropy_coders(False)
    for n in range(start, stop):
        target = HERE / 'local_artifacts/training_cache' / f'{n:04d}.pt'
        receipt = target.with_suffix('.json')
        if target.exists() and receipt.exists():
            previous = load(receipt)
            expected_code = {name: sha(HERE / name) for name in
                             ('cache_training.py', 'common_run.py', 'restoration.py')}
            if (sha(target) == previous['cache_sha256']
                    and previous['plan_sha256'] == sha(HERE / 'train_plan.json')
                    and previous['weight_manifest_sha256'] == sha(HERE / 'weight_manifest.json')
                    and previous['code_sha256'] == expected_code):
                continue
        row = plan[n]
        began = time.time()
        gt = load_frames(row, torch.device('cuda:0'))
        enc_p.clear_dpb(); enc_p.set_curr_poc(0)
        dec_p.clear_dpb(); dec_p.set_curr_poc(0)
        coded = io.BytesIO(); stream.write_sps(coded, sps)
        encoder_states = []
        with torch.inference_mode():
            for i, frame in enumerate(gt):
                q = row['external_qp'] if i == 0 else enc_p.shift_qp(row['external_qp'], INDEX_MAP[i % 8])
                if i == 0:
                    e = enc_i.compress(codec_input(frame), q)
                    enc_p.add_ref_frame(None, e['x_hat'])
                else:
                    e = enc_p.compress(codec_input(frame), q)
                encoder_states.append(None if i == 0 else dict(
                    reference=tensor_hash(enc_p.dpb[0].feature),
                    y_hat=tensor_hash(enc_p.coding_trace['y_hat']),
                    q_actual=tensor_hash(enc_p.q_actual)))
                stream.write_ip(coded, i == 0, 0, q, e['bit_stream'])
        stream_path = HERE / 'local_artifacts/training_streams' / f'{n:04d}.bin'
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream_path.write_bytes(coded.getvalue())
        buffer = io.BytesIO(stream_path.read_bytes()); helper = stream.SPSHelper()
        records = []
        with torch.inference_mode():
            for i in range(16):
                header = stream.read_header(buffer)
                while header['nal_type'] == stream.NalType.NAL_SPS:
                    helper.add_sps_by_id(stream.read_sps_remaining(buffer, header['sps_id']))
                    header = stream.read_header(buffer)
                sp = helper.get_sps_by_id(header['sps_id'])
                q, bits = stream.read_ip_remaining(buffer)
                assert (header['nal_type'] == stream.NalType.NAL_I) == (i == 0)
                if i == 0:
                    d = dec_i.decompress(bits, sp, q)
                    dec_p.add_ref_frame(None, d['x_hat'])
                else:
                    capture = {}
                    hooks = [dec_p.recon_generation_net.register_forward_pre_hook(
                        lambda _, args: capture.update(f=args[0].detach())),
                        dec_p.recon_generation_net.decoder.register_forward_pre_hook(
                        lambda _, args: capture.update(u=args[0].detach(), q_recon=args[1].detach()))]
                    try:
                        d = dec_p.decompress(bits, sp, q)
                    finally:
                        for hook in hooks:
                            hook.remove()
                    assert tensor_hash(dec_p.dpb[0].feature) == encoder_states[i]['reference']
                    assert tensor_hash(dec_p.decoding_trace['y_hat']) == encoder_states[i]['y_hat']
                    assert tensor_hash(dec_p.q_actual) == encoder_states[i]['q_actual']
                    if i >= 12:
                        record = {k: v.cpu().contiguous() for k, v in capture.items()}
                        record['q_actual'] = dec_p.q_actual.detach().cpu().contiguous()
                        record['base_rgb'] = d['x_hat'].detach().cpu().contiguous()
                        record['frame'] = i; record['source_frame_index'] = row['indices'][i]
                        record['actual_qp'] = q
                        record['reference_sha256'] = tensor_hash(dec_p.dpb[0].feature)
                        record['y_hat_sha256'] = tensor_hash(dec_p.decoding_trace['y_hat'])
                        records.append(record)
        assert buffer.tell() == len(coded.getvalue()) and len(records) == 4
        save_pt(torch, target, dict(records=records, row=row, stream_sha256=sha(stream_path),
                                    plan_sha256=sha(HERE / 'train_plan.json'),
                                    code_sha256={name: sha(HERE / name) for name in
                                                 ('cache_training.py', 'common_run.py', 'restoration.py')},
                                    weight_manifest_sha256=sha(HERE / 'weight_manifest.json')))
        dump(receipt, dict(index=n, source_sha256=row['source_sha256'], stream_path=str(stream_path),
                           stream_sha256=sha(stream_path), cache_sha256=sha(target),
                           plan_sha256=sha(HERE / 'train_plan.json'),
                           weight_manifest_sha256=sha(HERE / 'weight_manifest.json'),
                           code_sha256={name: sha(HERE / name) for name in
                                        ('cache_training.py', 'common_run.py', 'restoration.py')},
                           seconds=time.time() - began,
                           frames=[{k: v for k, v in r.items() if k not in ('f', 'u', 'q_actual', 'q_recon', 'base_rgb')}
                                   for r in records]))
        print('CACHE', n + 1, '/ 300', round(time.time() - began, 2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=300)
    args = parser.parse_args()
    run(args.gpu, args.start, args.stop)
