"""One persisted 64-frame v3 task; no optimizer or weight writes."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

P = Path(__file__).resolve().parent
V2 = P.parent / 'gvcrt_generation_aware_quant_v2_skip_domain_trend_300'
V1 = P.parent / 'gvcrt_generation_aware_quant_visual_condition_v1_trend_300'
HEADER = struct.Struct('>4sBBBB8s')
MAGIC = b'GV3R'
VERSION = 1
MODES = ('A', 'MAP', 'MEAN', 'ROLL', 'IDENTITY')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def initialize(gpu):
    assert gpu in (4, 5, 6, 7)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    sys.path.insert(0, str(V2))
    import train
    train.setup_path()
    import torch
    import evaluate as old_eval
    import ablation
    from core import codec_input, unit, quality_models
    from gvc_hooks import INDEX_MAP
    from src.utils import stream_helper as stream
    torch.set_num_threads(2)
    assert Path(train.__file__).resolve() == V2 / 'train.py'
    assert Path(ablation.__file__).resolve() == P / 'ablation.py'
    return train, torch, old_eval, ablation, codec_input, unit, quality_models, INDEX_MAP, stream


def key(mode, source, qp, alpha_index):
    return f'{mode}_s300_{source}_q{qp}_a{alpha_index}'


def source_record(source):
    records = json.loads((V2 / 'evaluation_sources.json').read_text())
    return next(x for x in records if x['id'] == source)


def old_record(mode, source, qp):
    branch = 'A' if mode == 'A' else 'H11'
    path = V2 / 'server_artifacts/evaluation' / f'{branch}_s300_{source}_q{qp}.json'
    if not path.exists(): return None
    row = json.loads(path.read_text())
    assert row['status'] == 'PASS' and row['external_qp'] == qp
    assert row['sequence'] == source and row['step'] == 300
    return row


def old_stream(mode, source, qp, row):
    branch = 'A' if mode == 'A' else 'H11'
    root = V1 if branch == 'A' else V2
    path = root / 'server_artifacts/streams' / f'{branch}_s300_{source}_q{qp}.bin'
    assert path.exists() and sha(path) == row['stream_sha256'], path
    return path


def encode_header(mode, alpha_index):
    return HEADER.pack(MAGIC, VERSION, MODES.index(mode), alpha_index, 0, b'\0' * 8)


def read_header(data):
    assert len(data) >= HEADER.size
    magic, version, variant, alpha_index, reserved, padding = HEADER.unpack(data[:HEADER.size])
    assert (magic, version, reserved, padding) == (MAGIC, VERSION, 0, b'\0' * 8)
    assert variant < len(MODES) and alpha_index < 5
    mode = MODES[variant]
    assert alpha_index == 2 or mode in ('MEAN', 'ROLL')
    return mode, alpha_index, data[HEADER.size:]


def load_models(ctx, mode, alpha_index):
    train, torch, _, ablation, *_ = ctx
    branch = 'A' if mode == 'A' else 'H11'
    im, pm = train.load_models(branch)
    checkpoint = (V1 if branch == 'A' else V2) / 'server_artifacts' / branch / 'step_300.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['step'] == 300 and payload['branch'] == branch
    pm.recon_generation_net.decoder.load_state_dict(payload['G'], strict=True)
    if branch == 'H11': pm.quant_adjust.load_state_dict(payload['H'], strict=True)
    ablation.upgrade_v3(pm, mode, alpha_index)
    im.eval().requires_grad_(False)
    pm.eval().requires_grad_(False)
    return im, pm, sha(checkpoint)


def stats(tensor):
    import torch
    a = tensor.detach().float().flatten()
    return {name: float(torch.quantile(a, q)) for name, q in
            [('p01', .01), ('p50', .5), ('p99', .99)]} | {'mean': float(a.mean()), 'std': float(a.std(unbiased=False))}


def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def run(mode, source_id, qp, alpha_index, gpu, rate_only=False, reuse=True):
    ctx = initialize(gpu)
    train, torch, old_eval, ablation, codec_input, unit, quality_models, INDEX_MAP, stream = ctx
    source = source_record(source_id)
    assert source['source_sha256'] == next(x for x in json.loads((V2/'evaluation_sources.json').read_text()) if x['id']==source_id)['source_sha256']
    assert mode in ablation.MODES and 0 <= alpha_index < len(ablation.ALPHAS)
    assert alpha_index == 2 or mode in ('MEAN', 'ROLL')
    task_key = key(mode, source_id, qp, alpha_index)
    root = P/'server_artifacts'
    task_path = root/('rate_only' if rate_only else 'evaluation')/f'{task_key}.json'
    if task_path.exists():
        current = json.loads(task_path.read_text())
        stream_path = Path(current['stream_path'])
        if current.get('status') == 'PASS' and stream_path.exists() and sha(stream_path) == current['stream_sha256']:
            print('REUSE_PASS', task_key, flush=True); return current
    frames = old_eval.frames_for(source)
    h, w = source['height'], source['width']
    ch, cw = (h+63)//64*64, (w+63)//64*64
    sps = {'sps_id': 0, 'height': ch, 'width': cw, 'ec_part': int(ch*cw > 1280*720), 'use_ada_i': 0}
    existing = old_record(mode, source_id, qp) if reuse and mode in ('A','MAP') and alpha_index==2 else None
    if existing is not None:
        assert existing['checkpoint_sha256'] == sha((V1 if mode=='A' else V2)/'server_artifacts'/('A' if mode=='A' else 'H11')/'step_300.pt')
        assert existing['source_frame_indices'] == source['indices']
        native_path = old_stream(mode, source_id, qp, existing)
        native_bytes = native_path.read_bytes()
        payload_origin = str(native_path)
        encoding_seconds = float(existing.get('encoding_seconds', 0))
        estimates = None
        encoder_refs = encoder_latents = encoder_qmaps = None
        diagnostics = []
    else:
        im, pm, checkpoint_sha = load_models(ctx, mode, alpha_index)
        im.set_use_two_entropy_coders(bool(sps['ec_part']))
        pm.set_use_two_entropy_coders(bool(sps['ec_part']))
        pm.clear_dpb(); pm.set_curr_poc(0)
        native = io.BytesIO(); stream.write_sps(native, sps)
        encoder_refs=[]; encoder_latents=[]; encoder_qmaps=[]; estimates=[]; diagnostics=[]
        encoding_seconds = 0
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for i, cpu in enumerate(frames):
                x = codec_input(cpu.cuda())
                q = qp if i==0 else pm.shift_qp(qp, INDEX_MAP[i%8])
                torch.cuda.synchronize(); begin=time.time()
                y_bits=z_bits=0.0
                if i==0:
                    encoded=im.compress(x,q)
                    pm.clear_dpb(); pm.set_curr_poc(0); pm.add_ref_frame(None, encoded['x_hat'])
                    encoder_refs.append(hashlib.sha256(encoded['x_hat'].cpu().numpy().tobytes()).hexdigest())
                    encoder_latents.append(None); encoder_qmaps.append(None)
                else:
                    trace={}
                    hook=pm.hyper_enc.register_forward_hook(lambda m,args,out: trace.update(z=out.detach()))
                    original=pm.compress_prior_2x
                    def capture(y, common, spatial):
                        result=original(y,common,spatial)
                        trace['prior']=tuple(t.detach() for t in result[:4])
                        return result
                    pm.compress_prior_2x=capture
                    try: encoded=pm.compress(x,q)
                    finally: hook.remove(); pm.compress_prior_2x=original
                    zhat=trace['z'].round().clamp(-128,127).float()
                    z_bits=float(pm.get_z_bits(zhat,q,pm.bit_estimator_z).sum())
                    y0,y1,s0,s1=trace['prior']
                    d={'frame':i,'q_clamp_fraction':pm.last_q_clamp_fraction or 0.0}
                    for j,(sym,sigma) in enumerate(((y0,s0),(y1,s1))):
                        active=sigma.float()>.12
                        d[f'pass{j}_skip_fraction']=float((~active).float().mean())
                        d[f'pass{j}_saturation_fraction']=float((((sym==-128)|(sym==127))&active).float().mean())
                        scale=sigma.float()[active].clamp(.11,16)
                        symbols=sym.float()[active]
                        normal=torch.distributions.Normal(torch.zeros_like(scale),scale)
                        prob=(normal.cdf(symbols+.5)-normal.cdf(symbols-.5)).clamp_min(1e-9)
                        y_bits+=float(-torch.log2(prob).sum())
                    if mode!='A':
                        d.update({f'm_{k}':v for k,v in stats(pm.last_m_effective).items()})
                        d.update({f'rho_{k}':v for k,v in stats(pm.coding_trace['rho']).items()})
                        if mode=='ROLL':
                            assert torch.equal(torch.sort(pm.last_m_raw.flatten()).values,
                                               torch.sort(pm.last_m_variant.flatten()).values)
                    diagnostics.append(d)
                    encoder_refs.append(hashlib.sha256(pm.dpb[0].feature.cpu().numpy().tobytes()).hexdigest())
                    encoder_latents.append(hashlib.sha256(pm.coding_trace['y_hat'].cpu().numpy().tobytes()).hexdigest())
                    encoder_qmaps.append(hashlib.sha256(pm.q_actual.cpu().numpy().tobytes()).hexdigest())
                estimates.append((y_bits,z_bits))
                stream.write_ip(native,i==0,0,q,encoded['bit_stream'])
                torch.cuda.synchronize(); encoding_seconds+=time.time()-begin
        native_bytes=native.getvalue()
        payload_origin='new_encode'
        del im,pm,native
        torch.cuda.empty_cache()
    wrapped=encode_header(mode,alpha_index)+native_bytes
    stream_path=root/'streams'/f'{task_key}.bin'
    stream_path.parent.mkdir(parents=True,exist_ok=True)
    stream_path.write_bytes(wrapped)
    assert read_header(stream_path.read_bytes())[:2]==(mode,alpha_index)
    # Decoder obtains both variant and alpha solely from the persisted file.
    decoded_mode, decoded_alpha, inner = read_header(stream_path.read_bytes())
    ia,pa,checkpoint_sha=load_models(ctx,decoded_mode,decoded_alpha)
    ia.set_use_two_entropy_coders(bool(sps['ec_part']))
    pa.set_use_two_entropy_coders(bool(sps['ec_part']))
    pa.clear_dpb();pa.set_curr_poc(0)
    if existing is not None:
        assert sha((V1 if mode=='A' else V2)/'server_artifacts'/('A' if mode=='A' else 'H11')/'step_300.pt')==checkpoint_sha
    quality=None; flow=perceptual=None
    if not rate_only:
        quality=quality_models(torch.device('cuda:0'))
        compat=Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback')
        sys.path.insert(0,str(compat))
        import flolpips_compat
        flow,perceptual=flolpips_compat.load_models(torch.device('cuda:0'))
        flow.eval().requires_grad_(False);perceptual.eval().requires_grad_(False)
    buffer=io.BytesIO(inner); helper=stream.SPSHelper()
    rows=[]; transitions=[]; prev_gt=prev_out=None; decode_seconds=0
    decoder_refs=[]; decoder_latents=[]; decoder_qmaps=[]; decoder_frames=[]
    with torch.inference_mode():
        for i,cpu in enumerate(frames):
            begin=buffer.tell(); head=stream.read_header(buffer)
            while head['nal_type']==stream.NalType.NAL_SPS:
                helper.add_sps_by_id(stream.read_sps_remaining(buffer,head['sps_id']))
                head=stream.read_header(buffer)
            decoded_sps=helper.get_sps_by_id(head['sps_id'])
            assert decoded_sps==sps
            q,bits=stream.read_ip_remaining(buffer)
            assert q==(qp if i==0 else pa.shift_qp(qp,INDEX_MAP[i%8]))
            assert (head['nal_type']==stream.NalType.NAL_I)==(i==0)
            torch.cuda.synchronize(); t=time.time()
            if i==0:
                decoded=ia.decompress(bits,decoded_sps,q)
                pa.clear_dpb();pa.set_curr_poc(0);pa.add_ref_frame(None,decoded['x_hat'])
                ref=decoded['x_hat']
                decoder_latents.append(None); decoder_qmaps.append(None)
            else:
                decoded=pa.decompress(bits,decoded_sps,q)
                ref=pa.dpb[0].feature
                decoder_latents.append(hashlib.sha256(pa.decoding_trace['y_hat'].cpu().numpy().tobytes()).hexdigest())
                decoder_qmaps.append(hashlib.sha256(pa.q_actual.cpu().numpy().tobytes()).hexdigest())
            decoder_refs.append(hashlib.sha256(ref.cpu().numpy().tobytes()).hexdigest())
            if encoder_refs is not None:
                assert decoder_refs[-1]==encoder_refs[i],(task_key,i,'reference')
                assert decoder_latents[-1]==encoder_latents[i],(task_key,i,'latent')
                assert decoder_qmaps[-1]==encoder_qmaps[i],(task_key,i,'qmap')
            torch.cuda.synchronize(); decode_seconds+=time.time()-t
            row={'frame':i,'source_frame_index':source['indices'][i], 'actual_qp':q,
                 'native_bytes':buffer.tell()-begin,'payload_bytes':len(bits),
                 'estimated_y_bits':estimates[i][0] if estimates else '',
                 'estimated_z_bits':estimates[i][1] if estimates else ''}
            if not rate_only:
                output=unit(decoded['x_hat'],h,w).float();gt=cpu.cuda()
                from experiments.gvcrt_neural_wrapper_v4_progressive_joint.eval_core import frame_metrics
                metric=frame_metrics(output,gt,quality)
                row.update(metric)
                if prev_gt is not None:
                    motion=flow(prev_gt,gt)-flow(prev_out,output)
                    val=float(perceptual(prev_gt,prev_out,motion,normalize=True).item())
                    assert math.isfinite(val)
                    transitions.append({'from_frame':i-1,'to_frame':i,'FloLPIPS':val})
                prev_gt,prev_out=gt,output
                if i in (15,47): decoder_frames.append((i,output.detach().cpu()))
            rows.append(row)
    assert buffer.tell()==len(inner) and len(rows)==64
    if not rate_only: assert len(transitions)==63
    # Native codec provenance is exact for reused streams. For new streams every
    # P latent, qmap, and causal reference was checked against the encoder above.
    assert existing is None or sha(old_stream(mode,source_id,qp,existing))==hashlib.sha256(inner).hexdigest()
    payload_bytes=sum(x['payload_bytes'] for x in rows)
    native_header_bytes=len(inner)-payload_bytes
    total_bytes=stream_path.stat().st_size
    assert total_bytes==HEADER.size+native_header_bytes+payload_bytes
    result={'status':'PASS','mode':mode,'alpha_index':alpha_index,'alpha':ablation.ALPHAS[alpha_index],
            'sequence':source_id,'dataset':source['dataset'],'external_qp':qp,'step':300,
            'source_sha256':source['source_sha256'],'source_frame_indices':source['indices'],
            'source_fps':source['source_fps'],'sampling_fps':source.get('sampled_fps',source['source_fps']),
            'height':h,'width':w,'frames':64,'IP':-1,'codec_payload_bytes':payload_bytes,
            'native_header_bytes':native_header_bytes,'experiment_header_bytes':HEADER.size,
            'native_stream_bytes':len(inner),'real_total_bytes':total_bytes,
            'actual_bpp':total_bytes*8/(64*h*w),
            'P_payload_bytes':sum(x['payload_bytes'] for x in rows[1:]),
            'estimated_P_y_bits':sum(x[0] for x in estimates[1:]) if estimates else existing.get('estimated_P_main_latent_bits',''),
            'estimated_P_z_bits':sum(x[1] for x in estimates[1:]) if estimates else existing.get('estimated_P_hyper_latent_bits',''),
            'encoding_seconds':encoding_seconds,'decoding_seconds':decode_seconds,
            'peak_MiB':torch.cuda.max_memory_reserved()/2**20,
            'stream_path':str(stream_path),'stream_sha256':sha(stream_path),
            'native_stream_sha256':hashlib.sha256(inner).hexdigest(),
            'payload_origin':payload_origin,'checkpoint_sha256':checkpoint_sha,
            'task_code_sha256':sha(P/'task.py'),'ablation_code_sha256':sha(P/'ablation.py'),
            'v2_runtime_sha256':sha(V2/'runtime.py'),'imported_runtime_path':str(Path(sys.modules['runtime'].__file__).resolve()),
            'all_64_reference_states_synchronized':encoder_refs is None or decoder_refs==encoder_refs,
            'all_P_latents_qmaps_exact':encoder_refs is None or (decoder_latents==encoder_latents and decoder_qmaps==encoder_qmaps),
            'rate_only':rate_only,'measurement':'reused_native_stream_new_container_new_decode' if existing else 'new_encode_and_independent_decode',
            'FID':'not_computed','latent_partition':'interleaved_RANS_no_exact_y_z_payload_split'}
    if existing is not None:
        result['reuse_record_sha256']=sha(V2/'server_artifacts/evaluation'/f'{"A" if mode=="A" else "H11"}_s300_{source_id}_q{qp}.json')
        # The existing result is used only after the wrapped native stream and
        # checkpoint were checked. The new decoder still runs all 64 frames.
    if not rate_only:
        result.update({'LPIPS':sum(x['LPIPS'] for x in rows)/64,
                       'DISTS':sum(x['DISTS'] for x in rows)/64,
                       'FloLPIPS':sum(x['FloLPIPS'] for x in transitions)/63,
                       'PSNR':-10*math.log10(max(sum(x['pixel_SSE'] for x in rows)/(64*h*w*3),1e-15))})
        from PIL import Image
        import numpy as np
        preview=P/'previews';preview.mkdir(exist_ok=True)
        for i,output in decoder_frames:
            crop=output[0,:,200:456,400:656].permute(1,2,0).clamp(0,1).numpy()
            Image.fromarray((crop*255).round().astype(np.uint8)).save(preview/f'{task_key}_f{i}.png')
    task_path.parent.mkdir(parents=True,exist_ok=True)
    task_path.write_text(json.dumps(result,indent=2)+'\n')
    write_csv(root/'frames'/f'{task_key}.csv',rows)
    write_csv(root/'transitions'/f'{task_key}.csv',transitions)
    write_csv(root/'diagnostics'/f'{task_key}.csv',diagnostics)
    print('TASK_PASS',task_key,'rate_only',rate_only,'bpp',result['actual_bpp'],flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['A','MAP','MEAN','ROLL','IDENTITY'],required=True)
    ap.add_argument('--source',choices=['UVal0','UVal1','ParkScene','Jockey'],required=True)
    ap.add_argument('--qp',type=int,required=True)
    ap.add_argument('--alpha-index',type=int,default=2)
    ap.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True)
    ap.add_argument('--rate-only',action='store_true')
    a=ap.parse_args()
    run(a.mode,a.source,a.qp,a.alpha_index,a.gpu,a.rate_only)
