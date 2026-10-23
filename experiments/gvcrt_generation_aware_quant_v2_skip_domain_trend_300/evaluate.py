"""Real stream encode followed by fresh bitstream-only model decode."""
import argparse, csv, hashlib, io, json, math, os, sys, time
from pathlib import Path
P = Path(__file__).resolve().parent
REPO = P.parents[1]
sys.path.insert(0,str(P))


def frames_for(source):
    import cv2, numpy as np, torch, train
    from PIL import Image
    output=[]
    if source['file_type']=='mp4':
        assert train.sha(source['path'])==source['source_sha256']
        cap=cv2.VideoCapture(source['path'])
        assert cap.isOpened(),source['path']
        cap.set(cv2.CAP_PROP_POS_FRAMES,source['indices'][0])
        try:
            for index in source['indices']:
                ok,frame=cap.read()
                assert ok,(source['id'],index)
                a=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                output.append(torch.from_numpy(a.copy()).permute(2,0,1).unsqueeze(0).float()/255)
        finally:cap.release()
    else:
        for position,index in enumerate(source['indices']):
            name=f'im{position+1:05d}.png' if source['id']=='ParkScene' else f'frame_{position:06d}.png'
            path=Path(source['path'])/name
            expected=source.get('frame_png_sha256') or source.get('frame_file_sha256')
            if expected is not None:assert train.sha(path)==expected[position]
            with Image.open(path) as image:a=np.asarray(image.convert('RGB'),dtype=np.uint8).copy()
            output.append(torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255)
    assert len(output)==64 and all(x.shape==(1,3,source['height'],source['width']) for x in output)
    return output


def load_checkpoint(train, branch, step):
    import torch
    im,pm=train.load_models(branch)
    old_branch = {'A':'A','H00':'B'}.get(branch)
    path = (P.parent/'gvcrt_generation_aware_quant_visual_condition_v1_trend_300'/'server_artifacts'/old_branch/f'step_{step}.pt') if old_branch else P/'server_artifacts'/branch/f'step_{step}.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['branch']==(old_branch or branch) and payload['step']==step
    pm.recon_generation_net.decoder.load_state_dict(payload['G'],strict=True)
    if branch != 'A':pm.quant_adjust.load_state_dict(payload['H'],strict=True)
    if False:pm.visual_condition.load_state_dict(payload['V'],strict=True)
    im.eval().requires_grad_(False);pm.eval().requires_grad_(False)
    return im,pm,train.sha(path)


def write_csv(path,rows):
    assert rows
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def save_preview(path,tensor):
    import numpy as np
    from PIL import Image
    a=(tensor.detach().float().clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    Image.fromarray(a).save(path)


def evaluate(branch,step,source_id,qp,gpu):
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    import train
    train.setup_path()
    import torch
    from core import codec_input, unit, quality_models
    from gvc_hooks import INDEX_MAP
    from src.utils import stream_helper as stream
    from importlib.util import spec_from_file_location, module_from_spec
    torch.set_num_threads(2)
    source=next(x for x in json.loads((P/'evaluation_sources.json').read_text()) if x['id']==source_id)
    assert step==300 or source_id.startswith('UVal')
    key=f'{branch}_s{step}_{source_id}_q{qp}'
    task=P/'server_artifacts/evaluation'/f'{key}.json'
    if task.exists() and json.loads(task.read_text()).get('status')=='PASS':return
    frames=frames_for(source)
    h,w=source['height'],source['width'];ch=(h+63)//64*64;cw=(w+63)//64*64
    sps={'sps_id':0,'height':ch,'width':cw,'ec_part':int(ch*cw>1280*720),'use_ada_i':0}
    im,pm,checkpoint_sha=load_checkpoint(train,branch,step)
    im.set_use_two_entropy_coders(bool(sps['ec_part']))
    pm.set_use_two_entropy_coders(bool(sps['ec_part']))
    pm.clear_dpb();pm.set_curr_poc(0)
    buffer=io.BytesIO();stream.write_sps(buffer,sps)
    encoder_hashes=[];encoder_latents=[];encoder_qmaps=[];actual=[];estimates=[];encoder_seconds=0;payload_bytes=0
    estimated_P_y_bits=0.0;estimated_P_z_bits=0.0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i,cpu in enumerate(frames):
            x=codec_input(cpu.cuda());q=qp if i==0 else pm.shift_qp(qp,INDEX_MAP[i%8]);actual.append(q)
            torch.cuda.synchronize();t=time.time()
            frame_y_bits=frame_z_bits=0.0
            if i==0:
                encoded=im.compress(x,q);pm.clear_dpb();pm.set_curr_poc(0)
                pm.add_ref_frame(None,encoded['x_hat'])
            else:
                trace={}
                zh=pm.hyper_enc.register_forward_hook(lambda m,args,out:trace.update(z=out.detach()))
                yh=pm.compress_prior_2x
                def prior_capture(y, common, spatial):
                    result=yh(y,common,spatial)
                    trace['prior']=tuple(t.detach() for t in result[:4])
                    return result
                pm.compress_prior_2x=prior_capture
                try:encoded=pm.compress(x,q)
                finally:
                    zh.remove();pm.compress_prior_2x=yh
                zhat=trace['z'].round().clamp(-128,127).float()
                frame_z_bits=float(pm.get_z_bits(zhat,q,pm.bit_estimator_z).sum())
                estimated_P_z_bits+=frame_z_bits
                y0,y1,s0,s1=trace['prior']
                for sym,sigma in ((y0,s0),(y1,s1)):
                    active=sigma.float()>.12
                    scale=sigma.float()[active].clamp(.11,16)
                    symbols=sym.float()[active]
                    gaussian=torch.distributions.Normal(torch.zeros_like(scale),scale)
                    prob=(gaussian.cdf(symbols+.5)-gaussian.cdf(symbols-.5)).clamp_min(1e-9)
                    frame_y_bits+=float(-torch.log2(prob).sum())
                estimated_P_y_bits+=frame_y_bits
            estimates.append((frame_y_bits,frame_z_bits))
            if i:
                encoder_latents.append(hashlib.sha256(pm.coding_trace['y_hat'].cpu().numpy().tobytes()).hexdigest())
                encoder_qmaps.append(hashlib.sha256(pm.q_actual.cpu().numpy().tobytes()).hexdigest())
            else:
                encoder_latents.append(None);encoder_qmaps.append(None)
            torch.cuda.synchronize();encoder_seconds+=time.time()-t
            payload_bytes+=len(encoded['bit_stream'])
            stream.write_ip(buffer,i==0,0,q,encoded['bit_stream'])
            if i==0:encoder_hashes.append(hashlib.sha256(encoded['x_hat'].cpu().numpy().tobytes()).hexdigest())
            else:encoder_hashes.append(hashlib.sha256(pm.dpb[0].feature.cpu().numpy().tobytes()).hexdigest())
    encoded_path=P/'server_artifacts/streams'/f'{key}.bin'
    encoded_path.parent.mkdir(parents=True,exist_ok=True)
    encoded_path.write_bytes(buffer.getvalue())
    total_bytes=encoded_path.stat().st_size
    del im,pm,buffer
    torch.cuda.empty_cache()
    # Fresh models and file contents are the decoder's only codec inputs.
    ia,pa,check2=load_checkpoint(train,branch,step)
    assert check2==checkpoint_sha
    ia.set_use_two_entropy_coders(bool(sps['ec_part']))
    pa.set_use_two_entropy_coders(bool(sps['ec_part']))
    pa.clear_dpb();pa.set_curr_poc(0)
    quality=quality_models(torch.device('cuda:0'))
    compat=Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback')
    sys.path.insert(0,str(compat))
    import flolpips_compat
    flow,perceptual=flolpips_compat.load_models(torch.device('cuda:0'))
    flow.eval().requires_grad_(False);perceptual.eval().requires_grad_(False)
    stream_file=io.BytesIO(encoded_path.read_bytes());helper=stream.SPSHelper()
    rows=[];trans=[];previous_gt=previous_out=None;decode_seconds=0;last_qmap=None;last_cond=None
    movie_frames=[]
    hook=None
    if False:
        hook=pa.visual_condition.register_forward_hook(lambda m,args,out: globals().update(_eval_condition_map=out[1].detach()))
    with torch.inference_mode():
        for i,cpu in enumerate(frames):
            begin=stream_file.tell();head=stream.read_header(stream_file)
            while head['nal_type']==stream.NalType.NAL_SPS:
                helper.add_sps_by_id(stream.read_sps_remaining(stream_file,head['sps_id']))
                head=stream.read_header(stream_file)
            read_sps=helper.get_sps_by_id(head['sps_id'])
            assert read_sps==sps
            q,bits=stream.read_ip_remaining(stream_file)
            assert q==actual[i] and (head['nal_type']==stream.NalType.NAL_I)==(i==0)
            torch.cuda.synchronize();t=time.time()
            if i==0:
                decoded=ia.decompress(bits,read_sps,q)
                pa.clear_dpb();pa.set_curr_poc(0);pa.add_ref_frame(None,decoded['x_hat'])
                decoded_reference_hash=hashlib.sha256(decoded['x_hat'].cpu().numpy().tobytes()).hexdigest()
            else:decoded=pa.decompress(bits,read_sps,q)
            if i:
                decoded_reference_hash=hashlib.sha256(pa.dpb[0].feature.cpu().numpy().tobytes()).hexdigest()
            if i:
                assert hashlib.sha256(pa.decoding_trace['y_hat'].cpu().numpy().tobytes()).hexdigest()==encoder_latents[i],(key,i,'latent_mismatch')
                assert hashlib.sha256(pa.q_actual.cpu().numpy().tobytes()).hexdigest()==encoder_qmaps[i],(key,i,'qmap_mismatch')
            assert decoded_reference_hash==encoder_hashes[i],(key,i,'causal_reference_mismatch')
            torch.cuda.synchronize();decode_seconds+=time.time()-t
            output=unit(decoded['x_hat'],h,w).float()
            gt=cpu.cuda()
            from experiments.gvcrt_neural_wrapper_v4_progressive_joint.eval_core import frame_metrics
            metric=frame_metrics(output,gt,quality)
            rows.append({'frame':i,'source_frame_index':source['indices'][i],
                         'actual_qp':q,'real_bytes':stream_file.tell()-begin,
                         'payload_bytes':len(bits),'estimated_y_bits':estimates[i][0],'estimated_z_bits':estimates[i][1],**metric})
            if previous_gt is not None:
                motion=flow(previous_gt,gt)-flow(previous_out,output)
                assert bool(torch.isfinite(motion).all())
                val=float(perceptual(previous_gt,previous_out,motion,normalize=True).item())
                assert math.isfinite(val)
                trans.append({'from_frame':i-1,'to_frame':i,'FloLPIPS':val})
            previous_gt,previous_out=gt,output
            if i==15:
                preview=P/'previews';preview.mkdir(exist_ok=True)
                crop=(slice(None),slice(None),slice(200,456),slice(400,656))
                save_preview(preview/f'{key}_frame15.png',output[crop])
                save_preview(preview/f'{source_id}_gt_frame15.png',gt[crop])
                if branch != 'A':last_qmap=pa.q_actual.detach().float().mean(1,keepdim=True).cpu()
                if False:last_cond=globals()['_eval_condition_map'].float().square().mean(1,keepdim=True).sqrt().cpu()
            if branch in ('A','D') and step==300 and source_id=='UVal0' and qp==2 and 12<=i<=15:
                import numpy as np
                both=torch.cat((gt[:,:,:,400:656][:,:,200:456,:],
                                output[:,:,:,400:656][:,:,200:456,:]),dim=-1)
                array=(both[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                movie_frames.append(array)
    if hook:hook.remove()
    assert stream_file.tell()==total_bytes and len(rows)==64 and len(trans)==63
    assert sum(x['real_bytes'] for x in rows)==total_bytes
    if last_qmap is not None:
        from PIL import Image
        import numpy as np
        a=last_qmap[0,0].numpy();a=np.clip((a-a.min())/(np.ptp(a)+1e-8)*255,0,255).astype('uint8')
        Image.fromarray(a).resize((256,256)).save(P/'previews'/f'{key}_quant.png')
    if last_cond is not None:
        from PIL import Image
        import numpy as np
        a=last_cond[0,0].numpy();a=np.clip((a-a.min())/(np.ptp(a)+1e-8)*255,0,255).astype('uint8')
        Image.fromarray(a).resize((256,256)).save(P/'previews'/f'{key}_condition.png')
    if movie_frames:
        import cv2
        movie=P/'previews'/f'{key}_GT_left_reconstruction_right.mp4'
        writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'mp4v'),4,(512,256))
        assert writer.isOpened()
        for array in movie_frames:
            canvas=cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
            cv2.putText(canvas,f'{branch} step {step}  {total_bytes*8/(64*h*w):.5f} bpp',
                        (8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2,cv2.LINE_AA)
            writer.write(canvas)
        writer.release()
    result={'status':'PASS','branch':branch,'step':step,'sequence':source_id,
            'dataset':source['dataset'],'external_qp':qp,'actual_qps':actual,
            'source_fps':source['source_fps'],'sampling_fps':source.get('sampled_fps',source['source_fps']),
            'source_frame_indices':source['indices'],
            'height':h,'width':w,'frames':64,'IP':-1,
            'real_total_bytes':total_bytes,'payload_bytes':payload_bytes,
            'header_bytes':total_bytes-payload_bytes,
            'P_payload_bytes':sum(r['payload_bytes'] for r in rows[1:]),
            'estimated_P_total_bits':estimated_P_y_bits+estimated_P_z_bits,
            'actual_P_payload_bits':sum(r['payload_bytes'] for r in rows[1:])*8,
            'estimated_P_main_latent_bits':estimated_P_y_bits,
            'estimated_P_hyper_latent_bits':estimated_P_z_bits,
            'latent_rate_attribution':'analytic_CDF_proxy_for_P_only; actual_RANS_y_z_partition_unavailable',
            'actual_bpp':total_bytes*8/(64*h*w),
            'kbps_20fps':total_bytes*8*20/(64*1000),
            'LPIPS':sum(x['LPIPS'] for x in rows)/64,
            'DISTS':sum(x['DISTS'] for x in rows)/64,
            'FloLPIPS':sum(x['FloLPIPS'] for x in trans)/63,
            'PSNR':-10*math.log10(max(sum(x['pixel_SSE'] for x in rows)/(64*h*w*3),1e-15)),
            'encoding_seconds':encoder_seconds,'decoding_seconds':decode_seconds,
            'peak_MiB':torch.cuda.max_memory_reserved()/2**20,
            'generator_parameters':sum(p.numel() for p in pa.recon_generation_net.decoder.parameters()),
            'quant_parameters':sum(p.numel() for p in pa.quant_adjust.parameters()) if branch != 'A' else 0,
            'condition_parameters':sum(p.numel() for p in pa.visual_condition.parameters()) if False else 0,
            'stream_path':str(encoded_path),'stream_sha256':train.sha(encoded_path),
            'checkpoint_sha256':checkpoint_sha,'independent_decode':True,
            'all_64_reference_states_synchronized':True,
            'all_P_latents_qmaps_exact':True,
            'measurement':'new_encode_and_independent_decode',
            'evaluation_code_sha256':train.sha(Path(__file__)),
            'FID':'not_computed','latent_partition':'combined_RANS_payload_not_separately_addressable'}
    task.parent.mkdir(parents=True,exist_ok=True)
    task.write_text(json.dumps(result,indent=2)+'\n')
    write_csv(P/'server_artifacts/evaluation_frames'/f'{key}.csv',rows)
    write_csv(P/'server_artifacts/evaluation_transitions'/f'{key}.csv',trans)
    print('EVAL_PASS',key,result['actual_bpp'],result['LPIPS'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--branch',choices=['A','H00','H10','H01','H11'],required=True)
    ap.add_argument('--step',type=int,choices=[0,100,300],required=True)
    ap.add_argument('--source',choices=['UVal0','UVal1','ParkScene','Jockey'],required=True)
    ap.add_argument('--qp',type=int,required=True)
    ap.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True)
    a=ap.parse_args();evaluate(a.branch,a.step,a.source,a.qp,a.gpu)
