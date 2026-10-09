import importlib.util
import io
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from experiment_utils import ROOT, REPO, V4, sha, write

V2=V4.parent/'gvcrt_neural_wrapper_v2_joint'
sys.path.insert(0,str(V2))
import core
spec=importlib.util.spec_from_file_location('v4_eval',V4/'eval_core.py')
eco=importlib.util.module_from_spec(spec); spec.loader.exec_module(eco)
COMPAT=Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback')
sys.path.insert(0,str(COMPAT))
import flolpips_compat as flo
FID_SOURCE=REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py'
spec=importlib.util.spec_from_file_location('v6_fid',FID_SOURCE)
fid=importlib.util.module_from_spec(spec); spec.loader.exec_module(fid)

def load_metrics(device):
    flow,perceptual=flo.load_models(device)
    return flow,perceptual,fid.InceptionPool3().to(device).eval().requires_grad_(False)

@torch.inference_mode()
def measure(frames,stream_path,joint,device,models,feature_path,expected):
    flow,perceptual,inception=models
    i_model,p_model=eco.build_models(device,None if joint is None else joint[0])
    before=core.compression_hash(i_model,p_model)
    p_model.clear_dpb(); p_model.set_curr_poc(0)
    data=Path(stream_path).read_bytes(); buffer=io.BytesIO(data); helper=eco.SPSHelper()
    refs,outs,transitions,decoded_hashes=[],[],[],[]
    previous_reference=previous_output=None
    for index,cpu in enumerate(frames):
        header=eco.read_header(buffer)
        while header['nal_type']==eco.NalType.NAL_SPS:
            helper.add_sps_by_id(eco.read_sps_remaining(buffer,header['sps_id'])); header=eco.read_header(buffer)
        sps=helper.get_sps_by_id(header['sps_id']); qp,bits=eco.read_ip_remaining(buffer)
        if header['nal_type']==eco.NalType.NAL_I:
            decoded=i_model.decompress(bits,sps,qp); p_model.clear_dpb(); p_model.add_ref_frame(None,decoded['x_hat'])
        else: decoded=p_model.decompress(bits,sps,qp)
        decoded_hashes.append(eco.tensor_sha(decoded['x_hat']))
        output=core.unit(decoded['x_hat'],1080,1920); reference=cpu.to(device)
        assert output.shape==reference.shape==(1,3,1080,1920)
        refs.append(inception(reference).cpu().numpy()); outs.append(inception(output).cpu().numpy())
        if previous_reference is not None:
            flow_difference=flow(previous_reference,reference)-flow(previous_output,output)
            mag=flow_difference.square().sum(1,keepdim=True).sqrt().sum()
            if not torch.isfinite(mag) or mag<=0: raise RuntimeError('official FloLPIPS undefined: zero/nonfinite motion weight')
            value=float(perceptual(previous_reference,previous_output,flow_difference,normalize=True).item())
            if not math.isfinite(value): raise RuntimeError('nonfinite FloLPIPS')
            transitions.append(dict(from_frame=index-1,to_frame=index,FloLPIPS=value))
        previous_reference,previous_output=reference,output
    reconstruction_hash=eco.sha256_bytes(''.join(decoded_hashes).encode())
    assert buffer.tell()==len(data) and len(decoded_hashes)==len(frames)
    assert before==core.compression_hash(i_model,p_model)==expected['compression_hash_after']
    if expected.get('reconstruction_sha256'): assert reconstruction_hash==expected['reconstruction_sha256']
    assert eco.sha256_bytes(data)==expected['bitstream_sha256']
    real=np.concatenate(refs); recon=np.concatenate(outs)
    assert real.shape==recon.shape==(len(frames),2048) and np.isfinite(real).all() and np.isfinite(recon).all()
    feature_path=Path(feature_path); feature_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=feature_path.with_suffix('.tmp')
    with open(temporary,'wb') as f: np.savez(f,real=real,reconstruction=recon)
    temporary.replace(feature_path)
    write(feature_path.with_suffix('.transitions.csv'),transitions)
    return dict(FloLPIPS=sum(r['FloLPIPS'] for r in transitions)/len(transitions),
                feature_path=str(feature_path),feature_sha256=sha(feature_path),num_transitions=len(transitions),
                metric_decode_pass=True,reconstruction_sha256=reconstruction_hash)
