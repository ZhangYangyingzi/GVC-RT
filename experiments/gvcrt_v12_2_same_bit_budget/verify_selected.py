#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,math,os,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];V121=ROOT.parent/'gvcrt_v12_1_multicontent_budget_oracle'
DBG=REPO/'expericent_generation_input/expericent_generator_aware_encoder_latent_rd_oracle_v12_debug';V11=REPO/'expericent_generation_input/expericent_generator_aware_latent_distortion_v11';V9=REPO/'expericent_generation_input/expericent_interface_causal_controls_v9'
for p in (REPO,V121,V9/'src',V11/'b2_latent_direction_magnitude_v11_7/src',DBG/'src'):sys.path.insert(0,str(p))
from gvc_hooks import load_models
from run_debug import load_b2
from common import sha256
from src.utils.stream_helper import read_header,read_sps_remaining,read_ip_remaining,NalType

def read_csv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r));
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def t_hash(x):return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()

def verify(row,device):
    path=Path(row['bitstream_path']);im,_=load_models(device);d1,_=load_b2(device);d2,_=load_b2(device)
    for d in (d1,d2):d.clear_dpb();d.set_curr_poc(0)
    spss={};frames=0;max_rgb=0.;max_feature=0.;finite=True;recon=[]
    with path.open('rb') as fd:
        while fd.tell()<path.stat().st_size:
            h=read_header(fd)
            if h['nal_type']==NalType.NAL_SPS:spss[h['sps_id']]=read_sps_remaining(fd,h['sps_id']);continue
            qp,payload=read_ip_remaining(fd);sps=spss[h['sps_id']]
            if h['nal_type']==NalType.NAL_I:
                x1=im.decompress(payload,sps,qp)['x_hat'];x2=im.decompress(payload,sps,qp)['x_hat'];d1.add_ref_frame(None,x1);d2.add_ref_frame(None,x2)
            else:
                x1=d1.decompress(payload,sps,qp)['x_hat'];x2=d2.decompress(payload,sps,qp)['x_hat'];max_feature=max(max_feature,float((d1.dpb[0].feature-d2.dpb[0].feature).abs().max()))
            max_rgb=max(max_rgb,float((x1-x2).abs().max()));finite=finite and bool(torch.isfinite(x1).all()) and bool(torch.isfinite(x2).all());recon.append(t_hash(x1));frames+=1
    original_sync=(V121/'parts'/f"{row['dataset']}_{int(row['video_id']):02d}_qp{row['qp']}_sync.csv")
    new_sync=ROOT/'parts'/f"{row['dataset']}_{int(row['video_id']):02d}_qp{row['qp']}_{row['branch']}_beta{float(row['selected_beta']):.12g}_state_sync.csv"
    sync_rows=read_csv(new_sync) if new_sync.exists() else [r for r in read_csv(original_sync) if r['branch']==row['branch'] and abs(float(r['beta'])-float(row['selected_beta']))<1e-10]
    symbol_equal=bool(sync_rows) and all(r.get('symbol_equal') in ('True','true','1') for r in sync_rows);state_ok=bool(sync_rows) and all(r.get('status')=='PASS' for r in sync_rows)
    ok=frames==64 and max_rgb==0 and max_feature==0 and finite and symbol_equal and state_ok and path.stat().st_size==int(row['selected_bytes']) and sha256(path)==row['bitstream_sha256']
    return {'dataset':row['dataset'],'video':row['video'],'video_id':row['video_id'],'qp':row['qp'],'branch':row['branch'],'selected_beta':row['selected_beta'],'bitstream_path':str(path),'actual_bytes':path.stat().st_size,'bitstream_sha256':sha256(path),'frames_decoded':frames,'symbol_equal':symbol_equal,'decoded_latent_max_abs_diff':max_feature,'reconstruction_max_abs_diff':max_rgb,'reconstruction_hash_sha256':hashlib.sha256(''.join(recon).encode()).hexdigest(),'decode_status':'PASS' if ok else 'FAIL','state_sync_status':'PASS' if state_ok and max_feature==0 else 'FAIL','finite_status':'PASS' if finite else 'FAIL'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);device=torch.device('cuda:0');rows=[r for r in read_csv(ROOT/'raw/same_bit_selected.csv') if r['bitstream_path']];out=[]
    for i,r in enumerate(rows):out.append(verify(r,device));print(f'verified {i+1}/{len(rows)} {r["dataset"]}:{r["video_id"]}:qp{r["qp"]}:{r["branch"]} {out[-1]["decode_status"]}',flush=True)
    write_csv(ROOT/'raw/selected_bitstream_audit.csv',out)
if __name__=='__main__':main()
