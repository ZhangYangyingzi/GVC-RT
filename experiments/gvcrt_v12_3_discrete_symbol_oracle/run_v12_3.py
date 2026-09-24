#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,math,os,sys,traceback
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
V121=ROOT.parent/'gvcrt_v12_1_multicontent_budget_oracle';V122=ROOT.parent/'gvcrt_v12_2_same_bit_budget'
DBG=REPO/'expericent_generation_input/expericent_generator_aware_encoder_latent_rd_oracle_v12_debug';V11=REPO/'expericent_generation_input/expericent_generator_aware_latent_distortion_v11';V9=REPO/'expericent_generation_input/expericent_interface_causal_controls_v9'
for p in (REPO,V121,V9/'src',V11/'b2_latent_direction_magnitude_v11_7/src',DBG/'src'):sys.path.insert(0,str(p))
from common import sha256
from gvc_hooks import load_models
from run_debug import load_b2,decode_capture,unit
from run_v12_1 import read_frames,basic_metrics,init_metric_models,perceptual,snap,restore,INDEX_MAP
from src.layers.cuda_inference import round_and_to_int8,restore_y_2x_with_cat_after,restore_y_2x,add_and_multiply
from src.utils.stream_helper import write_sps,write_ip,read_header,read_sps_remaining,read_ip_remaining,NalType

CFG=json.loads((ROOT/'config.json').read_text());MAN=json.loads((ROOT/'manifest.json').read_text())
SPS={'sps_id':0,'height':1088,'width':1920,'ec_part':1,'use_ada_i':0}

def read_csv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else []
    with p.open('w',newline='') as f:
        if fields:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def tag(v,qp):return f"{v['dataset']}_{int(v['video_id']):02d}_qp{qp}"
def digest_bytes(b):return hashlib.sha256(b).hexdigest()
def tensor_hashes(parts):
    h=hashlib.sha256()
    for x in parts:h.update(x.detach().to(torch.int8).cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def model_hash(*models):
    h=hashlib.sha256()
    for m in models:
        for n,p in m.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def parse_stream(path):
    payloads=[];cumulative=[];spss={};p=Path(path)
    with p.open('rb') as f:
        while f.tell()<p.stat().st_size:
            h=read_header(f)
            if h['nal_type']==NalType.NAL_SPS:spss[h['sps_id']]=read_sps_remaining(f,h['sps_id']);continue
            qp,payload=read_ip_remaining(f);payloads.append((h['nal_type']==NalType.NAL_I,qp,payload));cumulative.append(f.tell())
    if len(payloads)!=64 or cumulative[-1]!=p.stat().st_size:raise RuntimeError('baseline stream parse mismatch')
    return payloads,cumulative

def prepare(m,x,qp):
    qf=m.q_scale_feature[qp:qp+1];qe=m.q_scale_enc[qp:qp+1];qd=m.q_scale_dec[qp:qp+1];qr=m.q_scale_recon[qp:qp+1]
    feature=m.apply_feature_adaptor();ctx,ct=m.feature_extractor(feature,qf);y=m.enc(x,ctx,qe).detach()
    z=m.hyper_enc(m.pad_for_y(y));zh,zw=round_and_to_int8(z);params=m.res_prior_param_decoder(zh,ct);w0,w1,s0,s1,_=m.compress_prior_2x(y,params,m.y_spatial_prior)
    return ctx.detach(),ct.detach(),qd.detach(),qr.detach(),zh.detach().clone(),w0.detach().clone(),w1.detach().clone()

def reconstruct(m,z,w0,w1,ctx,ct,qd,qr):
    params=m.res_prior_param_decoder(z,ct);qdec,sc,mu=m.separate_prior_for_video_decoding(params);B,C,H,W=mu.shape;m0,m1=m.get_mask_2x(B,C,H,W,mu.dtype,mu.device)
    sw0=m.single_part_for_writing_2x(sc*m0);yh0,cat=restore_y_2x_with_cat_after(w0,mu,m0,params);sc1,mu1=m.y_spatial_prior(cat).chunk(2,1);sw1=m.single_part_for_writing_2x(sc1*m1);yh1=restore_y_2x(w1,mu1,m1);yhat=add_and_multiply(yh0,yh1,qdec);feat=m.dec(yhat,ctx,qd);rgb=m.recon_generation_net(feat,qr)
    return rgb,feat,sw0,sw1

def encode_symbols(m,z,w0,w1,ctx,ct,qd,qr,qp):
    with torch.no_grad():
        rgb,feat,sw0,sw1=reconstruct(m,z,w0,w1,ctx,ct,qd,qr);m.entropy_coder.reset();m.bit_estimator_z.encode_z(z.to(torch.int8),qp);m.gaussian_encoder.encode_y(w0,sw0);m.gaussian_encoder.encode_y(w1,sw1);m.entropy_coder.flush();payload=m.entropy_coder.get_encoded_stream()
    return payload,rgb,feat,sw0,sw1

def packet_size(payload,qp):
    b=io.BytesIO();write_ip(b,False,0,qp,payload);return len(b.getvalue())

def metrics_all(out,gt,metric_models):
    mse,psnr,ssim=basic_metrics(out,gt);lp,di=perceptual(out,gt,metric_models);return {'MSE':mse,'PSNR':psnr,'SSIM':ssim,'LPIPS':lp,'DISTS':di}

def apply_edits(symbols,edits):
    z,w0,w1=[x.clone() for x in symbols];d={'z':z,'w0':w0,'w1':w1}
    for fam,idx,delta in edits:d[fam].view(-1)[idx]+=delta
    return z,w0,w1

def evaluate_candidate(enc,dec,symbols,edits,ctx,ct,qd,qr,qp,gt,metric_models):
    z,w0,w1=apply_edits(symbols,edits);payload,_,efeat,_,_=encode_symbols(enc,z,w0,w1,ctx,ct,qd,qr,qp);state=snap(dec)
    try:
        out,cap=decode_capture(dec,payload,SPS,qp);dfeat=dec.dpb[0].feature;state_err=float((efeat-dfeat).abs().max());finite=bool(torch.isfinite(out).all() and torch.isfinite(dfeat).all());met=metrics_all(out,gt,metric_models);sym=bool(torch.equal(z,cap['z']) and torch.equal(w0,cap['w0']) and torch.equal(w1,cap['w1']))
    finally:restore(dec,state)
    return {'payload':payload,'metrics':met,'symbol_hash':tensor_hashes((z,w0,w1)),'decode_status':'PASS' if sym else 'FAIL','finite_status':'PASS' if finite else 'FAIL','state_max_abs':state_err,'symbols':(z,w0,w1)}

def sensitivity(m,symbols,ctx,ct,qd,qr,gt):
    z,w0,w1=[x.detach().clone().requires_grad_(True) for x in symbols];rgb,_,s0,s1=reconstruct(m,z,w0,w1,ctx,ct,qd,qr);loss=(unit(rgb)-unit(gt)).square().mean();g=torch.autograd.grad(loss,(z,w0,w1));return float(loss),g,(torch.ones_like(z,dtype=torch.bool),s0>.12,s1>.12)

def proposals(symbols,grads,active):
    out=[]
    for fam,x,g,a in zip(('z','w0','w1'),symbols,grads,active):
        valid=a.reshape(-1);idx=torch.nonzero(valid,as_tuple=False).reshape(-1);k=min(CFG['top_k'][fam],idx.numel())
        if k:
            score=g.detach().abs().reshape(-1)[idx];chosen=idx[torch.topk(score,k,largest=True,sorted=True).indices]
            for ii in chosen.tolist():
                old=int(x.reshape(-1)[ii].item())
                for delta in (-1,1):
                    if CFG['symbol_min']<=old+delta<=CFG['symbol_max']:out.append((fam,ii,delta,float(g.reshape(-1)[ii].abs())))
    return out

def candidate_row(v,qp,fi,rd,kind,edits,scores,current,ev,prefix,basecum):
    met=ev['metrics'];cm=current['metrics'];candidate_cum=prefix+packet_size(ev['payload'],rd['actual_qp']);e1=edits[0];e2=edits[1] if len(edits)>1 else ('','', '')
    dr=len(ev['payload'])-len(current['payload']);dd=met['MSE']-cm['MSE'];ctype='A' if dr<=0 and dd<0 else 'B' if dr<0 and dd>=0 else 'C' if dr>0 and dd<0 else 'D'
    return {'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'frame':fi,'round':rd['round'],'candidate_type':ctype,'single_or_pair':kind,'symbol_family_1':e1[0],'symbol_index_1':e1[1],'delta_1':e1[2],'symbol_family_2':e2[0],'symbol_index_2':e2[1],'delta_2':e2[2],'current_payload_bytes':len(current['payload']),'candidate_payload_bytes':len(ev['payload']),'delta_payload_bytes':dr,'current_cumulative_bytes':prefix+packet_size(current['payload'],rd['actual_qp']),'candidate_cumulative_bytes':candidate_cum,'baseline_cumulative_bytes':basecum,'budget_bank_after_candidate':basecum-candidate_cum,'current_RGB_MSE':cm['MSE'],'candidate_RGB_MSE':met['MSE'],'delta_RGB_MSE':dd,'current_PSNR':cm['PSNR'],'candidate_PSNR':met['PSNR'],'delta_PSNR':met['PSNR']-cm['PSNR'],'LPIPS':met['LPIPS'],'DISTS':met['DISTS'],'gradient_score_1':scores[0],'gradient_score_2':scores[1] if len(scores)>1 else '','bitstream_hash':digest_bytes(ev['payload']),'symbol_hash':ev['symbol_hash'],'decode_status':ev['decode_status'],'finite_status':ev['finite_status'],'accepted':False,'state_max_abs':ev['state_max_abs'],'_edits':edits}

def independent_decode(path,expected_symbols,expected_recon,device):
    im,_=load_models(device);dec,_=load_b2(device);dec.clear_dpb();dec.set_curr_poc(0);spss={};fi=0;max_recon=0.;symbols_ok=True;finite=True
    with Path(path).open('rb') as f:
        while f.tell()<Path(path).stat().st_size:
            h=read_header(f)
            if h['nal_type']==NalType.NAL_SPS:spss[h['sps_id']]=read_sps_remaining(f,h['sps_id']);continue
            qp,payload=read_ip_remaining(f);sps=spss[h['sps_id']]
            if h['nal_type']==NalType.NAL_I:out=im.decompress(payload,sps,qp)['x_hat'];dec.add_ref_frame(None,out)
            else:
                out,cap=decode_capture(dec,payload,sps,qp);z,w0,w1=expected_symbols[fi];symbols_ok&=bool(torch.equal(z.to(device),cap['z']) and torch.equal(w0.to(device),cap['w0']) and torch.equal(w1.to(device),cap['w1']))
            finite&=bool(torch.isfinite(out).all());max_recon=max(max_recon,0. if tensor_hash(out)==expected_recon[fi] else float('inf'));fi+=1
    return fi,symbols_ok,finite,max_recon

def run_cell(v,requested_qp,device):
    t=tag(v,requested_qp);done=ROOT/'parts'/f'{t}_done.json'
    if done.exists() and json.loads(done.read_text()).get('status')=='PASS':print('reuse',t,flush=True);return
    base_path=V121/'bitstreams'/f'{t}_baseline.bin';base_row=read_csv(V121/'parts'/f'{t}_baseline.csv')[0];payloads,base_cum=parse_stream(base_path)
    if sha256(base_path)!=next(r['actual_sha256'] for r in read_csv(V122/'sanity/identity_audit.csv') if r['kind']=='reused_baseline_bitstream' and r['dataset']==v['dataset'] and int(r['video_id'])==int(v['video_id']) and int(r['qp'])==requested_qp):raise RuntimeError('baseline sha mismatch')
    frames=read_frames(v,device);metric_models=init_metric_models(device);im,_=load_models(device);enc,_=load_b2(device);dec,_=load_b2(device);enc.clear_dpb();dec.clear_dpb();enc.set_curr_poc(0);dec.set_curr_poc(0);before=model_hash(im,enc,dec)
    stream=io.BytesIO();write_sps(stream,SPS);candidate_rows=[];accepted=[];frame_rows=[];sync=[];coverage={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':requested_qp,'total_active_z_symbols':0,'total_active_w0_symbols':0,'total_active_w1_symbols':0,'num_z_symbols_proposed':0,'num_w0_symbols_proposed':0,'num_w1_symbols_proposed':0,'num_single_exact_evaluations':0,'num_pair_exact_evaluations':0,'num_rate_saving_candidates':0,'num_quality_improving_candidates':0,'num_joint_rate_nonincrease_quality_improve_candidates':0,'num_committed_moves':0}
    expected_symbols=[None]*64;expected_recon=['']*64;changed={'z':set(),'w0':set(),'w1':set()};single_count=pair_count=0
    is_i,iqp,ipayload=payloads[0];out=im.decompress(ipayload,SPS,iqp)['x_hat'];enc.add_ref_frame(None,out);dec.add_ref_frame(None,out);write_ip(stream,True,0,iqp,ipayload);met=metrics_all(out,frames[0],metric_models);frame_rows.append(dict(dataset=v['dataset'],video=v['name'],video_id=v['video_id'],qp=requested_qp,frame=0,frame_type='I',actual_qp=iqp,actual_bytes=len(ipayload),**met));expected_recon[0]=tensor_hash(out)
    for fi in range(1,64):
        aq=enc.shift_qp(requested_qp,INDEX_MAP[fi%8]);ctx,ct,qd,qr,z,w0,w1=prepare(enc,frames[fi],aq);symbols=(z,w0,w1);rounds=0
        for ri in range(CFG['maximum_rounds_per_p_frame']):
            rounds=ri+1;_,grads,active=sensitivity(enc,symbols,ctx,ct,qd,qr,frames[fi]);
            if ri==0:
                coverage['total_active_z_symbols']+=int(active[0].sum());coverage['total_active_w0_symbols']+=int(active[1].sum());coverage['total_active_w1_symbols']+=int(active[2].sum())
            prop=proposals(symbols,grads,active)
            for fam in ('z','w0','w1'):coverage[f'num_{fam}_symbols_proposed']+=sum(p[0]==fam for p in prop)
            current=evaluate_candidate(enc,dec,symbols,[],ctx,ct,qd,qr,aq,frames[fi],metric_models);prefix=len(stream.getvalue());rd={'round':ri,'actual_qp':aq};singles=[]
            for fam,idx,delta,score in prop:
                ev=evaluate_candidate(enc,dec,symbols,[(fam,idx,delta)],ctx,ct,qd,qr,aq,frames[fi],metric_models);row=candidate_row(v,requested_qp,fi,rd,'single',[(fam,idx,delta)],[score],current,ev,prefix,base_cum[fi]);candidate_rows.append(row);singles.append(row);coverage['num_single_exact_evaluations']+=1
                coverage['num_rate_saving_candidates']+=row['delta_payload_bytes']<0;coverage['num_quality_improving_candidates']+=row['delta_RGB_MSE']<0;coverage['num_joint_rate_nonincrease_quality_improve_candidates']+=row['delta_payload_bytes']<=0 and row['delta_RGB_MSE']<0
            donors=sorted([r for r in singles if r['delta_payload_bytes']<0 and r['decode_status']=='PASS' and r['finite_status']=='PASS'],key=lambda r:r['delta_RGB_MSE']/(-r['delta_payload_bytes']))[:CFG['pair_top_k_donors']]
            receivers=sorted([r for r in singles if r['delta_payload_bytes']>0 and r['delta_RGB_MSE']<0 and r['decode_status']=='PASS' and r['finite_status']=='PASS'],key=lambda r:-r['delta_RGB_MSE']/r['delta_payload_bytes'],reverse=True)[:CFG['pair_top_k_receivers']]
            pairs=[]
            for d in donors:
                for rec in receivers:
                    e1=d['_edits'][0];e2=rec['_edits'][0]
                    if e1[:2]==e2[:2]:continue
                    ev=evaluate_candidate(enc,dec,symbols,[e1,e2],ctx,ct,qd,qr,aq,frames[fi],metric_models);row=candidate_row(v,requested_qp,fi,rd,'pair',[e1,e2],[d['gradient_score_1'],rec['gradient_score_1']],current,ev,prefix,base_cum[fi]);candidate_rows.append(row);pairs.append(row);coverage['num_pair_exact_evaluations']+=1
                    coverage['num_rate_saving_candidates']+=row['delta_payload_bytes']<0;coverage['num_quality_improving_candidates']+=row['delta_RGB_MSE']<0;coverage['num_joint_rate_nonincrease_quality_improve_candidates']+=row['delta_payload_bytes']<=0 and row['delta_RGB_MSE']<0
            feasible=[r for r in singles+pairs if r['decode_status']=='PASS' and r['finite_status']=='PASS' and r['state_max_abs']==0 and r['candidate_cumulative_bytes']<=r['baseline_cumulative_bytes'] and r['candidate_RGB_MSE']<r['current_RGB_MSE']]
            if not feasible:break
            best=min(feasible,key=lambda r:r['candidate_RGB_MSE']);best['accepted']=True;old_symbols=symbols;symbols=apply_edits(symbols,best['_edits']);coverage['num_committed_moves']+=1
            vals=[]
            for fam,idx,delta in best['_edits']:
                oi=('z','w0','w1').index(fam);old=int(old_symbols[oi].view(-1)[idx]);new=old+delta;vals.append((fam,idx,old,new));changed[fam].add(idx)
            bm=current['metrics'];am={'MSE':best['candidate_RGB_MSE'],'PSNR':best['candidate_PSNR'],'LPIPS':best['LPIPS'],'DISTS':best['DISTS']};x1=vals[0];x2=vals[1] if len(vals)>1 else ('','','','')
            accepted.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':requested_qp,'frame':fi,'round':ri,'edit_type':best['single_or_pair'],'symbol_family_1':x1[0],'symbol_index_1':x1[1],'old_value_1':x1[2],'new_value_1':x1[3],'symbol_family_2':x2[0],'symbol_index_2':x2[1],'old_value_2':x2[2],'new_value_2':x2[3],'frame_bytes_before':best['current_payload_bytes'],'frame_bytes_after':best['candidate_payload_bytes'],'delta_frame_bytes':best['delta_payload_bytes'],'cumulative_bytes_after':best['candidate_cumulative_bytes'],'baseline_cumulative_bytes':best['baseline_cumulative_bytes'],'budget_bank':best['budget_bank_after_candidate'],'RGB_MSE_before':bm['MSE'],'RGB_MSE_after':am['MSE'],'PSNR_before':bm['PSNR'],'PSNR_after':am['PSNR'],'LPIPS_before':bm['LPIPS'],'LPIPS_after':am['LPIPS'],'DISTS_before':bm['DISTS'],'DISTS_after':am['DISTS']})
            if best['single_or_pair']=='single':single_count+=1
            else:pair_count+=1
        payload,_,efeat,_,_=encode_symbols(enc,*symbols,ctx,ct,qd,qr,aq)
        formal_cumulative_bytes=len(stream.getvalue())+packet_size(payload,aq)
        if formal_cumulative_bytes>base_cum[fi]:
            after_failure=model_hash(im,enc,dec)
            write_csv(ROOT/'parts'/f'{t}_candidate_exact_eval.partial.csv',[{k:v for k,v in r.items() if not k.startswith('_')} for r in candidate_rows])
            write_csv(ROOT/'parts'/f'{t}_accepted_edits.partial.csv',accepted)
            write_csv(ROOT/'parts'/f'{t}_frame_metrics.partial.csv',frame_rows)
            write_csv(ROOT/'parts'/f'{t}_state_sync.partial.csv',sync)
            write_csv(ROOT/'parts'/f'{t}_search_coverage.partial.csv',[coverage])
            failure={'status':'FAIL','failure_type':'CAUSAL_BUDGET_VIOLATION','dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':requested_qp,'failure_frame':fi,'selected_cumulative_bytes':formal_cumulative_bytes,'baseline_cumulative_bytes':base_cum[fi],'budget_gap_bytes':formal_cumulative_bytes-base_cum[fi],'frames_committed':len(frame_rows),'model_hash_before':before,'model_hash_after':after_failure,'model_unchanged':before==after_failure,'decode_status_for_committed_frames':'PASS' if all(r['status']=='PASS' for r in sync) else 'FAIL','finite_status_for_committed_frames':'PASS' if all(r['finite'] for r in sync) else 'FAIL'}
            (ROOT/'parts'/f'{t}_failure_audit.json').write_text(json.dumps(failure,indent=2)+'\n')
            raise RuntimeError(f'causal budget violation frame={fi} selected_cumulative={formal_cumulative_bytes} baseline_cumulative={base_cum[fi]}')
        out,cap=decode_capture(dec,payload,SPS,aq);symok=bool(torch.equal(symbols[0],cap['z']) and torch.equal(symbols[1],cap['w0']) and torch.equal(symbols[2],cap['w1']));enc.add_ref_frame(efeat,None);state_err=float((enc.dpb[0].feature-dec.dpb[0].feature).abs().max());finite=bool(torch.isfinite(out).all());write_ip(stream,False,0,aq,payload);met=metrics_all(out,frames[fi],metric_models);frame_rows.append(dict(dataset=v['dataset'],video=v['name'],video_id=v['video_id'],qp=requested_qp,frame=fi,frame_type='P',actual_qp=aq,actual_bytes=len(payload),**met));sync.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':requested_qp,'frame':fi,'symbol_equal':symok,'state_max_abs':state_err,'finite':finite,'status':'PASS' if symok and state_err==0 and finite else 'FAIL'});expected_symbols[fi]=tuple(x.detach().cpu() for x in symbols);expected_recon[fi]=tensor_hash(out)
        if fi%4==0:
            write_csv(ROOT/'parts'/f'{t}_candidate_exact_eval.partial.csv',[{k:v for k,v in r.items() if not k.startswith('_')} for r in candidate_rows]);write_csv(ROOT/'parts'/f'{t}_accepted_edits.partial.csv',accepted);print(t,'frame',fi,'/63 candidates',len(candidate_rows),'accepted',len(accepted),flush=True)
    bs=stream.getvalue();path=ROOT/'bitstreams'/f'{t}_discrete.bin';path.write_bytes(bs);frames_dec,sym_ind,finite_ind,recon_diff=independent_decode(path,expected_symbols,expected_recon,device);after=model_hash(im,enc,dec)
    basebytes=int(base_row['actual_bytes']);discbytes=len(bs);agg={k:sum(float(r[k]) for r in frame_rows)/64 for k in ('PSNR','SSIM','LPIPS','DISTS')};final={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':requested_qp,'baseline_bytes':basebytes,'discrete_bytes':discbytes,'rate_ratio':discbytes/basebytes,'rate_gap_bytes':discbytes-basebytes,'rate_gap_percent':100*(discbytes/basebytes-1),'baseline_PSNR':base_row['PSNR'],'discrete_PSNR':agg['PSNR'],'delta_PSNR':agg['PSNR']-float(base_row['PSNR']),'baseline_SSIM':base_row['MS_SSIM'],'discrete_SSIM':agg['SSIM'],'delta_SSIM':agg['SSIM']-float(base_row['MS_SSIM']),'baseline_LPIPS':base_row['LPIPS'],'discrete_LPIPS':agg['LPIPS'],'delta_LPIPS':agg['LPIPS']-float(base_row['LPIPS']),'baseline_DISTS':base_row['DISTS'],'discrete_DISTS':agg['DISTS'],'delta_DISTS':agg['DISTS']-float(base_row['DISTS']),'num_accepted_single_edits':single_count,'num_accepted_pair_edits':pair_count,'num_changed_z_symbols':len(changed['z']),'num_changed_w0_symbols':len(changed['w0']),'num_changed_w1_symbols':len(changed['w1']),'bitstream_path':str(path),'bitstream_sha256':sha256(path),'decode_status':'PASS' if frames_dec==64 and sym_ind and recon_diff==0 else 'FAIL','state_sync_status':'PASS' if all(r['status']=='PASS' for r in sync) else 'FAIL','finite_status':'PASS' if finite_ind and all(r['finite'] for r in sync) else 'FAIL'}
    coverage['num_committed_moves']=len(accepted);audit={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':requested_qp,'bitstream_path':str(path),'bitstream_sha256':sha256(path),'frames_decoded':frames_dec,'all_entropy_symbols_recovered_exactly':sym_ind,'reconstruction_max_abs_diff':recon_diff,'decode_status':final['decode_status'],'state_sync_status':final['state_sync_status'],'finite_status':final['finite_status']}
    ok=discbytes<=basebytes and all(final[k]=='PASS' for k in ('decode_status','state_sync_status','finite_status')) and before==after
    write_csv(ROOT/'parts'/f'{t}_candidate_exact_eval.csv',[{k:v for k,v in r.items() if not k.startswith('_')} for r in candidate_rows]);write_csv(ROOT/'parts'/f'{t}_accepted_edits.csv',accepted);write_csv(ROOT/'parts'/f'{t}_frame_metrics.csv',frame_rows);write_csv(ROOT/'parts'/f'{t}_state_sync.csv',sync);write_csv(ROOT/'parts'/f'{t}_search_coverage.csv',[coverage]);write_csv(ROOT/'parts'/f'{t}_final_stream.csv',[final]);write_csv(ROOT/'parts'/f'{t}_bitstream_audit.csv',[audit])
    done.write_text(json.dumps({'status':'PASS' if ok else 'FAIL','tag':t,'baseline_bytes':basebytes,'discrete_bytes':discbytes,'model_hash_before':before,'model_hash_after':after,'model_unchanged':before==after,'budget_pass':discbytes<=basebytes,'final':final},indent=2)+'\n')
    if not ok:raise RuntimeError(f'cell integrity failure {t}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--cells',required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);device=torch.device('cuda:0');lookup={(v['dataset'],int(v['video_id'])):v for v in MAN['videos']}
    for spec in a.cells.split(','):
        ds,vid,qp=spec.split(':');t=f'{ds}_{int(vid):02d}_qp{qp}'
        try:run_cell(lookup[(ds,int(vid))],int(qp),device)
        except Exception as e:
            (ROOT/'parts'/f'{t}_FAILED.json').write_text(json.dumps({'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()},indent=2)+'\n');print('FAILED',t,repr(e),flush=True)
if __name__=='__main__':main()
