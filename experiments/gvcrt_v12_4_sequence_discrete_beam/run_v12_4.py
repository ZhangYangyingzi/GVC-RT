#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,math,os,sys,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
V121=ROOT.parent/'gvcrt_v12_1_multicontent_budget_oracle';V122=ROOT.parent/'gvcrt_v12_2_same_bit_budget';V123=ROOT.parent/'gvcrt_v12_3_discrete_symbol_oracle'
DBG=REPO/'expericent_generation_input/expericent_generator_aware_encoder_latent_rd_oracle_v12_debug';V11=REPO/'expericent_generation_input/expericent_generator_aware_latent_distortion_v11';V9=REPO/'expericent_generation_input/expericent_interface_causal_controls_v9'
for p in (REPO,V121,V9/'src',V11/'b2_latent_direction_magnitude_v11_7/src',DBG/'src'):sys.path.insert(0,str(p))
from common import sha256
from gvc_hooks import load_models
from run_debug import load_b2,decode_capture,unit
from run_v12_1 import read_frames,basic_metrics,init_metric_models,perceptual,INDEX_MAP
from src.layers.cuda_inference import round_and_to_int8,restore_y_2x_with_cat_after,restore_y_2x,add_and_multiply
from src.utils.stream_helper import write_sps,write_ip,read_header,read_sps_remaining,read_ip_remaining,NalType
from src.models.video_model_gvcrt import RefFrame
CFG=json.loads((ROOT/'config.json').read_text());MAN=json.loads((ROOT/'manifest.json').read_text());SPS={'sps_id':0,'height':1088,'width':1920,'ec_part':1,'use_ada_i':0}

def read_csv(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows,fields=None):
 rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else []
 with p.open('w',newline='') as f:
  if fields:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def tag(v,q):return f"{v['dataset']}_{int(v['video_id']):02d}_qp{q}"
def hbytes(b):return hashlib.sha256(b).hexdigest()
def thash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def shash(xs):
 h=hashlib.sha256()
 for x in xs:h.update(x.detach().to(torch.int8).cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def mhash(*models):
 h=hashlib.sha256()
 for m in models:
  for n,p in m.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def state_hash(s):
 h=hashlib.sha256();h.update(str(s[0]).encode())
 for poc,fr,ft in s[1]:
  h.update(str(poc).encode())
  for x in (fr,ft):
   if x is not None:h.update(x.contiguous().numpy().tobytes())
 return h.hexdigest()
def snap_cpu(m):return (m.curr_poc,[(r.poc,None if r.frame is None else r.frame.detach().cpu().clone(),None if r.feature is None else r.feature.detach().cpu().clone()) for r in m.dpb])
def restore_device(m,s,device):
 m.curr_poc=s[0];m.dpb=[]
 for poc,fr,ft in s[1]:
  r=RefFrame();r.poc=poc;r.frame=None if fr is None else fr.to(device);r.feature=None if ft is None else ft.to(device);m.dpb.append(r)
def parse_stream(path):
 payloads=[];cum=[];p=Path(path);spss={}
 with p.open('rb') as f:
  while f.tell()<p.stat().st_size:
   h=read_header(f)
   if h['nal_type']==NalType.NAL_SPS:spss[h['sps_id']]=read_sps_remaining(f,h['sps_id']);continue
   q,b=read_ip_remaining(f);payloads.append((h['nal_type']==NalType.NAL_I,q,b));cum.append(f.tell())
 if len(payloads)!=64 or cum[-1]!=p.stat().st_size:raise RuntimeError('baseline parse mismatch')
 return payloads,cum
def packet(payload,qp,is_i=False):
 b=io.BytesIO();write_ip(b,is_i,0,qp,payload);return b.getvalue()
def prepare(m,x,qp):
 qf=m.q_scale_feature[qp:qp+1];qe=m.q_scale_enc[qp:qp+1];qd=m.q_scale_dec[qp:qp+1];qr=m.q_scale_recon[qp:qp+1]
 feature=m.apply_feature_adaptor();ctx,ct=m.feature_extractor(feature,qf);y=m.enc(x,ctx,qe).detach();z=m.hyper_enc(m.pad_for_y(y));zh,_=round_and_to_int8(z);params=m.res_prior_param_decoder(zh,ct);w0,w1,_,_,_=m.compress_prior_2x(y,params,m.y_spatial_prior)
 return ctx.detach(),ct.detach(),qd.detach(),qr.detach(),(zh.detach().clone(),w0.detach().clone(),w1.detach().clone())
def reconstruct(m,z,w0,w1,ctx,ct,qd,qr):
 params=m.res_prior_param_decoder(z,ct);qdec,sc,mu=m.separate_prior_for_video_decoding(params);B,C,H,W=mu.shape;m0,m1=m.get_mask_2x(B,C,H,W,mu.dtype,mu.device);sw0=m.single_part_for_writing_2x(sc*m0);yh0,cat=restore_y_2x_with_cat_after(w0,mu,m0,params);sc1,mu1=m.y_spatial_prior(cat).chunk(2,1);sw1=m.single_part_for_writing_2x(sc1*m1);yh1=restore_y_2x(w1,mu1,m1);yhat=add_and_multiply(yh0,yh1,qdec);feat=m.dec(yhat,ctx,qd);rgb=m.recon_generation_net(feat,qr)
 return rgb,feat,sw0,sw1
def encode_checked(m,symbols,ctx,ct,qd,qr,qp):
 z,w0,w1=symbols
 if any(int(x.min())<CFG['symbol_min'] or int(x.max())>CFG['symbol_max'] for x in symbols):return {'valid':False,'reason':'SYMBOL_RANGE'}
 with torch.no_grad():
  rgb,feat,sw0,sw1=reconstruct(m,z,w0,w1,ctx,ct,qd,qr);thr=m.gaussian_encoder.force_zero_thres
  if thr is not None:
   if bool(((sw0<=thr)&(w0!=0)).any()) or bool(((sw1<=thr)&(w1!=0)).any()):return {'valid':False,'reason':'INVALID_INACTIVE_NONZERO'}
  m.entropy_coder.reset();m.bit_estimator_z.encode_z(z.to(torch.int8),qp);m.gaussian_encoder.encode_y(w0,sw0);m.gaussian_encoder.encode_y(w1,sw1);m.entropy_coder.flush();payload=m.entropy_coder.get_encoded_stream()
 return {'valid':True,'reason':'','payload':payload,'rgb_encoder':rgb,'feature':feat}
def metrics(out,gt,models):
 mse,psnr,ssim=basic_metrics(out,gt);lp,di=perceptual(out,gt,models);return {'MSE':mse,'PSNR':psnr,'SSIM':ssim,'LPIPS':lp,'DISTS':di}
def apply_edits(symbols,edits):
 out=[x.clone() for x in symbols];d={'z':out[0],'w0':out[1],'w1':out[2]}
 for fam,idx,delta in edits:d[fam].view(-1)[idx]+=delta
 return tuple(out)
def evaluate(enc,dec,symbols,edits,ctx,ct,qd,qr,qp,gt,models):
 sy=apply_edits(symbols,edits);e=encode_checked(enc,sy,ctx,ct,qd,qr,qp)
 if not e['valid']:return {'valid':False,'reason':e['reason'],'symbols':sy}
 ds=snap_cpu(dec)
 try:
  out,cap=decode_capture(dec,e['payload'],SPS,qp);dfeat=dec.dpb[0].feature;equal=bool(torch.equal(sy[0],cap['z']) and torch.equal(sy[1],cap['w0']) and torch.equal(sy[2],cap['w1']));serr=float((e['feature']-dfeat).abs().max());finite=bool(torch.isfinite(out).all() and torch.isfinite(dfeat).all());met=metrics(out,gt,models)
 finally:restore_device(dec,ds,sy[0].device)
 return {'valid':equal and finite and serr==0,'reason':'' if equal and finite and serr==0 else 'ENTROPY_DECODE_MISMATCH','symbols':sy,'payload':e['payload'],'feature':e['feature'],'out_hash':thash(out),'metrics':met,'decode_status':'PASS' if equal else 'FAIL','finite_status':'PASS' if finite else 'FAIL','state_max_abs':serr,'symbol_hash':shash(sy)}
def sensitivity(m,symbols,ctx,ct,qd,qr,gt):
 xs=[x.detach().clone().requires_grad_(True) for x in symbols];rgb,_,s0,s1=reconstruct(m,*xs,ctx,ct,qd,qr);loss=(unit(rgb)-unit(gt)).square().mean();g=torch.autograd.grad(loss,xs);thr=m.gaussian_encoder.force_zero_thres;active=(torch.ones_like(xs[0],dtype=torch.bool),torch.ones_like(s0,dtype=torch.bool) if thr is None else s0>thr,torch.ones_like(s1,dtype=torch.bool) if thr is None else s1>thr);return g,active
def ranked_proposals(symbols,grads,active):
 fams=('z','w0','w1');pool=[]
 for fam,x,g,a in zip(fams,symbols,grads,active):
  ids=torch.nonzero(a.reshape(-1),as_tuple=False).reshape(-1);vals=x.reshape(-1);gv=g.detach().reshape(-1)
  if ids.numel()==0:continue
  ks=min(12,ids.numel());top=ids[torch.topk(gv[ids].abs(),ks).indices]
  for ii in top.tolist():
   old=int(vals[ii]);delta=-1 if float(gv[ii])>0 else 1
   if CFG['symbol_min']<=old+delta<=CFG['symbol_max']:pool.append((fam,ii,delta,float(gv[ii].abs()),'SENSITIVITY'))
  nz=ids[vals[ids]!=0]
  if nz.numel():
   kr=min(12,nz.numel());topr=nz[torch.topk(vals[nz].abs().float(),kr).indices]
   for ii in topr.tolist():
    old=int(vals[ii]);delta=-1 if old>0 else 1;pool.append((fam,ii,delta,float(abs(old)),'RATE_PROXY'))
  for ii in top[:4].tolist():
   old=int(vals[ii]);delta=1 if float(gv[ii])>0 else -1
   if CFG['symbol_min']<=old+delta<=CFG['symbol_max']:pool.append((fam,ii,delta,float(gv[ii].abs()),'MIXED'))
 unique=[];seen=set()
 for x in sorted(pool,key=lambda z:z[3],reverse=True):
  k=x[:3]
  if k not in seen:seen.add(k);unique.append(x)
 selected=[]
 for kind,n in [('SENSITIVITY',8),('RATE_PROXY',8),('MIXED',8)]:selected += [x for x in unique if x[4]==kind and x not in selected][:n]
 for x in unique:
  if len(selected)>=CFG['max_single_exact_per_state']:break
  if x not in selected:selected.append(x)
 return selected[:CFG['max_single_exact_per_state']]
def choose_actions(actions):
 no=next(a for a in actions if a['action_type']=='NO_EDIT');valid=[a for a in actions if a['valid'] and a is not no];out=[no]
 cats=[]
 if valid:cats.append(min(valid,key=lambda a:a['metrics']['MSE']))
 if valid:cats.append(min(valid,key=lambda a:a['packet_bytes']))
 imp=[a for a in valid if a['delta_mse']<0];free=[a for a in imp if a['delta_bytes']<=0];pairs=[a for a in valid if a['action_type']=='PAIR']
 if imp:cats.append(max(imp,key=lambda a:(-a['delta_mse'])/(abs(a['delta_bytes'])+1)))
 if free:cats.append(min(free,key=lambda a:a['metrics']['MSE']))
 if pairs:cats.append(min(pairs,key=lambda a:a['metrics']['MSE']))
 for a in cats:
  if all(a is not x for x in out) and len(out)<CFG['max_successors_per_state']:out.append(a)
 for a in sorted(valid,key=lambda a:(a['metrics']['MSE'],a['packet_bytes'])):
  if all(a is not x for x in out) and len(out)<CFG['max_successors_per_state']:out.append(a)
 return out
def pareto(states):
 keep=[]
 for i,a in enumerate(states):
  dom=False
  for j,b in enumerate(states):
   if i!=j and b['cum_bytes']<=a['cum_bytes'] and b['cum_mse']<=a['cum_mse'] and (b['cum_bytes']<a['cum_bytes'] or b['cum_mse']<a['cum_mse']):dom=True;break
  a['pareto_dominated']=dom
  if not dom:keep.append(a)
 return keep
def select_beam(states,baseline_total):
 front=pareto(states);chosen=[]
 def bucket(s):
  g=(s['cum_bytes']-s['base_cum'])/baseline_total
  return 0 if g<=0 else 1 if g<=.0025 else 2 if g<=.005 else 3
 for b in range(4):
  z=[s for s in front if bucket(s)==b]
  if z:chosen.append(min(z,key=lambda s:s['cum_mse']))
 for s in sorted(front,key=lambda x:(x['cum_mse'],x['cum_bytes'])):
  if all(s is not x for x in chosen) and len(chosen)<CFG['beam_width']:chosen.append(s)
 return chosen[:CFG['beam_width']],states

def independent(path,expected_symbols,expected_recon,device):
 im,_=load_models(device);dec,_=load_b2(device);dec.clear_dpb();dec.set_curr_poc(0);fi=0;sym=True;finite=True;recon=True;spss={}
 with Path(path).open('rb') as f:
  while f.tell()<Path(path).stat().st_size:
   h=read_header(f)
   if h['nal_type']==NalType.NAL_SPS:spss[h['sps_id']]=read_sps_remaining(f,h['sps_id']);continue
   q,payload=read_ip_remaining(f);sps=spss[h['sps_id']]
   if h['nal_type']==NalType.NAL_I:out=im.decompress(payload,sps,q)['x_hat'];dec.add_ref_frame(None,out)
   else:
    out,cap=decode_capture(dec,payload,sps,q);z,w0,w1=expected_symbols[fi];sym &= bool(torch.equal(z.to(device),cap['z']) and torch.equal(w0.to(device),cap['w0']) and torch.equal(w1.to(device),cap['w1']))
   finite &= bool(torch.isfinite(out).all());recon &= thash(out)==expected_recon[fi];fi+=1
 return {'frames':fi,'symbol_equal':sym,'finite':finite,'reconstruction_equal':recon,'decode_status':'PASS' if fi==64 and sym and recon else 'FAIL'}
def action_row(v,qp,fi,parent,cid,action,current,basecum,baseline_total):
 edits=action['edits'];e1=edits[0] if edits else ('','','');e2=edits[1] if len(edits)>1 else ('','','');sy=parent.get('natural_symbols');old1=int(sy[('z','w0','w1').index(e1[0])].view(-1)[e1[1]]) if edits else '';old2=int(sy[('z','w0','w1').index(e2[0])].view(-1)[e2[1]]) if len(edits)>1 else '';m=action.get('metrics',{});before=current['metrics'];cum_after=parent['cum_bytes']+action.get('packet_bytes',0)
 return {'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'frame':fi,'trajectory_id':'','parent_trajectory_id':parent['tid'],'candidate_id':cid,'action_type':action['action_type'],'symbol_family_1':e1[0],'symbol_index_1':e1[1],'old_value_1':old1,'new_value_1':'' if not edits else old1+e1[2],'symbol_family_2':e2[0],'symbol_index_2':e2[1],'old_value_2':old2,'new_value_2':'' if len(edits)<2 else old2+e2[2],'frame_bytes_before':current['packet_bytes'],'frame_bytes_after':action.get('packet_bytes',''),'delta_frame_bytes':action.get('delta_bytes',''),'cumulative_bytes_before':parent['cum_bytes'],'cumulative_bytes_after':cum_after if action.get('valid') else '','baseline_cumulative_bytes':basecum,'rate_debt_bytes':cum_after-basecum if action.get('valid') else '','rate_debt_ratio':(cum_after-basecum)/baseline_total if action.get('valid') else '','frame_MSE_before':before['MSE'],'frame_MSE_after':m.get('MSE',''),'delta_frame_MSE':action.get('delta_mse',''),'cumulative_MSE_before':parent['cum_mse'],'cumulative_MSE_after':parent['cum_mse']+m.get('MSE',0) if action.get('valid') else '','PSNR':m.get('PSNR',''),'SSIM':m.get('SSIM',''),'LPIPS':m.get('LPIPS',''),'DISTS':m.get('DISTS',''),'decode_status':action.get('decode_status','INVALID'),'finite_status':action.get('finite_status','INVALID'),'valid_status':'VALID' if action.get('valid') else action.get('reason','INVALID'),'entered_beam':False}

def run_cell(v,qp,device):
 t=tag(v,qp);done=ROOT/'parts'/f'{t}_done.json'
 if done.exists() and json.loads(done.read_text()).get('status')=='PASS':print('reuse',t,flush=True);return
 basepath=V121/'bitstreams'/f'{t}_baseline.bin';payloads,basec=parse_stream(basepath);baseline_total=basec[-1];identity=read_csv(V122/'sanity/identity_audit.csv');expected=next(r['actual_sha256'] for r in identity if r['kind']=='reused_baseline_bitstream' and r['dataset']==v['dataset'] and int(r['video_id'])==int(v['video_id']) and int(r['qp'])==qp)
 if sha256(basepath)!=expected:raise RuntimeError('baseline sha mismatch')
 frames=read_frames(v,device);models=init_metric_models(device);im,_=load_models(device);enc,_=load_b2(device);dec,_=load_b2(device);before=mhash(im,enc,dec);enc.clear_dpb();dec.clear_dpb();enc.set_curr_poc(0);dec.set_curr_poc(0)
 stream=io.BytesIO();write_sps(stream,SPS);_,iq,ip=payloads[0];out=im.decompress(ip,SPS,iq)['x_hat'];enc.add_ref_frame(None,out);dec.add_ref_frame(None,out);stream.write(packet(ip,iq,True));m0=metrics(out,frames[0],models);base_step={'trajectory_id':f'{t}_BASELINE','frame':0,'chosen_action':'BASELINE','candidate_id':f'{t}_b0','actual_frame_bytes':len(packet(ip,iq,True)),'baseline_frame_bytes':len(packet(ip,iq,True)),'cumulative_bytes':len(stream.getvalue()),'baseline_cumulative_bytes':basec[0],'frame_RGB_MSE':m0['MSE'],'baseline_frame_RGB_MSE':m0['MSE'],'cumulative_RGB_MSE':m0['MSE'],'symbol_edits':'','DPB_state_hash':state_hash(snap_cpu(dec))}
 root={'tid':f'{t}_ROOT','parent':'','enc_state':snap_cpu(enc),'dec_state':snap_cpu(dec),'stream':stream.getvalue(),'cum_bytes':len(stream.getvalue()),'cum_mse':m0['MSE'],'steps':[dict(base_step,trajectory_id=f'{t}_ROOT',chosen_action='NO_EDIT')],'frame_metrics':[dict(frame=0,actual_qp=iq,actual_bytes=len(packet(ip,iq,True)),**m0)],'expected_symbols':[None],'expected_recon':[thash(out)],'single':0,'pair':0,'noedit':1,'modified_frames':0,'is_baseline':False,'base_cum':basec[0]}
 baseline=dict(root);baseline.update(tid=f'{t}_BASELINE',is_baseline=True,steps=[base_step]);beam=[root];candidates=[];beam_rows=[];sync=[];counter=0;coverage={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'num_frames':64,'num_beam_states_expanded':0,'num_single_exact_evaluations':0,'num_pair_exact_evaluations':0,'num_rate_saving_actions':0,'num_quality_improving_actions':0,'num_free_improvement_actions':0,'num_donor_actions_entered_beam':0,'num_receiver_actions_entered_beam':0,'num_pair_actions_entered_beam':0,'num_complete_trajectories':0,'num_final_budget_feasible_trajectories':0,'invalid_inactive_nonzero':0,'invalid_entropy_decode':0}
 debt_cap=CFG['temporary_rate_debt_ratio']*baseline_total
 for fi in range(1,64):
  aq=enc.shift_qp(qp,INDEX_MAP[fi%8])
  # Advance immutable original baseline path using its saved payload.
  restore_device(enc,baseline['enc_state'],device);restore_device(dec,baseline['dec_state'],device);_,bq,bp=payloads[fi];boe,bce=decode_capture(enc,bp,SPS,bq);bod,bcd=decode_capture(dec,bp,SPS,bq);bmet=metrics(bod,frames[fi],models);bsym=(bcd['z'].detach().cpu(),bcd['w0'].detach().cpu(),bcd['w1'].detach().cpu());bpacket=packet(bp,bq);baseline=dict(baseline,enc_state=snap_cpu(enc),dec_state=snap_cpu(dec),stream=baseline['stream']+bpacket,cum_bytes=baseline['cum_bytes']+len(bpacket),cum_mse=baseline['cum_mse']+bmet['MSE'],expected_symbols=baseline['expected_symbols']+[bsym],expected_recon=baseline['expected_recon']+[thash(bod)],frame_metrics=baseline['frame_metrics']+[dict(frame=fi,actual_qp=bq,actual_bytes=len(bpacket),**bmet)],steps=baseline['steps']+[{'trajectory_id':baseline['tid'],'frame':fi,'chosen_action':'BASELINE','candidate_id':f'{t}_b{fi}','actual_frame_bytes':len(bpacket),'baseline_frame_bytes':len(bpacket),'cumulative_bytes':baseline['cum_bytes']+len(bpacket),'baseline_cumulative_bytes':basec[fi],'frame_RGB_MSE':bmet['MSE'],'baseline_frame_RGB_MSE':bmet['MSE'],'cumulative_RGB_MSE':baseline['cum_mse']+bmet['MSE'],'symbol_edits':'','DPB_state_hash':state_hash(snap_cpu(dec))}],base_cum=basec[fi],noedit=baseline['noedit']+1)
  expanded=[]
  for parent in beam:
   coverage['num_beam_states_expanded']+=1;restore_device(enc,parent['enc_state'],device);restore_device(dec,parent['dec_state'],device);ctx,ct,qd,qr,natural=prepare(enc,frames[fi],aq);parent['natural_symbols']=natural;cur=evaluate(enc,dec,natural,[],ctx,ct,qd,qr,aq,frames[fi],models)
   if not cur['valid']:raise RuntimeError(f'NO_EDIT invalid frame {fi} trajectory {parent["tid"]}: {cur["reason"]}')
   cur.update(action_type='NO_EDIT',edits=[],packet_bytes=len(packet(cur['payload'],aq)),delta_bytes=0,delta_mse=0)
   grads,active=sensitivity(enc,natural,ctx,ct,qd,qr,frames[fi]);props=ranked_proposals(natural,grads,active);actions=[cur];single=[]
   for fam,idx,delta,score,kind in props:
    coverage['num_single_exact_evaluations']+=1;ev=evaluate(enc,dec,natural,[(fam,idx,delta)],ctx,ct,qd,qr,aq,frames[fi],models);ev.update(action_type='SINGLE',edits=[(fam,idx,delta)],proposal_kind=kind,gradient_score=score)
    if ev['valid']:
     ev['packet_bytes']=len(packet(ev['payload'],aq));ev['delta_bytes']=ev['packet_bytes']-cur['packet_bytes'];ev['delta_mse']=ev['metrics']['MSE']-cur['metrics']['MSE'];coverage['num_rate_saving_actions']+=ev['delta_bytes']<0;coverage['num_quality_improving_actions']+=ev['delta_mse']<0;coverage['num_free_improvement_actions']+=ev['delta_bytes']<=0 and ev['delta_mse']<0
    else:
     coverage['invalid_inactive_nonzero']+=ev['reason']=='INVALID_INACTIVE_NONZERO';coverage['invalid_entropy_decode']+=ev['reason']=='ENTROPY_DECODE_MISMATCH'
    single.append(ev);actions.append(ev)
   donors=sorted([a for a in single if a['valid'] and a['delta_bytes']<0],key=lambda a:(a['delta_mse']/(max(1,-a['delta_bytes']))))[:CFG['pair_top_k_donors']];receivers=sorted([a for a in single if a['valid'] and a['delta_mse']<0],key=lambda a:(a['delta_mse']/(abs(a['delta_bytes'])+1)))[:CFG['pair_top_k_receivers']]
   pc=0
   for d in donors:
    for rec in receivers:
     if pc>=CFG['max_pair_exact_per_state']:break
     if d['edits'][0][:2]==rec['edits'][0][:2]:continue
     pc+=1;coverage['num_pair_exact_evaluations']+=1;eds=[d['edits'][0],rec['edits'][0]];ev=evaluate(enc,dec,natural,eds,ctx,ct,qd,qr,aq,frames[fi],models);ev.update(action_type='PAIR',edits=eds,proposal_kind='DONOR_RECEIVER',gradient_score='')
     if ev['valid']:
      ev['packet_bytes']=len(packet(ev['payload'],aq));ev['delta_bytes']=ev['packet_bytes']-cur['packet_bytes'];ev['delta_mse']=ev['metrics']['MSE']-cur['metrics']['MSE'];coverage['num_rate_saving_actions']+=ev['delta_bytes']<0;coverage['num_quality_improving_actions']+=ev['delta_mse']<0;coverage['num_free_improvement_actions']+=ev['delta_bytes']<=0 and ev['delta_mse']<0
     else:
      coverage['invalid_inactive_nonzero']+=ev['reason']=='INVALID_INACTIVE_NONZERO';coverage['invalid_entropy_decode']+=ev['reason']=='ENTROPY_DECODE_MISMATCH'
     actions.append(ev)
   rows=[]
   for a in actions:
    counter+=1;cid=f'{t}_c{counter}';row=action_row(v,qp,fi,parent,cid,a,cur,basec[fi],baseline_total);a['row']=row;a['cid']=cid;rows.append(row)
   candidates+=rows
   for a in choose_actions(actions):
    if not a['valid']:continue
    newcum=parent['cum_bytes']+a['packet_bytes']
    if newcum-basec[fi]>debt_cap:continue
    restore_device(enc,parent['enc_state'],device);restore_device(dec,parent['dec_state'],device);enc.add_ref_frame(a['feature'],None);out2,cap2=decode_capture(dec,a['payload'],SPS,aq);sy=a['symbols'];sym=bool(torch.equal(sy[0],cap2['z']) and torch.equal(sy[1],cap2['w0']) and torch.equal(sy[2],cap2['w1']));serr=float((enc.dpb[0].feature-dec.dpb[0].feature).abs().max());finite=bool(torch.isfinite(out2).all());counter+=1;tid=f'{t}_T{counter}';pkt=packet(a['payload'],aq);edtxt=json.dumps(a['edits']);step={'trajectory_id':tid,'frame':fi,'chosen_action':a['action_type'],'candidate_id':a['cid'],'actual_frame_bytes':len(pkt),'baseline_frame_bytes':basec[fi]-basec[fi-1],'cumulative_bytes':newcum,'baseline_cumulative_bytes':basec[fi],'frame_RGB_MSE':a['metrics']['MSE'],'baseline_frame_RGB_MSE':bmet['MSE'],'cumulative_RGB_MSE':parent['cum_mse']+a['metrics']['MSE'],'symbol_edits':edtxt,'DPB_state_hash':state_hash(snap_cpu(dec))}
    st={'tid':tid,'parent':parent['tid'],'enc_state':snap_cpu(enc),'dec_state':snap_cpu(dec),'stream':parent['stream']+pkt,'cum_bytes':newcum,'cum_mse':parent['cum_mse']+a['metrics']['MSE'],'steps':parent['steps']+[step],'frame_metrics':parent['frame_metrics']+[dict(frame=fi,actual_qp=aq,actual_bytes=len(pkt),**a['metrics'])],'expected_symbols':parent['expected_symbols']+[tuple(x.detach().cpu() for x in sy)],'expected_recon':parent['expected_recon']+[thash(out2)],'single':parent['single']+(a['action_type']=='SINGLE'),'pair':parent['pair']+(a['action_type']=='PAIR'),'noedit':parent['noedit']+(a['action_type']=='NO_EDIT'),'modified_frames':parent['modified_frames']+(a['action_type']!='NO_EDIT'),'is_baseline':False,'base_cum':basec[fi],'row':a['row']}
    expanded.append(st);sync.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'frame':fi,'trajectory_id':tid,'parent_trajectory_id':parent['tid'],'symbol_equal':sym,'state_max_abs':serr,'finite':finite,'status':'PASS' if sym and serr==0 and finite else 'FAIL'})
  beam,allstates=select_beam(expanded,baseline_total)
  for s in beam:s['row']['entered_beam']=True;coverage['num_donor_actions_entered_beam']+=s['row']['delta_frame_bytes']<0;coverage['num_receiver_actions_entered_beam']+=s['row']['delta_frame_MSE']<0;coverage['num_pair_actions_entered_beam']+=s['row']['action_type']=='PAIR'
  survivors=[baseline]+beam
  for rank,s in enumerate(survivors):beam_rows.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'frame':fi,'trajectory_id':s['tid'],'parent_trajectory_id':s.get('parent',''),'cumulative_bytes':s['cum_bytes'],'baseline_cumulative_bytes':basec[fi],'rate_gap_bytes':s['cum_bytes']-basec[fi],'rate_gap_percent_of_final_baseline':100*(s['cum_bytes']-basec[fi])/baseline_total,'cumulative_RGB_MSE':s['cum_mse'],'average_RGB_MSE':s['cum_mse']/(fi+1),'num_single_edits':s['single'],'num_pair_edits':s['pair'],'num_no_edit_frames':s['noedit'],'beam_rank':rank,'pareto_dominated':False,'is_baseline_path':s['is_baseline']})
  if fi%4==0:
   write_csv(ROOT/'parts'/f'{t}_candidate_actions.partial.csv',candidates);write_csv(ROOT/'parts'/f'{t}_beam_states.partial.csv',beam_rows);write_csv(ROOT/'parts'/f'{t}_state_sync.partial.csv',sync);print(t,'frame',fi,'beam',len(beam),'candidates',len(candidates),flush=True)
 finals=[baseline]+beam;coverage['num_complete_trajectories']=len(finals);coverage['num_final_budget_feasible_trajectories']=sum(s['cum_bytes']<=baseline_total for s in finals);base=baseline;frows=[];aud=[];allsteps=[];allframes=[]
 for s in finals:
  path=ROOT/'bitstreams'/f'{t}_{s["tid"].split("_")[-1]}.bin';path.write_bytes(s['stream']);a=independent(path,s['expected_symbols'],s['expected_recon'],device);avg={k:sum(float(x[k]) for x in s['frame_metrics'])/64 for k in ('MSE','PSNR','SSIM','LPIPS','DISTS')};bavg={k:sum(float(x[k]) for x in base['frame_metrics'])/64 for k in ('MSE','PSNR','SSIM','LPIPS','DISTS')};feas=s['cum_bytes']<=baseline_total
  frows.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'trajectory_id':s['tid'],'is_baseline_path':s['is_baseline'],'is_selected':False,'total_bytes':s['cum_bytes'],'baseline_bytes':baseline_total,'rate_ratio':s['cum_bytes']/baseline_total,'rate_gap_bytes':s['cum_bytes']-baseline_total,'rate_gap_percent':100*(s['cum_bytes']/baseline_total-1),'average_RGB_MSE':avg['MSE'],'baseline_RGB_MSE':bavg['MSE'],'delta_RGB_MSE':avg['MSE']-bavg['MSE'],'PSNR':avg['PSNR'],'baseline_PSNR':bavg['PSNR'],'delta_PSNR':avg['PSNR']-bavg['PSNR'],'SSIM':avg['SSIM'],'baseline_SSIM':bavg['SSIM'],'delta_SSIM':avg['SSIM']-bavg['SSIM'],'LPIPS':avg['LPIPS'],'baseline_LPIPS':bavg['LPIPS'],'delta_LPIPS':avg['LPIPS']-bavg['LPIPS'],'DISTS':avg['DISTS'],'baseline_DISTS':bavg['DISTS'],'delta_DISTS':avg['DISTS']-bavg['DISTS'],'num_single_edits':s['single'],'num_pair_edits':s['pair'],'num_modified_frames':s['modified_frames'],'final_budget_feasible':feas,'bitstream_path':str(path),'bitstream_sha256':sha256(path),'decode_status':a['decode_status'],'state_sync_status':'PASS' if a['symbol_equal'] and all(x['status']=='PASS' for x in sync if x['trajectory_id']==s['tid']) else ('PASS' if s['is_baseline'] and a['symbol_equal'] else 'FAIL'),'finite_status':'PASS' if a['finite'] else 'FAIL'});aud.append(dict(dataset=v['dataset'],video=v['name'],video_id=v['video_id'],qp=qp,trajectory_id=s['tid'],bitstream_path=str(path),bitstream_sha256=sha256(path),**a));allsteps+=s['steps'];allframes += [dict(dataset=v['dataset'],video=v['name'],video_id=v['video_id'],qp=qp,trajectory_id=s['tid'],is_baseline_path=s['is_baseline'],**x) for x in s['frame_metrics']]
 feasible=[r for r in frows if r['final_budget_feasible'] and r['decode_status']=='PASS' and r['finite_status']=='PASS'];selected=min(feasible,key=lambda r:float(r['average_RGB_MSE']));selected['is_selected']=True;stype='BASELINE' if selected['is_baseline_path'] else 'DISCRETE_SEQUENCE';final={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'selected_type':stype,'baseline_bytes':selected['baseline_bytes'],'selected_bytes':selected['total_bytes'],'rate_gap_bytes':selected['rate_gap_bytes'],'rate_gap_percent':selected['rate_gap_percent'],'baseline_PSNR':selected['baseline_PSNR'],'selected_PSNR':selected['PSNR'],'delta_PSNR':selected['delta_PSNR'],'baseline_SSIM':selected['baseline_SSIM'],'selected_SSIM':selected['SSIM'],'delta_SSIM':selected['delta_SSIM'],'baseline_LPIPS':selected['baseline_LPIPS'],'selected_LPIPS':selected['LPIPS'],'delta_LPIPS':selected['delta_LPIPS'],'baseline_DISTS':selected['baseline_DISTS'],'selected_DISTS':selected['DISTS'],'delta_DISTS':selected['delta_DISTS'],'num_modified_frames':selected['num_modified_frames'],'num_single_edits':selected['num_single_edits'],'num_pair_edits':selected['num_pair_edits'],'bitstream_path':selected['bitstream_path'],'bitstream_sha256':selected['bitstream_sha256'],'decode_status':selected['decode_status'],'state_sync_status':selected['state_sync_status'],'finite_status':selected['finite_status']}
 after=mhash(im,enc,dec);ok=len(frows)>=1 and final['selected_bytes']<=baseline_total and all(r['decode_status']=='PASS' and r['finite_status']=='PASS' for r in frows) and before==after
 write_csv(ROOT/'parts'/f'{t}_candidate_actions.csv',candidates);write_csv(ROOT/'parts'/f'{t}_beam_states.csv',beam_rows);write_csv(ROOT/'parts'/f'{t}_trajectory_steps.csv',allsteps);write_csv(ROOT/'parts'/f'{t}_final_trajectories.csv',frows);write_csv(ROOT/'parts'/f'{t}_final_stream.csv',[final]);write_csv(ROOT/'parts'/f'{t}_frame_metrics.csv',allframes);write_csv(ROOT/'parts'/f'{t}_search_coverage.csv',[coverage]);write_csv(ROOT/'parts'/f'{t}_state_sync.csv',sync);write_csv(ROOT/'parts'/f'{t}_bitstream_audit.csv',aud);done.write_text(json.dumps({'status':'PASS' if ok else 'FAIL','tag':t,'model_hash_before':before,'model_hash_after':after,'model_unchanged':before==after,'final':final},indent=2)+'\n')
 if not ok:raise RuntimeError(f'cell integrity failure {t}')

def sanity(device):
 v=next(x for x in MAN['videos'] if x['dataset']=='fresh_ulong' and int(x['video_id'])==10);frames=read_frames(v,device);im,_=load_models(device);enc,_=load_b2(device);dec,_=load_b2(device);enc.clear_dpb();dec.clear_dpb();ie=im.compress(frames[0],1);out=im.decompress(ie['bit_stream'],SPS,1)['x_hat'];enc.add_ref_frame(None,out);dec.add_ref_frame(None,out);root_e=snap_cpu(enc);root_d=snap_cpu(dec);aq=enc.shift_qp(1,INDEX_MAP[1]);ctx,ct,qd,qr,sy=prepare(enc,frames[1],aq);base=encode_checked(enc,sy,ctx,ct,qd,qr,aq);_,_,s0,s1=reconstruct(enc,*sy,ctx,ct,qd,qr);thr=enc.gaussian_encoder.force_zero_thres;ids=torch.nonzero((s1>thr).reshape(-1),as_tuple=False).reshape(-1);edits=[('w1',int(ids[0]),1),('w1',int(ids[1]),-1)];hashes=[]
 for ed in edits:
  restore_device(enc,root_e,device);restore_device(dec,root_d,device);s2=apply_edits(sy,[ed]);e=encode_checked(enc,s2,ctx,ct,qd,qr,aq)
  if not e['valid']:
   raise RuntimeError('sibling candidate invalid')
  enc.add_ref_frame(e['feature'],None)
  o,c=decode_capture(dec,e['payload'],SPS,aq)
  hashes.append(state_hash(snap_cpu(dec)))
 restore_device(dec,root_d,device);root_again=state_hash(snap_cpu(dec));passed=hashes[0]!=hashes[1] and root_again==state_hash(root_d);row={'status':'PASS' if passed else 'FAIL','sibling_1_state_hash':hashes[0],'sibling_2_state_hash':hashes[1],'siblings_different':hashes[0]!=hashes[1],'root_restore_exact':root_again==state_hash(root_d),'edits':edits};(ROOT/'sanity/sibling_state.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row,indent=2));return passed

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--cells',default='');ap.add_argument('--sanity-only',action='store_true');a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);torch.manual_seed(20260922);device=torch.device('cuda:0')
 if a.sanity_only:raise SystemExit(0 if sanity(device) else 2)
 lookup={(v['dataset'],int(v['video_id'])):v for v in MAN['videos']}
 for spec in a.cells.split(','):
  ds,vid,q=spec.split(':');t=f'{ds}_{int(vid):02d}_qp{q}'
  try:run_cell(lookup[(ds,int(vid))],int(q),device)
  except Exception as e:(ROOT/'parts'/f'{t}_FAILED.json').write_text(json.dumps({'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()},indent=2)+'\n');print('FAILED',t,repr(e),flush=True)
if __name__=='__main__':main()
