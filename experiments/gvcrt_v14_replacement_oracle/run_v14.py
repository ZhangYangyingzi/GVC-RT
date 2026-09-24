#!/usr/bin/env python3
import argparse,csv,faulthandler,gc,hashlib,json,math,shutil,struct,sys,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];V13=ROOT.parent/'gvcrt_v13_generator_aware_mixed_precision';V131=ROOT.parent/'gvcrt_v13_1_structured_mixed_precision'
for p in (ROOT,REPO,V13,V131):sys.path.insert(0,str(p))
import run_v13 as base
import v14_codec as codec
CFG=json.loads((ROOT/'config.json').read_text());MAN=json.loads((ROOT/'manifest.json').read_text());SPS=base.SPS
class Tee:
 def __init__(self,s,p):self.s=s;self.f=open(p,'a',buffering=1)
 def write(self,x):self.s.write(x);self.f.write(x);return len(x)
 def flush(self):self.s.flush();self.f.flush()
def write_csv(path,rows):
 rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else []
 with path.open('w',newline='') as f:
  if fields:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def read_csv(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
def tag(v,q):return f"{v['dataset']}_{int(v['video_id']):02d}_qp{q}"
def gtag(v,q,g):return f'{tag(v,q)}_{g}'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pix(rgb):return base.unit(rgb)
def sse(rgb,gt):
 d=(pix(rgb)-pix(gt)).double();return float(d.square().sum()),d.numel()
def basic(rgb,gt):
 z,n=sse(rgb,gt);m=z/n;return z,n,m,-10*math.log10(max(m,1e-20))
def state_hash(s):return base.state_hash(s)

def baseline(v,qp):
 t=tag(v,qp);src=V131/'bitstreams'/f'{t}_baseline.bin';dst=ROOT/'bitstreams'/f'{t}_baseline.bin';shutil.copy2(src,dst)
 row=next(r for r in read_csv(V131/'baseline_cells.csv') if r['dataset']==v['dataset'] and int(r['video_id'])==int(v['video_id']) and int(r['qp'])==qp)
 if sha(src)!=row['bitstream_sha256'] or sha(dst)!=sha(src):raise RuntimeError('baseline hash identity')
 frames=[r for r in read_csv(V131/'frame_metrics.csv') if r['dataset']==v['dataset'] and int(r['video_id'])==int(v['video_id']) and int(r['qp'])==qp and r['trajectory']=='baseline']
 out={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'frames':64,'baseline_bytes':dst.stat().st_size,'baseline_bits':dst.stat().st_size*8,
  'I_frame_bytes':row['I_bytes'],'z_bytes':row['z_bytes'],'w0_bytes':row['w0_bytes'],'w1_bytes':row['w1_bytes'],'PSNR':row['PSNR'],'MS_SSIM':row['MS_SSIM'],'LPIPS':row['LPIPS'],'DISTS':row['DISTS'],
  'bitstream_path':str(dst),'bitstream_sha256':sha(dst),'codec_payload_identity_pass':True,'identity_gate_pass':True,'model_hash_before':row['model_hash_before'],'model_hash_after':row['model_hash_after']}
 return out,frames

def evaluate(enc,dec,prep,modes,aq,gt,ds,tile,metrics=None,original=False,global_action=None):
 base.restore_device(dec,ds,prep['w1'].device);cand=prep if modes==prep['modes'] else codec.apply_actions(enc,prep,modes,tile,True)
 if original:payload,comp=codec.encode_original(enc,cand,aq)
 else:payload,comp=codec.encode_replacement(enc,cand,aq,tile,global_action)
 out,cap=codec.decode_replacement(dec,payload,SPS,aq,tile)
 if not torch.equal(cap['z'],cand['z']) or not torch.equal(cap['w0'],cand['w0']) or not torch.equal(cap['w1'],cand['w1']):raise RuntimeError('entropy symbol mismatch')
 if not original and (cap['modes']!=modes or not torch.equal(cap['q_pred'],cand['q_pred'])):raise RuntimeError('predictor/map mismatch')
 if not torch.equal(cap['latent'],cand['latent']) or not torch.equal(cap['feature'],cand['feature']) or not torch.equal(out,cand['rgb']):raise RuntimeError('reconstruction mismatch')
 z,n,m,p=basic(out,gt);met={'pixel_SSE':z,'pixels':n,'MSE':m,'sequence_compatible_pixel_SSE':z,'mean_frame_PSNR':p}
 if metrics is not None:met.update(base.frame_metric(out,gt,metrics))
 return cand,payload,comp,out,met,base.snap_cpu(dec),cap

def predictor_rows(v,qp,gran,fi,prep,tile):
 qt,pr,sc=prep['q_true'],prep['q_pred'],prep['s1'];active=sc!=0;rows=[]
 def add(level,key,sel):
  e=(qt-pr)[sel].float();rows.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'granularity':gran,'frame':fi,'level':level,**key,'num_symbols':int(e.numel()),'exact_match_rate':float((e==0).float().mean()) if e.numel() else 1.,'absolute_error':float(e.abs().mean()) if e.numel() else 0.,'squared_error':float(e.square().mean()) if e.numel() else 0.,'signed_error':float(e.mean()) if e.numel() else 0.,'RMSE':float(e.square().mean().sqrt()) if e.numel() else 0.})
 add('frame',{},active)
 for ch in range(qt.shape[1]):
  sel=torch.zeros_like(active);sel[:,ch:ch+1]=active[:,ch:ch+1];add('channel',{'channel':ch},sel)
 h,w=qt.shape[-2:];gh,gw=codec.block_grid(h,w,tile,tile)
 for bi in range(gh*gw):
  by,bx=divmod(bi,gw);sel=torch.zeros_like(active);sel[...,by*tile:min((by+1)*tile,h),bx*tile:min((bx+1)*tile,w)]=active[...,by*tile:min((by+1)*tile,h),bx*tile:min((bx+1)*tile,w)];add('tile',{'tile':bi,'tile_y':by,'tile_x':bx},sel)
 return rows

@torch.no_grad()
def build_proposals(v,frames,qp,gran,tile,device,metrics):
 gt=gtag(v,qp,gran);done=ROOT/'parts'/f'{gt}_proposals_done.json';pf=ROOT/'artifacts'/f'{gt}_proposals.json'
 if done.exists():return json.loads(pf.read_text()),json.loads(done.read_text())['coverage']
 iframe,enc,dec=base.load_triplet(device);base.init_sequence(iframe,enc,dec,frames[0],qp);inter=[];fc=[];corr=[];maps=[];predrows=[];recover=[];proposals={};rans=0
 for fi in range(1,64):
  enc.update(.12);dec.update(.12);enc.set_use_two_entropy_coders(True);dec.set_use_two_entropy_coders(True);aq=enc.shift_qp(qp,base.INDEX_MAP[fi%8]);prep=codec.prepare_frame(enc,frames[fi],aq,None,tile);ds=base.snap_cpu(dec);bp,bc=codec.encode_original(enc,prep,aq);bout,bcap=codec.decode_replacement(dec,bp,SPS,aq,tile);bz,bn,bm,bps=basic(bout,frames[fi]);base_eval=base.frame_metric(bout,frames[fi],metrics);base_packet=len(base.packet(bp,aq));h,w=prep['shape'][-2:];gh,gw=codec.block_grid(h,w,tile,tile);nt=gh*gw;predrows+=predictor_rows(v,qp,gran,fi,prep,tile);bytile={}
  for bi in range(nt):
   bytile[bi]={}
   for name in ('PRED_ONLY','CORR2','CORR4'):
    modes=[0]*nt;modes[bi]=codec.ACTION[name];cand,pay,comp,out,met,_,cap=evaluate(enc,dec,prep,modes,aq,frames[fi],ds,tile,metrics);rans+=1;bits=(base_packet-len(base.packet(pay,aq)))*8;dm=met['pixel_SSE']-bz
    row={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'granularity':gran,'frame':fi,'tile':bi,'tile_y':bi//gw,'tile_x':bi%gw,'mode':name,'baseline_frame_bytes':base_packet,'candidate_frame_bytes':len(base.packet(pay,aq)),'actual_saved_bits':bits,'original_w1_bits':bc['w1_bytes']*8,'retained_w1_bits':comp['original_retained_w1_bytes']*8,'correction_bits':comp['replacement_correction_bytes']*8,'mode_map_bits':comp['mode_map_bits'],'total_frame_bits':len(base.packet(pay,aq))*8,'baseline_SSE':bz,'candidate_SSE':met['pixel_SSE'],'delta_SSE':dm,'MSE':met['MSE'],'mean_frame_PSNR':met['mean_frame_PSNR'],'SSIM':met['MS_SSIM'],'LPIPS':met['LPIPS'],'DISTS':met['DISTS'],'rd_score':dm/bits if bits>0 else '','decode_status':'PASS'}
    inter.append(row);bytile[bi][name]=row
    ys=slice(bi//gw*tile,min((bi//gw+1)*tile,h));xs=slice(bi%gw*tile,min((bi%gw+1)*tile,w));ts=prep['s1'][...,ys,xs];qt=prep['q_true'][...,ys,xs]
    est_orig=codec.estimated_symbol_bits(enc,qt,ts);step={'PRED_ONLY':0,'CORR2':2,'CORR4':4}[name]
    est_corr=0.0 if step==0 else codec.estimated_symbol_bits(enc,cand['correction'][...,ys,xs],ts,step)
    corr.append({'dataset':v['dataset'],'video':v['name'],'qp':qp,'frame':fi,'tile':bi,'mode':name,'step':step,'num_symbols':int(codec.active_mask(enc,ts).sum()),'original_w1_bits':bc['w1_bytes']*8,'correction_bits':comp['replacement_correction_bytes']*8,'estimated_original_bits':est_orig,'estimated_correction_bits':est_corr,'symbol_equal_to_baseline':torch.equal(cap['w1'],prep['w1']),'cdf_equal_encoder_decoder':True,'decode_status':'PASS'})
  modesets={'FULL_000':[0]*nt}
  for name in ('PRED_ONLY','CORR2','CORR4'):
   ranked=sorted([r for r in (bytile[i][name] for i in range(nt)) if r['actual_saved_bits']>0],key=lambda r:(r['delta_SSE']/r['actual_saved_bits'],r['tile']))
   for frac in CFG['candidate_fractions'][1:]:
    m=[0]*nt
    for r in ranked[:min(len(ranked),math.ceil(nt*frac))]:m[r['tile']]=codec.ACTION[name]
    modesets[f'{name}_F{int(round(frac*100)):03d}']=m
  for lam in CFG['mixed_lambdas']:
   m=[]
   for bi in range(nt):
    choices=[('FULL',0.,0.)]+[(n,bytile[bi][n]['delta_SSE'],-bytile[bi][n]['actual_saved_bits']) for n in ('PRED_ONLY','CORR2','CORR4')]
    pick=min(choices,key=lambda x:(x[1]+lam*x[2],x[0]))[0];m.append(codec.ACTION[pick])
   modesets[f'MIX_L{lam:g}']=m
  uniq={};proposals[str(fi)]={}
  for name,modes in modesets.items():
   key=tuple(modes)
   if key in uniq:continue
   uniq[key]=name;original=all(x==0 for x in modes);cand,pay,comp,out,met,_,_=evaluate(enc,dec,prep,modes,aq,frames[fi],ds,tile,metrics,original=original);rans+=not original;proposals[str(fi)][name]=modes
   fc.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'granularity':gran,'frame':fi,'candidate_name':name,'number_tiles':nt,'FULL_tiles':modes.count(0),'PRED_ONLY_tiles':modes.count(1),'CORR2_tiles':modes.count(2),'CORR4_tiles':modes.count(3),'actual_frame_bytes':len(base.packet(pay,aq)),'pixel_SSE':met['pixel_SSE'],'MSE':met['MSE'],'mean_frame_PSNR':met['mean_frame_PSNR'],'LPIPS':met['LPIPS'],'DISTS':met['DISTS']})
   if original:mi={'syntax':'ORIGINAL_NO_MAP','bits':0,'data':b'','rle_bits':0,'dominant_bits':0,'packed_bits':0}
   else:mi=codec.encode_mode_map_best(modes)
   maps.append({'dataset':v['dataset'],'video':v['name'],'qp':qp,'granularity':gran,'frame':fi,'candidate_name':name,'syntax_type':mi['syntax'],'raw_modes':json.dumps(modes,separators=(',',':')),'encoded_bytes':len(mi['data']),'encoded_bits':mi['bits'],'header_bytes':0 if original else codec.HEADER.size,'padding_alignment_bits':len(mi['data'])*8-mi['bits'],'rle_bits':mi['rle_bits'],'dominant_bits':mi['dominant_bits'],'packed_bits':mi['packed_bits']})
  for bi in range(nt):
   d={'dataset':v['dataset'],'video':v['name'],'qp':qp,'granularity':gran,'frame':fi,'tile':bi}
   pr=[r for r in predrows if r['frame']==fi and r['level']=='tile' and r.get('tile')==bi][-1];d.update(num_symbols=pr['num_symbols'],predictor_exact_match_rate=pr['exact_match_rate'],predictor_MAE=pr['absolute_error'],predictor_RMSE=pr['RMSE'])
   for n in ('PRED_ONLY','CORR2','CORR4'):
    r=bytile[bi][n];d.update({f'{n}_delta_actual_bits':-r['actual_saved_bits'],f'{n}_delta_SSE':r['delta_SSE'],f'{n}_delta_LPIPS':float(r['LPIPS'])-float(base_eval['LPIPS']),f'{n}_delta_DISTS':float(r['DISTS'])-float(base_eval['DISTS'])})
   recover.append(d)
  base.restore_device(dec,ds,device);codec.decode_replacement(dec,bp,SPS,aq,tile);enc.add_ref_frame(prep['feature'],None)
  if fi%2==0:
   write_csv(str(ROOT/'parts'/f'{gt}_tile_interventions.partial.csv'),inter);pf.write_text(json.dumps(proposals,separators=(',',':')))
  print(gt,'proposal',fi,'tiles',nt,'maps',len(proposals[str(fi)]),flush=True)
 for suffix,rows in [('tile_interventions',inter),('frame_candidates',fc),('correction_coder_audit',corr),('mode_map_audit',maps),('predictor_symbol_audit',predrows),('recoverability_stats',recover)]:write_csv(ROOT/'parts'/f'{gt}_{suffix}.csv',rows)
 pf.write_text(json.dumps(proposals,separators=(',',':')));cov={'tiles':sum(1 for _ in recover),'tile_exact_evaluations':len(inter),'frame_candidates':len(fc),'actual_RANS_encodes':rans};done.write_text(json.dumps({'status':'PASS','coverage':cov},indent=2));return proposals,cov

def dedup(states):
 d={};n=0
 for s in states:
  k=(hashlib.sha256(s['stream']).hexdigest(),state_hash(s['dec_state']))
  if k in d:n+=1
  else:d[k]=s
 return list(d.values()),n

def select_beam(states,bprefix):
 states,dups=dedup(states);states=[s for s in states if s['bytes']/bprefix<=CFG['prefix_max_rate_ratio']];pareto=[]
 for i,a in enumerate(states):
  if not any(i!=j and b['bytes']<=a['bytes'] and b['sse']<=a['sse'] and (b['bytes']<a['bytes'] or b['sse']<a['sse']) for j,b in enumerate(states)):pareto.append(a)
 chosen=[]
 for lo,hi in CFG['rate_buckets']:
  pool=[s for s in pareto if lo<=s['bytes']/bprefix<hi]
  for s in sorted(pool,key=lambda x:(x['sse'],x['bytes']))[:2]:
   if s not in chosen:chosen.append(s)
 for seq in (sorted(pareto,key=lambda x:(x['sse'],x['bytes'])),sorted(pareto,key=lambda x:(x['bytes'],x['sse'])),sorted(states,key=lambda x:(x['sse'],x['bytes']))):
  for s in seq:
   if len(chosen)>=CFG['beam_width']:break
   if s not in chosen:chosen.append(s)
 return chosen[:CFG['beam_width']],dups

@torch.no_grad()
def search(v,frames,qp,gran,tile,device,props,brow,bframes):
 gt=gtag(v,qp,gran);done=ROOT/'parts'/f'{gt}_search_done.json';af=ROOT/'artifacts'/f'{gt}_final_states.pt'
 if done.exists():return torch.load(af,map_location='cpu',weights_only=False),json.loads(done.read_text())['coverage']
 iframe,enc,dec=base.load_triplet(device);initial,rgb0,_=base.init_sequence(iframe,enc,dec,frames[0],qp);z0,n0,_,_=basic(rgb0,frames[0]);root={'tid':'ROOT','enc_state':base.snap_cpu(enc),'dec_state':base.snap_cpu(dec),'stream':initial,'bytes':len(initial),'sse':z0,'pixels':n0,'history':[],'baseline':True};bs=root;beam=[];rows=[];serial=expanded=rans=dups=0;bprefix=len(initial);ck=ROOT/'artifacts'/f'{gt}_checkpoint.pt';start=1
 if ck.exists():
  x=torch.load(ck,map_location='cpu',weights_only=False);bs=x['bs'];beam=x['beam'];rows=x['rows'];serial=x['serial'];expanded=x['expanded'];rans=x['rans'];dups=x['dups'];bprefix=x['bprefix'];start=x['next'];print(gt,'resume',start,flush=True)
 for fi in range(start,64):
  bprefix+=int(float(bframes[fi]['actual_bytes']));aq=enc.shift_qp(qp,base.INDEX_MAP[fi%8]);succ=[]
  for parent in [bs]+beam:
   base.restore_device(enc,parent['enc_state'],device);base.restore_device(dec,parent['dec_state'],device);enc.update(.12);dec.update(.12);enc.set_use_two_entropy_coders(True);dec.set_use_two_entropy_coders(True);prep=codec.prepare_frame(enc,frames[fi],aq,None,tile);ds=base.snap_cpu(dec);expanded+=1
   for name,modes in props[str(fi)].items():
    original=all(x==0 for x in modes);cand,pay,comp,out,met,newdec,_=evaluate(enc,dec,prep,modes,aq,frames[fi],ds,tile,None,original=original);rans+=not original;base.restore_device(enc,parent['enc_state'],device);enc.add_ref_frame(cand['feature'],None);pkt=base.packet(pay,aq);serial+=1;s={'tid':f'{gt}_T{serial}','enc_state':base.snap_cpu(enc),'dec_state':newdec,'stream':parent['stream']+pkt,'bytes':parent['bytes']+len(pkt),'sse':parent['sse']+met['pixel_SSE'],'pixels':parent['pixels']+met['pixels'],'history':parent['history']+[name],'baseline':parent['baseline'] and original};succ.append(s)
  b=[s for s in succ if s['baseline']]
  if len(b)!=1:raise RuntimeError('baseline multiplicity')
  bs=b[0];beam,d=select_beam([s for s in succ if not s['baseline']],bprefix);dups+=d
  for rank,s in enumerate(beam):rows.append({'dataset':v['dataset'],'video':v['name'],'qp':qp,'granularity':gran,'frame':fi,'trajectory_id':s['tid'],'cumulative_actual_bytes':s['bytes'],'baseline_prefix_bytes':bprefix,'rate_ratio':s['bytes']/bprefix,'cumulative_pixel_SSE':s['sse'],'aggregate_MSE':s['sse']/s['pixels'],'beam_rank':rank,'bitstream_hash':hashlib.sha256(s['stream']).hexdigest(),'DPB_hash':state_hash(s['dec_state'])})
  if fi%2==0:write_csv(ROOT/'parts'/f'{gt}_beam_states.partial.csv',rows);torch.save({'bs':bs,'beam':beam,'rows':rows,'serial':serial,'expanded':expanded,'rans':rans,'dups':dups,'bprefix':bprefix,'next':fi+1},ck)
  print(gt,'beam',fi,len(beam),flush=True)
 write_csv(ROOT/'parts'/f'{gt}_beam_states.csv',rows);final=[]
 for s in [bs]+beam:
  p=ROOT/'bitstreams'/f"{s['tid']}.bin";p.write_bytes(s['stream']);final.append({'trajectory_id':s['tid'],'history':s['history'],'is_baseline_path':bool(s['baseline']),'total_bytes':s['bytes'],'total_pixel_SSE':s['sse'],'total_pixels':s['pixels'],'aggregate_MSE':s['sse']/s['pixels'],'sequence_PSNR':-10*math.log10(s['sse']/s['pixels']),'bitstream_path':str(p),'bitstream_sha256':sha(p)})
 torch.save(final,af);cov={'beam_states_expanded':expanded,'actual_RANS_encodes':rans,'duplicate_states_pruned':dups,'complete_trajectories':len(final)};done.write_text(json.dumps({'status':'PASS','coverage':cov},indent=2));ck.unlink(missing_ok=True);return final,cov
@torch.no_grad()
def replay_audit(v,frames,qp,gran,tile,device,metrics,props,state,label):
 iframe,enc,dec=base.load_triplet(device);stream,rgb0,ip=base.init_sequence(iframe,enc,dec,frames[0],qp);expected=[{'rgb':codec.tensor_hash(rgb0),'state':state_hash(base.snap_cpu(dec))}];tot={'I_frame_bytes':len(ip),'z_bytes':0,'w0_bytes':0,'original_retained_w1_bytes':0,'replacement_correction_bytes':0,'mode_map_bytes':0,'alignment_bytes':0};frame_rows=[{'frame':0,'actual_bytes':len(base.packet(ip,qp,True)),**base.frame_metric(rgb0,frames[0],metrics)}]
 for fi,name in enumerate(state['history'],1):
  enc.update(.12);dec.update(.12);enc.set_use_two_entropy_coders(True);dec.set_use_two_entropy_coders(True);aq=enc.shift_qp(qp,base.INDEX_MAP[fi%8]);prep=codec.prepare_frame(enc,frames[fi],aq,None,tile);modes=props[str(fi)][name];ds=base.snap_cpu(dec);original=all(x==0 for x in modes);cand,pay,comp,out,met,newdec,cap=evaluate(enc,dec,prep,modes,aq,frames[fi],ds,tile,None,original=original);enc.add_ref_frame(cand['feature'],None);stream+=base.packet(pay,aq)
  if original:
   tot['z_bytes']+=comp['z_bytes'];tot['w0_bytes']+=comp['w0_bytes'];tot['original_retained_w1_bytes']+=comp['w1_bytes']
  else:
   for k in ('z_bytes','w0_bytes','original_retained_w1_bytes','replacement_correction_bytes','mode_map_bytes','alignment_bytes'):tot[k]+=comp[k]
  expected.append({'modes':modes,'qpred':codec.tensor_hash(cand['q_pred']),'w1':codec.tensor_hash(cap['w1']),'latent':codec.tensor_hash(cap['latent']),'feature':codec.tensor_hash(cap['feature']),'rgb':codec.tensor_hash(out),'state':state_hash(newdec)})
 if stream!=Path(state['bitstream_path']).read_bytes():raise RuntimeError('trajectory replay mismatch')
 path=ROOT/'bitstreams'/f'{gtag(v,qp,gran)}_{label}_{state["trajectory_id"]}.bin';path.write_bytes(stream);payloads,cum=base.parse_stream(path);iframe2,_,dec2=base.load_triplet(device);dec2.clear_dpb();dec2.set_curr_poc(0);maps=predok=cdfok=symok=latentok=stateok=rgbok=True;rows=[];total_sse=total_pixels=0
 for fi,(is_i,aq,pay) in enumerate(payloads):
  if is_i:out=iframe2.decompress(pay,SPS,aq)['x_hat'];dec2.add_ref_frame(None,out)
  else:
   dec2.update(.12);dec2.set_use_two_entropy_coders(True)
   out,cap=codec.decode_replacement(dec2,pay,SPS,aq,tile);e=expected[fi];parsed=codec.parse_replacement(pay,tile);dm=[0]*len(e['modes']) if parsed is None else parsed['modes'];maps &= dm==e['modes'];predok &= parsed is None or codec.tensor_hash(cap['q_pred'])==e['qpred'];symok &= codec.tensor_hash(cap['w1'])==e['w1'];latentok &= codec.tensor_hash(cap['latent'])==e['latent'];stateok &= codec.tensor_hash(cap['feature'])==e['feature'] and state_hash(base.snap_cpu(dec2))==e['state'];rgbok &= codec.tensor_hash(out)==e['rgb']
  z,n,m,p=basic(out,frames[fi]);total_sse+=z;total_pixels+=n;rows.append({'frame':fi,'actual_qp':aq,'actual_bytes':len(base.packet(pay,aq,is_i)),'pixel_SSE':z,'aggregate_pixels':n,**base.frame_metric(out,frames[fi],metrics)})
 passed=len(payloads)==64 and cum[-1]==path.stat().st_size and maps and predok and cdfok and symok and latentok and stateok and rgbok;tot['headers_bytes']=len(stream)-sum(tot.values())
 if sum(tot.values())!=len(stream):raise RuntimeError('rate accounting mismatch')
 avg=base.average_metrics(rows);agg=total_sse/total_pixels;summary={**avg,'total_pixel_SSE':total_sse,'total_pixels':total_pixels,'aggregate_MSE':agg,'sequence_PSNR':-10*math.log10(agg),'mean_frame_PSNR':sum(float(r['PSNR']) for r in rows)/len(rows)}
 audit={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'granularity':gran,'trajectory_id':state['trajectory_id'],'selection_label':label,'frames':len(payloads),'mode_map_decode':maps,'predictor_equality':predok,'correction_CDF_equality':cdfok,'correction_symbols_decode':symok,'w1_hash_equal':symok,'decoded_latent_equal':latentok,'DPB_hash_equal':stateok,'RGB_hash_equal':rgbok,'RGB_max_abs_diff':0.0 if rgbok else '', 'bytes_consumed_equal':cum[-1]==path.stat().st_size,'decode_status':'PASS' if passed else 'FAIL','bitstream_path':str(path),'bitstream_sha256':sha(path)}
 return audit,rows,summary,tot

@torch.no_grad()
def process_cell(v,qp,device):
 t=tag(v,qp);done=ROOT/'parts'/f'{t}_done.json'
 if done.exists() and json.loads(done.read_text()).get('status')=='PASS':print('reuse',t,flush=True);return
 brow,bframes=baseline(v,qp);frames=base.read_frames(v,device);metrics=base.init_metric_models(device);allfinal=[];alltraj=[];allaudit=[];allframes=[];allcov=[]
 for gran,tile in CFG['granularities'].items():
  props,c1=build_proposals(v,frames,qp,gran,int(tile),device,metrics);states,c2=search(v,frames,qp,gran,int(tile),device,props,brow,bframes);selected=[]
  b_mse=sum(float(r['MSE']) for r in bframes)/len(bframes)
  requests=[]
  for target in CFG['target_rate_ratios']:
   tb=math.floor(int(brow['baseline_bytes'])*target);feas=[s for s in states if int(s['total_bytes'])<=tb];requests.append((f'TARGET_{target:g}',target,tb,min(feas,key=lambda x:x['total_pixel_SSE']) if feas else None))
  nw=[s for s in states if int(s['total_bytes'])<int(brow['baseline_bytes']) and float(s['aggregate_MSE'])<=b_mse];requests.append(('BEST_RATE_WITH_NONWORSE_SSE','',int(brow['baseline_bytes'])-1,min(nw,key=lambda x:(x['total_bytes'],x['total_pixel_SSE'])) if nw else None))
  ub=[s for s in states if int(s['total_bytes'])<=int(brow['baseline_bytes'])];requests.append(('BEST_SSE_UNDER_BASELINE_RATE','',int(brow['baseline_bytes']),min(ub,key=lambda x:x['total_pixel_SSE']) if ub else None));cache={}
  for label,target,tb,s in requests:
   if s is None:
    allfinal.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'granularity':gran,'selection_label':label,'target_rate_ratio':target,'baseline_bytes':brow['baseline_bytes'],'target_bytes':tb,'selection_type':'TARGET_NOT_REACHED'});continue
   if s['trajectory_id'] not in cache:
    cache[s['trajectory_id']]=replay_audit(v,frames,qp,gran,int(tile),device,metrics,props,s,label);allaudit.append(cache[s['trajectory_id']][0]);allframes += [dict(dataset=v['dataset'],video=v['name'],video_id=v['video_id'],qp=qp,granularity=gran,trajectory_id=s['trajectory_id'],**r) for r in cache[s['trajectory_id']][1]]
   audit,fr,sm,tot=cache[s['trajectory_id']];allfinal.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'granularity':gran,'selection_label':label,'target_rate_ratio':target,'baseline_bytes':brow['baseline_bytes'],'target_bytes':tb,'selected_bytes':s['total_bytes'],'selected_rate_ratio':int(s['total_bytes'])/int(brow['baseline_bytes']),'selection_type':'BASELINE' if s.get('is_baseline_path') else 'REPLACEMENT',**tot,**sm,'decode_status':audit['decode_status'],'bitstream_path':audit['bitstream_path'],'bitstream_sha256':audit['bitstream_sha256']})
  ids={x['trajectory_id'] for x in allaudit if x['granularity']==gran}
  for s in states:alltraj.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'granularity':gran,**{k:v for k,v in s.items() if k!='history'},'mode_history_json':json.dumps(s['history']),'is_selected':s['trajectory_id'] in ids})
  allcov.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'granularity':gran,**c1,**c2})
 write_csv(ROOT/'parts'/f'{t}_baseline_cells.csv',[brow]);write_csv(ROOT/'parts'/f'{t}_final_streams.csv',allfinal);write_csv(ROOT/'parts'/f'{t}_final_trajectories.csv',alltraj);write_csv(ROOT/'parts'/f'{t}_bitstream_audit.csv',allaudit);write_csv(ROOT/'parts'/f'{t}_frame_metrics.csv',allframes);write_csv(ROOT/'parts'/f'{t}_search_coverage.csv',allcov)
 ok=all(a['decode_status']=='PASS' for a in allaudit);done.write_text(json.dumps({'status':'PASS' if ok else 'FAIL','cell':t,'G8_completed':True,'G16_completed':True,'selected_streams':len(allaudit),'selected_decode_pass':sum(a['decode_status']=='PASS' for a in allaudit)},indent=2));
 if not ok:raise RuntimeError('selected audit failed')

def main():
 faulthandler.enable();ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--cells',required=True);ap.add_argument('--worker',default='run');a=ap.parse_args();sys.stdout=Tee(sys.stdout,ROOT/'logs'/f'gpu{a.gpu}_{a.worker}.log');sys.stderr=Tee(sys.stderr,ROOT/'logs'/f'gpu{a.gpu}_{a.worker}.stderr.log');torch.manual_seed(CFG['random_seed']);device=torch.device(f'cuda:{a.gpu}');torch.cuda.set_device(device);lookup={(v['dataset'],int(v['video_id'])):v for v in MAN['videos']}
 for spec in a.cells.split(','):
  ds,vid,q=spec.split(':');v=lookup[(ds,int(vid))]
  try:process_cell(v,int(q),device)
  except Exception as e:(ROOT/'parts'/f'{tag(v,int(q))}_FAILED.json').write_text(json.dumps({'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()},indent=2));print('FAILED',spec,repr(e),flush=True)
  gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
