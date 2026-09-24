#!/usr/bin/env python3
import csv,hashlib,json,math,sys,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];V13=ROOT.parent/'gvcrt_v13_generator_aware_mixed_precision';V131=ROOT.parent/'gvcrt_v13_1_structured_mixed_precision'
for p in (ROOT,REPO,V13,V131):sys.path.insert(0,str(p))
import run_v13 as base
import v14_codec as codec
CFG=json.loads((ROOT/'config.json').read_text());MAN=json.loads((ROOT/'manifest.json').read_text());SPS=base.SPS

def write_csv(path,rows):
 rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else []
 with open(path,'w',newline='') as f:
  if fields:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha(b):return hashlib.sha256(b).hexdigest()
def parse_any(path):
 payloads=[];path=Path(path)
 with path.open('rb') as f:
  while f.tell()<path.stat().st_size:
   h=base.read_header(f)
   if h['nal_type']==base.NalType.NAL_SPS:base.read_sps_remaining(f,h['sps_id']);continue
   qp,payload=base.read_ip_remaining(f);payloads.append((h['nal_type']==base.NalType.NAL_I,qp,payload))
 return payloads
@torch.no_grad()
def run_branch(video,frames,qp,device,name):
 iframe,enc,dec=base.load_triplet(device);stream,rgb0,ip=base.init_sequence(iframe,enc,dec,frames[0],qp);expected=[];rows=[];symbols=[];base_ref=[]
 expected.append({'rgb':codec.tensor_hash(rgb0),'state':base.state_hash(base.snap_cpu(dec))})
 for fi in range(1,8):
  aq=enc.shift_qp(qp,base.INDEX_MAP[fi%8]);prep=codec.prepare_frame(enc,frames[fi],aq,None,8);n=len(prep['modes'])
  action=codec.ACTION[name];modes=[action]*n;cand=codec.apply_actions(enc,prep,modes,8,True)
  if name=='FULL':payload,comp=codec.encode_original(enc,prep,aq);out,cap=codec.decode_replacement(dec,payload,SPS,aq,8);cand=prep
  else:payload,comp=codec.encode_replacement(enc,cand,aq,8,global_action=action);out,cap=codec.decode_replacement(dec,payload,SPS,aq,8)
  predictor_equal=True if name=='FULL' else torch.equal(cap['q_pred'],cand['q_pred'])
  if name!='FULL' and (not predictor_equal or not torch.equal(cap['w1'],cand['w1']) or not torch.equal(out,cand['rgb'])):raise RuntimeError(f'{name} mismatch frame {fi}')
  enc.add_ref_frame(cand['feature'],None);pkt=base.packet(payload,aq);stream+=pkt
  mse,psnr,_=base.basic_metrics(out,frames[fi]);rows.append({'branch':name,'frame':fi,'payload_bytes':len(payload),'packet_bytes':len(pkt),'MSE':mse,'PSNR':psnr,'w1_hash':codec.tensor_hash(cap['w1']),'rgb_hash':codec.tensor_hash(out),'state_hash':base.state_hash(base.snap_cpu(dec)),'predictor_equal':predictor_equal})
  expected.append({'rgb':codec.tensor_hash(out),'w1':codec.tensor_hash(cap['w1']),'state':base.state_hash(base.snap_cpu(dec))})
  qt,qp0=prep['q_true'],prep['q_pred'];active=prep['s1']!=0
  for ch in range(qt.shape[1]):
   sel=active[:,ch];e=(qt[:,ch][sel]-qp0[:,ch][sel]).float();symbols.append({'branch':name,'frame':fi,'channel':ch,'num_symbols':int(e.numel()),'encoder_decoder_predictor_equal':predictor_equal,'exact_match_rate':float((e==0).float().mean()) if e.numel() else 1.,'MAE':float(e.abs().mean()) if e.numel() else 0.,'RMSE':float(e.square().mean().sqrt()) if e.numel() else 0.})
 path=ROOT/'bitstreams'/f'micro_{name}.bin';path.write_bytes(stream)
 # Fresh independent decode from the saved stream.
 payloads=parse_any(path);iframe2,_,dec2=base.load_triplet(device);dec2.clear_dpb();dec2.set_curr_poc(0);ok=True
 for fi,(is_i,aq,payload) in enumerate(payloads):
  if is_i:out=iframe2.decompress(payload,SPS,aq)['x_hat'];dec2.add_ref_frame(None,out)
  else:out,cap=codec.decode_replacement(dec2,payload,SPS,aq,8);ok &= codec.tensor_hash(cap['w1'])==expected[fi]['w1']
  ok &= codec.tensor_hash(out)==expected[fi]['rgb'] and base.state_hash(base.snap_cpu(dec2))==expected[fi]['state']
 return {'branch':name,'total_bytes':len(stream),'bitstream_sha256':sha(stream),'independent_decode_pass':ok,'bitstream_path':str(path)},rows,symbols,expected

def main():
 device=torch.device('cuda:7');torch.cuda.set_device(device);torch.manual_seed(CFG['random_seed']);v=[x for x in MAN['videos'] if x['dataset']=='fresh_ulong' and int(x['video_id'])==0][0];frames=base.read_frames(v,device)[:8];before=[];allrows=[];allsym=[];exp={}
 for name in ('FULL','PRED_ONLY','CORR1','CORR2','CORR4'):
  summary,rows,symbols,expected=run_branch(v,frames,3,device,name);before.append(summary);allrows+=rows;allsym+=symbols;exp[name]=expected;print('micro',name,summary['total_bytes'],summary['independent_decode_pass'],flush=True)
 full=next(x for x in before if x['branch']=='FULL');ref=V131/'bitstreams'/'fresh_ulong_00_qp3_baseline.bin';_,cum=base.parse_stream(ref);ref8=ref.read_bytes()[:cum[7]]
 def changed(name,key):return any(exp[name][i].get(key)!=exp['FULL'][i].get(key) for i in range(1,8))
 checks={'baseline_identity_pass':(ROOT/'bitstreams'/'micro_FULL.bin').read_bytes()==ref8,
  'FULL_pass':full['independent_decode_pass'],'PRED_pass':changed('PRED_ONLY','w1') and changed('PRED_ONLY','rgb') and next(x for x in before if x['branch']=='PRED_ONLY')['independent_decode_pass'],
  'CORR1_identity_pass':not changed('CORR1','w1') and not changed('CORR1','rgb') and not changed('CORR1','state') and next(x for x in before if x['branch']=='CORR1')['independent_decode_pass'],
  'CORR2_pass':changed('CORR2','w1') and changed('CORR2','rgb') and next(x for x in before if x['branch']=='CORR2')['independent_decode_pass'],
  'CORR4_pass':changed('CORR4','w1') and changed('CORR4','rgb') and next(x for x in before if x['branch']=='CORR4')['independent_decode_pass'],
  'encoder_decoder_predictor_equal':all(r['encoder_decoder_predictor_equal'] for r in allsym)}
 checks['status']='PASS' if all(checks.values()) else 'FAIL';(ROOT/'micro_test.json').write_text(json.dumps(checks,indent=2)+'\n');write_csv(ROOT/'micro_streams.csv',before);write_csv(ROOT/'micro_symbol_audit.csv',allsym);write_csv(ROOT/'sanity'/'micro_frame_metrics.csv',allrows);print(json.dumps(checks,indent=2));
 if checks['status']!='PASS':raise SystemExit(2)
if __name__=='__main__':
 try:main()
 except Exception:
  (ROOT/'logs'/'micro_traceback.log').write_text(traceback.format_exc());raise
