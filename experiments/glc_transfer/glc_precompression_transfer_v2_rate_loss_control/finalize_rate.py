"""Mechanical, training-only rate checks and common backend freeze."""
from preflight import *
from data import load
from bootstrap import V1
def main():
 import numpy as np
 from evaluate import csvwrite
 from run_pipeline import parallel,stage
 expected_signature=dict(plan=sha(R/'plans/rate_audit.json'),worker=sha(R/'rate_audit.py'),runtime=sha(R/'runtime.py'),source=sha(V1/'final_P/C_latent_P_1000.pt'))
 stale=[]
 for i in range(40):
  for m in ('A_original','B_rgb_P','C_latent_P'):
   path=R/f'audits/rate_pairs/{i:02d}_{m}.json'
   if not path.exists() or load(path).get('signature')!=expected_signature:stale.append((f'rate_pair_{i:02d}_{m}',['rate_audit.py','--index',str(i),'--method',m]))
 if stale:
  save('audits/rate_audit_validation_refresh.json',dict(reason='explicit hyper/categorical symbol hash coverage added; refresh older audit implementation only',jobs=[j[0] for j in stale],formal_training_or_evaluation_repeated=False));parallel(stale,need=5000,max_workers=4,per_gpu=2)
 from src.entropy_coder import parse_gop_stream
 from src.entropy_coder.bitstream import parse_frame_packet,_FRAME_HEADER,_GOP_HEADER,_FRAME_SIZE
 crosscheck=[]
 for d in load(V1/'eval_manifest.json'):
  for method in ('A_original','B_rgb_P','C_latent_P'):
   for q in (0,28,63):
    point=V1/f'results/points/{d}_0_{method}_{q}.json';old=load(point);stream=next(Path(p) for p in old['artifacts'] if p.endswith('.glc'));assert sha(stream)==old['artifacts'][str(stream)]
    raw=stream.read_bytes();h,w,packets=parse_gop_stream(raw);assert len(packets)==64 and len(raw)==old['result']['bytes'];details=[parse_frame_packet(p) for p in packets]
    payload=sum(len(p['y_payload']) for p in details);hyper=sum((p['z_height']*p['z_width']*14+7)//8 for p in details);header=_GOP_HEADER.size+64*(_FRAME_SIZE.size+_FRAME_HEADER.size);assert len(raw)==payload+hyper+header
    crosscheck.append(dict(dataset=d,method=method,q=q,width=w,height=h,frames=64,complete_bytes=len(raw),payload_bytes=payload,hyper_bytes=hyper,known_header_bytes=header,source=str(point),source_sha256=sha(point),stream_sha256=sha(stream),scope='existing full-resolution byte accounting only; excluded from backend selection and beta calibration'))
 save('audits/full_resolution_stream_crosscheck.json',dict(status='PASS',completed=len(crosscheck),rows=crosscheck,additional_encodes=0,training_data_calibration_unaffected=True,estimated_bits='not present in frozen v1 full-resolution receipts; not invented'))
 # Also handles a driver already running before gradient jobs were added.
 jobs=[(f'component_{i:02d}_{m}',['gradient_audit.py','--index',str(i),'--method',m]) for i in range(40) for m in ('B_rgb_P','C_latent_P') if not (R/f'audits/component_gradients/{i:02d}_{m}.json').exists()]
 if jobs:parallel(jobs,need=10500,max_workers=4)
 pairs=[load(R/f'audits/rate_pairs/{i:02d}_{m}.json') for i in range(40) for m in ('A_original','B_rgb_P','C_latent_P')]
 assert all(p['status']=='PASS' and p['independent_decode'] and p['symbols_and_reference_history_identical'] for p in pairs)
 gradient=[load(R/f'audits/component_gradients/{i:02d}_{m}.json') for i in range(40) for m in ('B_rgb_P','C_latent_P')];assert all(p['status']=='PASS' for p in gradient)
 audit_paths=sorted(list((R/'audits/rate_pairs').glob('*.json'))+list((R/'audits/component_gradients').glob('*.json')))
 assert len(audit_paths)==200
 save('audits/rate_audit_index.json',dict(status='PASS',paired_completed=len(pairs),paired_expected=120,gradient_completed=len(gradient),gradient_expected=80,artifacts={str(p.relative_to(R)):sha(p) for p in audit_paths}))
 rows=[];mismatch=[]
 for p in pairs:
  official=sum(f['official_y_bits'] for f in p['frames']);table=sum(f['coder_table_y_bits'] for f in p['frames']);payload=p['payload_bits'];escape=sum(f['escape_symbols'] for f in p['frames'])
  r=dict(index=p['index'],domain=p['domain'],method=p['method'],q=p['q'],official_y_bits=official,coder_table_y_bits=table,payload_bits=payload,known_hyper_bits=p['hyper_bits'],frame_header_bits=p['frame_header_bits'],GOP_header_and_lengths_bits=p['GOP_header_and_lengths_bits'],complete_bytes=p['complete_bytes'],payload_minus_official=payload-official,payload_minus_table=payload-table,table_official_relative_difference=(table-official)/max(official,1),escape_symbols=escape,scale_above_table=p['coder_statistics']['scale_above_table'],rate_axis='bits per identical 64 x 256 x 256 training crop; headers separated',opaque_overhead='y rANS flush and byte alignment remain combined')
  # A finite-precision difference alone is not evidence for changing a backend.
  # Require measured >1% systematic symbol cost disagreement, payload agreement
  # improvement, and an observed truncation/escape event in at least two windows.
  if abs(table-official)>.01*max(official,1) and abs(payload-table)<abs(payload-official) and (escape>0 or r['scale_above_table']>0):mismatch.append(r)
  rows.append(r)
 changed=len({r['index'] for r in mismatch})>=2
 backend='coder_cdf_forward_continuous_backward' if changed else 'official_verified'
 save('audits/rate_backend_decision.json',dict(status='PASS',backend=backend,changed=changed,paired_points_completed=len(pairs),paired_points_expected=120,gradient_audits_completed=len(gradient),gradient_audits_expected=80,quantization_and_reference_mismatch=False,initial_reference_mismatch_record='audits/reference_first_difference.json',reference_forward_alignment_changed=True,reference_fix='v2-only exact categorical codebook forward; identity STE backward preserved',structural_probability_mismatch_reproduced=changed,evidence=mismatch,rule='at least two training windows: >1% symbol-cost disagreement plus closer table cost to actual payload and measured escapes/table-range overflow; finite precision alone insufficient',backward='continuous table-range-clamped Gaussian mass approximation; exact CDF/escape symbol costs in forward; opaque per-stream flush remains recorded separately' if changed else 'unchanged v1 official Gaussian estimator/backward, on repaired categorical forward',H00_label='continuation on common reference-aligned base; not strict v1 numerical-training-path continuation',coder_unchanged=True,v1_source_sha256=sha(V1/'runtime.py')))
 csvwrite(R/'audits/paired_rate_summary.csv',rows)
 if changed:stage('calibrate_backend',['calibrate_backend.py'])
 else:save('calibration/final.json',load(V1/'calibration/final.json'))
 protocol=load(R/'protocol.json');protocol.update(rate_backend=backend,rate_backend_changed=changed,reference_forward_alignment_changed=True,categorical_STE='exact codebook lookup forward, unchanged identity backward, v2-only',beta_sha256=sha(R/'calibration/final.json'),training_plan_sha256=sha(R/'plans/shared.json'));save('protocol.json',protocol)
 save('audits/rate_freeze.json',dict(status='PASS',backend=backend,changed=changed,beta_sha256=sha(R/'calibration/final.json'),backend_code_sha256=sha(R/'rate_backend.py'),training_plan_sha256=sha(R/'plans/shared.json'),rate_audit_plan_sha256=sha(R/'plans/rate_audit.json'),old_beta=load(V1/'calibration/final.json')['beta'],new_beta=load(R/'calibration/final.json')['beta'],shared_by_all_four_branches=True))
 print('RATE FROZEN',backend,'changed',changed,flush=True)
if __name__=='__main__':main()
