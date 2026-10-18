"""Save an explicitly partial checkpoint without changing the running scheduler."""
from base import *

def main():
 receipts=[load(p) for p in sorted((ROOT/'manifests/receipts').glob('*.json'))]
 passed=[r for r in receipts if r['status']=='PASS']
 rows=[]
 for r in passed:
  j=r['job'];x=r['result'];a=r['rate']
  rows.append(dict(id=j['id'],kind=j['kind'],dataset=j['dataset'],QP=j['qp'],actual_total_bits=a['actual_total_bits'],actual_bpp=x['bpp'],kbps=x['kbps'],native_codec_bits=a['native_codec_stream_bits'],RANS_payload_bits=a['RANS_payload_bits'],control_header_bits=a['control_header_bits'],LPIPS=x['LPIPS'],DISTS=x['DISTS'],FloLPIPS=x['FloLPIPS'],FID=None,FID_status=x['FID_status'],**r['delta'],encode_ms=r['timings']['encode_ms'],decode_ms=r['timings']['decode_ms'],wall_seconds=r['wall_seconds'],success=True,gpu=r['gpu'],plan=json.dumps(j['plan'])))
 write('metrics/checkpoint_raw_results.csv',rows)
 write('baseline/baseline_summary.csv',[dict(x,original_native_bpp=x['native_codec_bits']/(16*1920*1080),original_native_kbps=x['native_codec_bits']*30/(16*1000)) for x in rows if x['kind']=='baseline'])
 write('baseline/reproducibility.csv',[dict(id=r['job']['id'],**r['reproducibility']) for r in passed if r['job']['kind']=='repeat'])
 local=[x for x in rows if x['kind'] in ('noop','local','joint')]
 if local:write('oracle/local_sensitivity/block_intervention_results.csv',local)
 else:
  p=ROOT/'oracle/local_sensitivity/block_intervention_results.csv'
  if not p.exists():p.write_text('id,kind,dataset,QP,actual_total_bits,actual_bpp,delta_actual_bits,delta_actual_bpp,delta_lpips,delta_dists,delta_flolpips,delta_fid,success\n')
 cfg=load(ROOT/'configs/experiment.json');gate=ROOT/'audit/local_control_correctness.json'
 dump('oracle/local_sensitivity/intervention_manifest.json',dict(checkpoint_only=True,run_status=load(ROOT/'run_status.json'),valid_real_rate_control=load(gate)['real_local_rate_control'] if gate.exists() else None,planned_single_interventions=64,blocks=cfg['blocks'],levels=cfg['local_levels'],completed=[r['job'] for r in passed if r['job']['kind'] in ('noop','local','joint')],raw_symbols='oracle/local_sensitivity/symbols/',full_tensors='artifacts/runs/; paths and SHA256 in each worker receipt'))
 dump('oracle/local_sensitivity/failed_interventions.json',dict(checkpoint_only=True,failed_tasks=[r for r in receipts if r['status']!='PASS'],previous_failed_attempts=[dict(path=str(p),record=load(p)) for p in (ROOT/'logs/failed_attempts').glob('*.json')]))
 dump('oracle/local_sensitivity/actual_rate_accounting.json',dict(checkpoint_only=True,actual_total_definition='all native codec bytes plus all experimental signalling',stock_original_definition='native codec stream without experiment signalling',fixed_header_bytes=10,decision_bytes=11,records={r['job']['id']:r['rate'] for r in passed}))
 dump('manifests/binary_artifacts_index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'artifacts').rglob('*')) if p.is_file() and p.suffix!='.tmp'])
 exits=[]
 for p in (ROOT/'logs/stage_exits').glob('*.json'):exits.extend(load(p))
 dump('logs/GPU_task_exit_status.json',dict(checkpoint_only=True,all_final_tasks_finished=False,stage_worker_exits=exits,standalone_decoder_exits=[load(p) for p in (ROOT/'logs/standalone_exits').glob('*.json')],live_status='run_status.json'))
 dump('manifests/checkpoint_status.json',dict(checkpoint_only=True,successful_receipts=len(passed),failed_receipts=len(receipts)-len(passed),state=load(ROOT/'run_status.json'),time=time.time()))
 print('Checkpoint:',len(passed),'successful receipts; experiment remains',load(ROOT/'run_status.json')['status'])

if __name__=='__main__':main()
