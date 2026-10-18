"""Hash-validated baseline reuse; no writes into v1."""
from preflight import *
from bootstrap import V1
from data import load
def main():
 from evaluation_stage import expected_signatures
 assert sha(R/'eval_manifest.json')==sha(V1/'eval_manifest.json')
 old=load(V1/'protocol.json');new=load(R/'protocol.json')
 for k in ('clip_frames','q_grid','fa_index_map','reset_ref_y_interval','I_bypasses_P','precision','test_frames'):assert old[k]==new[k],k
 assert load(R/'audits/checkpoint_audit.json')==load(V1/'audits/checkpoint_audit.json')
 for path,h in load(R/'audits/v1_source_inventory.json')['files'].items():assert sha(path)==h
 assert load(R/'audits/pretraining_checks.json')['status']=='PASS'
 assert load(R/'audits/metric_dependencies.json')['status']=='PASS'
 for x in load(R/'audits/reference_dependencies.json')['files']:assert sha(x['source'])==x['sha256']
 signatures=expected_signatures();rows=[]
 for file in sorted((V1/'results/points').glob('*.json')):
  old_receipt=load(file);m=old_receipt['result']['method']
  if m not in ('A_original','C_latent_P'):continue
  assert old_receipt['status']=='PASS' and old_receipt['independent_decode'] and old_receipt['frozen_unchanged']
  assert old_receipt['signature']['runtime']==sha(V1/'runtime.py') and old_receipt['signature']['evaluation']==sha(V1/'evaluate.py')
  assert all(Path(p).exists() and sha(p)==h for p,h in old_receipt['artifacts'].items())
  method='A_original' if m=='A_original' else 'v1_C_latent_P'
  if m!='A_original':assert old_receipt['signature']['P_sha256']==sha(V1/'final_P/C_latent_P_1000.pt')
  r=dict(old_receipt['result'],method=method);dest=f"results/points/{r['dataset']}_{r['video']}_{method}_{r['q']}.json"
  provenance=dict(path=str(file),sha256=sha(file),original_signature=old_receipt['signature'],manifest_identical=True,precision_and_codec_identical=True,metrics_and_GT_unchanged=True,adapter_step0_strict_replay_audit=sha(R/'audits/pretraining_checks.json'))
  save(dest,dict(old_receipt,signature=signatures[method],result=r,reuse=provenance));rows.append(dict(destination=dest,**provenance))
 expected=sum(len(v) for v in load(R/'eval_manifest.json').values())*20;assert len(rows)==expected
 for d in load(R/'eval_manifest.json'):
  for method,source in [('A_original','A_original'),('v1_C_latent_P','C_latent_P')]:
   for i in (16,32,63):
    src=V1/'previews'/d/f'{source}_{i}.png';target=R/'previews'/d/f'{method}_{i}.png';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(src.read_bytes())
 save('audits/baseline_reuse.json',dict(status='PASS',completed=len(rows),expected=expected,records=rows,source_untouched=True));print('REUSED',len(rows),flush=True)
if __name__=='__main__':main()
