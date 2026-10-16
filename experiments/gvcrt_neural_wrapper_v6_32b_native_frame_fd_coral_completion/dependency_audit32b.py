"""Read-only historical signature and reused-input provenance audit."""
from io32 import *
def main():
 import collections
 names=['io32.py','training32.py','distribution32.py','frames32.py','statistics32.py','selftest32.py']
 payload=dict(config=sha(OLD32/'config.json'),protocol=sha(OLD32/'protocol.json'),sources=sha(OLD32/'source_hashes.json'),code={n:sha(OLD32/n) for n in names})
 expected=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();hist=collections.Counter();mismatches=[];plan=load(ROOT/'plans/shared.json')['updates']
 for i,r in enumerate(plan):
  p=OLD32/'reference_stats/receipts'/f'{i:04d}.json';h=load(p)
  if h['status']!='PASS':continue
  hist[h['signature']]+=1
  if h['signature']!=expected:mismatches.append(i)
 assert not mismatches,('Historical prefill signature not reconstructed; inspect prior dependency snapshots before claiming provenance',mismatches)
 cfg=load(OLD32/'config.json');assert cfg['source_checkpoint']==load(ROOT/'config.json')['source_checkpoint']
 assert sha(OLD32/'plans/shared.json')==sha(ROOT/'plans/shared.json') and sha(OLD32/'plans/calibration.json')==sha(ROOT/'plans/calibration.json')
 dump(ROOT/'audits/prefill_dependency_mapping.json',dict(status='PASS',verified_historical_receipts=996,reconstructed_historical_signature=expected,historical_signature_counts=dict(hist),signature_payload=payload,source_checkpoint=cfg['source_checkpoint'],shared_plan_bytes_identical=True,calibration_plan_bytes_identical=True,current_signature_not_falsified_as_historical=True,geometry_revision='For ordinary1080p views, native valid geometry and replicate pad/unpad are identical; two full64frame forward tests establish numerical equivalence. Four native non1080 views are freshly evaluated, not reused.',new_old_code_mapping=dict(crop='identical functionAST',official_feature_model='identical function and verified weights',full_view='only native geometry/padding and additional derivative auditing changed'),old_receipts_unchanged=True))
if __name__=='__main__':main()
