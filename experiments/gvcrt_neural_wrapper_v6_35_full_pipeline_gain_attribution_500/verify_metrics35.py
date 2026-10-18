import csv,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;V24=R.parent/'gvcrt_neural_wrapper_v6_24_projected_adversarial_control';V2B=R.parent/'gvcrt_neural_wrapper_v6_2b_final_generalization_test';V34=R.parent/'gvcrt_neural_wrapper_v6_34b_p_joint_fd_trend_500'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
expected=json.load(open(V34/'evaluation_config.json'))['metric_module_hashes'];audit=V2B/'metric_implementation_audit.json';audit_sha=sha(audit);assert audit_sha=='d3608bf06010e1e5cbac598094ea12994670518b105feab4e662a5e5d456a962'
checks=[]
for ds in ('uvg_holdout','hevc_b'):
 p=V24/'results/ip_-1'/ds/'per_sequence_all_qp.csv'
 for r in csv.DictReader(p.open()):
  if r['method']!='original' or int(r['QP']) not in (0,5,9):continue
  dep=json.loads(r.get('metric_dependency_verification') or '{}')
  runtime=json.loads(r.get('runtime_audit') or '{}')
  if runtime.get('metric_module_hashes'):assert runtime['metric_module_hashes']==expected,(ds,r['sequence'],runtime['metric_module_hashes'],expected)
  if dep:
   assert dep.get('status')=='PASS' and dep.get('source_metric_audit_sha256')==audit_sha and sha(dep['source_metric_audit'])==audit_sha
  if r.get('metric_implementation_sha256'):assert r['metric_implementation_sha256']==audit_sha
  assert runtime.get('metric_module_hashes') or dep.get('status')=='PASS' or r.get('metric_implementation_sha256')==audit_sha
  checks.append({'dataset':ds,'sequence':r['sequence'],'QP':int(r['QP']),'dependency_audit_sha256':audit_sha,'runtime_hashes_present':bool(runtime.get('metric_module_hashes')),'runtime_hashes_match':not runtime.get('metric_module_hashes') or runtime['metric_module_hashes']==expected})
assert len(checks)==21
(R/'audits/metric_reuse_verification.json').write_text(json.dumps({'status':'PASS','v24_original_points':len(checks),'source_metric_audit':str(audit),'source_metric_audit_sha256':audit_sha,'v634b_runtime_metric_module_hashes':expected,'records':checks},indent=2)+'\n')
print('metric reuse verification PASS',len(checks))
