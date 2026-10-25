from util import *
import shutil

def main():
 import torch
 assert load(V2/'completion.json')['status']=='COMPLETE' and load(V2/'preflight.json')['status']=='PASS'
 dependencies=load(V2/'dependency_manifest.json')
 for p,h in dependencies.items():assert sha(p)==h,(p,'historical dependency mismatch')
 for n in ['data_manifest.json','train_plan.json','evaluation_sources.json']:assert sha(P/n)==sha(V2/n)
 for p,e in load(P/'weight_manifest.json').items():assert Path(p).is_file() and sha(p)==e['sha256'] and Path(p).stat().st_size==e['bytes']
 for b in ['N0','Q0']:
  for s in (0,100,300):legacy_checkpoint(b,s)
 if (P/'preflight.json').exists():
  assert load(P/'preflight.json')['status']=='PASS' and load(P/'preflight.json')['signature']==signature()
  print('SOURCE_ALREADY_PREPARED; hashes verified and committed state preserved',flush=True);return
 gpath=Path(next(p for p in load(P/'weight_manifest.json') if Path(p).name=='G_basic_100.pt'))
 cp=load(V2/'checkpoint_index.json')['H11']['0'];h=torch.load(cp['path'],map_location='cpu',weights_only=False)
 assert h['step']==0 and h['branch']=='H11'
 dump(P/'initialization_manifest.json',dict(status='VERIFIED_SOURCES',G_bridge=dict(path=str(gpath),sha256=sha(gpath)),H=dict(path=cp['path'],sha256=cp['sha256'],source_step=0,identity_multiplier=1),Phi=dict(source='original codec probability network state',weight_manifest_sha256=sha(P/'weight_manifest.json')),common_start=True,no_step300_initialization=True))
 plan=load(P/'train_plan.json');assert len(plan['rows'])==300
 sources={row['path']:row['source_sha256'] for row in plan['rows']}
 for p,h in sources.items():assert sha(p)==h
 dump(P/'audits/inherited_v2_preflight.json',dict(status='PASS',source_path=str(V2/'preflight.json'),source_sha256=sha(V2/'preflight.json'),runtime_sha256=sha(V2/'runtime.py'),train_sha256=sha(V2/'train.py'),inherited_checks='skip STE, two-pass coordinates, synthetic skip/clamp/saturation, independent decode; no H00/H10/H01 rerun',plan_sha256=sha(P/'train_plan.json'),manifest_sha256=sha(P/'data_manifest.json'),source_v2_commit='3f9fae83a8be672c4c4999e09b7c4b08bf4a8fb4'))
 # Frozen historical files, not copied mutable completion state.
 dependencies.update({str(V2/n):sha(V2/n) for n in ['PROTOCOL.md','checkpoint_index.json','delivery_manifest.json','completion.json','preflight.json','evaluation.csv','effective_runtime_audit.json','initialization_audit.json']})
 dump(P/'dependency_manifest.json',dependencies)
 dump(P/'run_status.json',dict(status='PREPARED',new_updates={'N1':0,'Q1':0},reused_updates={'N0':300,'Q0':300}))
 print('SOURCE_PREPARE_PASS',len(sources),flush=True)
if __name__=='__main__':main()
