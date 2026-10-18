"""Read-only v1 provenance checks and isolated v2 preparation."""
from preflight import *
V1=GLC/'experiments/glc_precompression_transfer_v1_abc'
BRANCHES={'H00_continue':dict(dists=.5,rate_multiplier=1.),'H10_rate2':dict(dists=.5,rate_multiplier=2.),'H01_dists025':dict(dists=.25,rate_multiplier=1.),'H11_rate2_dists025':dict(dists=.25,rate_multiplier=2.)}
SOURCE_SHA='328452a25222cd0870bd24d51c1cc13b6e0dc574745b4c568fa50667bd6587ff'
def main():
 import torch
 from runtime import Sender
 from data import load
 required=['config.json','protocol.json','runtime.py','train.py','evaluate.py','calibration/final.json','final_integrity.json','audits/gradient.json','audits/identity.json','audits/checkpoint_audit.json','train_manifest_ulong.json','train_manifest_uvg.json','eval_manifest.json','dataset_split.json','branches/C_latent_P/checkpoint_index.json','final_P/C_latent_P_1000.pt','final_P/B_rgb_P_1000.pt']
 assert load(V1/'final_integrity.json')['status']=='PASS'
 for key in ('gradient','identity'):assert load(V1/f'audits/{key}.json')['status']=='PASS'
 source=V1/'final_P/C_latent_P_1000.pt';assert sha(source)==SOURCE_SHA
 p=Sender('C_latent_P');s=torch.load(source,map_location='cpu',weights_only=False);assert s['step']==1000;p.load_state_dict(s['P'],strict=True)
 index=load(V1/'branches/C_latent_P/checkpoint_index.json')['1000'];assert index['P_sha256']==SOURCE_SHA and index['module_sha256']==state_hash(p.state_dict())
 snapshots={str(V1/k):sha(V1/k) for k in required}
 for name,h in load(V1/'audits/source_inventory.json')['files'].items():assert sha(name)==h
 for model in load(V1/'audits/checkpoint_audit.json')['models']:assert sha(model['path'])==model['sha256']
 for path in required:
  if path.endswith('.pt'):continue
  target=R/'dependencies/v1'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((V1/path).read_bytes())
 save('audits/source_experiment.json',dict(status='PASS',files=snapshots,source_checkpoint=str(source),source_step=1000,source_sha256=SOURCE_SHA,P_module_sha256=state_hash(p.state_dict()),v1_status=load(V1/'status.json').get('status'),v1_complete_from_integrity=True))
 for key in ('checkpoint_audit','gradient','identity','source_inventory','leakage'):
  save(f'audits/v1_{key}.json',load(V1/f'audits/{key}.json'))
 save('audits/checkpoint_audit.json',load(V1/'audits/checkpoint_audit.json'))
 config=load(V1/'config.json');config.update(seed=20261016,initialization=dict(path=str(source),sha256=SOURCE_SHA,source_step=1000,new_adaptation_step=0,zero_initialization_forbidden=True),branches=BRANCHES,trainable='latent sender P only; GLC and generator frozen/eval',expected_evaluation_points=dict(new=1840,reused=920,total=2760));config.pop('B',None);config['objective']['DISTS']='per-branch: 0.5 for H00/H10; 0.25 for H01/H11';save('config.json',config)
 protocol=load(V1/'protocol.json');protocol.update(experiment=R.name,branches=list(BRANCHES),seed=20261016,source_step=1000,rate_evaluation='real_local_nonofficial_rans',loss='LPIPS + branch.dists*DISTS(require_grad=True) + lambda_rec*L1 + branch.rate_multiplier*beta_base(q)*R',rate_backend='pending paired training-data audit',evaluation_models=['A_original','v1_C_latent_P',*BRANCHES]);save('protocol.json',protocol)
 save('eval_manifest.json',load(V1/'eval_manifest.json'))
 save('status.json',dict(status='RUNNING',phase='preparation',updates={b:0 for b in BRANCHES},evaluation_completed=0))
 print('PASS: v1 source, final C sender and frozen codec hashes verified',flush=True)
if __name__=='__main__':main()
