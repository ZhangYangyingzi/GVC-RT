"""Verify and import logical N0/Q0 baselines without new optimizer updates."""
from util import *
import shutil,csv

def main():
 import torch
 torch.set_num_threads(2)
 plan=load(P/'train_plan.json');assert sha(P/'train_plan.json')==sha(V2/'train_plan.json')
 csvrows=list(csv.DictReader((V2/'training_log.csv').open()));records=[];conditions={};checkpoints={};train=[]
 for logical,old in [('N0','A'),('Q0','H11')]:
  rows=[r for r in csvrows if r['branch']==old];assert len(rows)==300 and [int(r['step']) for r in rows]==list(range(1,301))
  assert all(int(r['optimizer_internal_step'])==int(r['step']) for r in rows)
  train.extend([{**r,'branch':logical,'training_origin':'reused_v1_A' if logical=='N0' else 'reused_v2_H11'} for r in rows]);checkpoints[logical]={}
  for step in (0,100,300):
   path=legacy_checkpoint(logical,step);st=torch.load(path,map_location='cpu',weights_only=False)
   assert st['step']==step and st.get('cursor',step)==step and st['plan_hash']==plan['canonical_plan_sha256']
   assert st['branch']==old and not st.get('V')
   assert {int(s['step']) for s in st['optimizer']['state'].values()}==({step} if step else set())
   assert {g['name']:g['lr'] for g in st['optimizer']['param_groups']}==({'G':1e-6} if logical=='N0' else {'G':1e-6,'H':1e-4})
   assert all(t.isfinite().all() for t in st['G'].values())
   checkpoints[logical][str(step)]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,source_branch=old,updates=step,reused=True)
   del st
  conditions[logical]=dict(source_method=old,new_updates=0,reused_updates=300,training_csv_path=str(V2/'training_log.csv'),training_csv_sha256=sha(V2/'training_log.csv'),data_manifest_sha256=sha(P/'data_manifest.json'),train_plan_sha256=sha(P/'train_plan.json'),optimizer_verified=True,objective='LPIPS + 0.1383589248954108 L1 + beta(schedule_s1p0)*estimated_bpp',forward_precision='native v2 codec FP16, G/H FP32',skip_fix_for_N0='G-only with frozen pre-G path: hard forward and G derivative unchanged; initial full-sequence parity separately checked',Phi='original frozen probability networks',visual_condition=False)
  for source in sorted((V2/'server_artifacts/evaluation').glob(old+'_*.json')):
   r=load(source);assert r['status']=='PASS' and r['frames']==64 and r['IP']==-1 and r['independent_decode'] and r['all_64_reference_states_synchronized']
   cp=legacy_checkpoint(logical,r['step']);assert sha(cp)==r['checkpoint_sha256']
   stream=Path(r['stream_path']);assert sha(stream)==r['stream_sha256'] and stream.stat().st_size==r['real_total_bytes']
   key=f"{logical}_s{r['step']}_{r['sequence']}_q{r['external_qp']}"
   fr=V2/'server_artifacts/evaluation_frames'/(source.stem+'.csv');tr=V2/'server_artifacts/evaluation_transitions'/(source.stem+'.csv')
   fs=list(csv.DictReader(fr.open()));ts=list(csv.DictReader(tr.open()));assert len(fs)==64 and len(ts)==63
   assert sum(int(f['real_bytes']) for f in fs)==r['real_total_bytes'] and sum(int(f['payload_bytes']) for f in fs[1:])*8==r['actual_P_payload_bits']
   for folder,path in [('evaluation_frames',fr),('evaluation_transitions',tr)]:
    dest=P/'server_artifacts'/folder/(key+'.csv');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
   result=dict(r,branch=logical,measurement='reused_v2_verified_evaluation',reuse_source=str(source),reuse_source_sha256=sha(source),reuse_training_updates=300,new_training_updates=0,signature=signature(),frames_sha256=sha(fr),transitions_sha256=sha(tr),legacy_latent_assertion_recorded=r.get('all_P_latents_qmaps_exact') is True)
   dump(P/'server_artifacts/evaluation'/(key+'.json'),result)
   records.append(dict(method=logical,step=r['step'],sequence=r['sequence'],QP=r['external_qp'],source_path=str(source),source_sha256=sha(source),checkpoint_sha256=sha(cp),stream_sha256=sha(stream),frame_csv_sha256=sha(fr),transition_csv_sha256=sha(tr),all64_reference_check=True,legacy_P_latent_qmap_flag=r.get('all_P_latents_qmaps_exact','not_recorded'),no_new_training=True))
   if r['step']==300 and r['external_qp']==2:
    src=V2/'previews'/f'{old}_s300_{r["sequence"]}_q2_frame15.png'
    if src.exists():shutil.copy2(src,P/'previews'/f'{logical}_s300_{r["sequence"]}_q2_frame15.png')
    gt=V2/'previews'/f'{r["sequence"]}_gt_frame15.png'
    if gt.exists():shutil.copy2(gt,P/'previews'/gt.name)
 dump(P/'reuse_provenance.json',dict(status='PASS',conditions=conditions,evaluation_records=records,checkpoint_index=checkpoints,notes='Legacy N0 tasks with no explicit P-latent/qmap flag retain not_recorded; no assertion is fabricated. Their all64 reference synchronization and source/checkpoint/stream/metric hashes were verified in v2. New initial full-sequence identity is checked separately.'))
 write(P/'reference_v2/reused_training.csv',train)
 print('REUSE_PASS',len(records),flush=True)
if __name__=='__main__':main()
