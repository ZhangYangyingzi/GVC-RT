"""Freeze the specified paired continuation, without modifying any historical file."""
from pathlib import Path
import json,hashlib,shutil,subprocess,time,collections,os
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];PREV=ROOT.parent/'gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion';OLD=ROOT.parent/'gvcrt_neural_wrapper_v6_32_learning_rate_fd_coral_control'
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert not (ROOT/'config.json').exists(),'Existing frozen configuration; do not overwrite'
 for d in ['plans','calibration','reference_stats','branches','tasks','features','bitstreams','raw','evaluation/checkpoints','results','audits','logs']: (ROOT/d).mkdir(parents=True,exist_ok=True)
 for n in ['io32.py','training32.py','frames32.py','distribution32.py','calibration32b.py','adapter32.py','evaluate32.py','driver32b.py','launch32b.py','publish32b.py','reassemble.py','check_once.py','watch_event.py','wait_status.py']:
  shutil.copy2(PREV/n,ROOT/n)
 p=ROOT/'io32.py';s=p.read_text().replace("BRANCHES=('A_C1_lr1','B_C1_lr2','C_C1_lr2_FD','D_C1_lr2_mean_CORAL')","PREV=ROOT.parent/'gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion'\nBRANCHES=('B_C1_lr2','C_C1_lr2_FD','D_C1_lr2_mean_CORAL')")
 s=s.replace("names+=['train.py','calibration32b.py']","names+=['train.py','calibration32b.py','migration33.py']");p.write_text(s)
 for n in ['eval_manifest.json','evaluation_config.json','qp_semantics_audit.json','train_manifest_ulong.json','train_manifest_uvg.json']:
  shutil.copy2(PREV/n,ROOT/n)
 for n in ['shared.json','calibration.json']:shutil.copy2(PREV/'plans'/n,ROOT/'plans'/n)
 assert sha(ROOT/'plans/shared.json')=='e83011570849e027883fc9e34e7bf04907e79dabbcfb89382a50571f774902db'
 shutil.copy2(PREV/'reference_stats/index.json',ROOT/'reference_stats/index.json');shutil.copy2(PREV/'calibration/final.json',ROOT/'calibration/final.json')
 assert sha(ROOT/'calibration/final.json')=='c4af94f15e07e757cbd4322887576d6d9d0c30a91a431a6bea14f8c87f98d5b8'
 for e in load(ROOT/'reference_stats/index.json')['entries']:assert sha(e['path'])==e['sha256']
 plan=load(ROOT/'plans/shared.json');updates=[]
 for k,row in enumerate(plan['updates'],1):
  assert row['adaptation_step']==k
  raw=json.dumps(row,sort_keys=True,separators=(',',':')).encode()
  updates.append(dict(additional_step=k,source_plan_index=k-1,source_plan_step=k,adaptation_step=1000+k,source_row_canonical_sha256=hashlib.sha256(raw).hexdigest(),source_row=row))
 counts=collections.Counter((r['source_row']['domain'],r['source_row']['external_qp']) for r in updates);assert len(updates)==1000 and len(counts)==20 and set(counts.values())=={50}
 dump(ROOT/'plans/continuation.json',dict(schema='v633_second_exposure',source_path=str(PREV/'plans/shared.json'),source_sha256=sha(ROOT/'plans/shared.json'),mapping='additional k -> original row k-1; global adaptation1000+k',new_unique_sources=0,new_unique_windows=0,updates=updates))
 cp=load(ROOT/'audits/source_checkpoint_inspection.json')['sources'];sources={x['branch']:x for x in cp}
 cfg=load(PREV/'config.json');cfg.update(schema='v633',experiment=ROOT.name,source_experiment='v632 B and v632b C/D respective training step1000',branches=list(sources),branch_sources=sources,fresh_optimizer=False,restore_source_optimizer=True,updates=1000,initial_adaptation_step=1000,final_adaptation_step=2000,primary_checkpoint=2000,checkpoint_steps=[1000,1001,1250,1500,1750,2000],branch_lrs={b:[6e-6,1e-6] for b in sources},optimizer=dict(bridge_lr=6e-6,generator_lr=1e-6,betas=[.9,.999],eps=1e-8,weight_decay=1e-4,grad_clip=1.),expected_points=4094,historical_points=2300,new_encodes=1794,discriminator_optimizer_updates=0,new_prefill_windows=0,new_calibration_windows=0,reference_stats_index_sha256=sha(ROOT/'reference_stats/index.json'),calibration_sha256=sha(ROOT/'calibration/final.json'),training_plan='plans/continuation.json',original_training_plan_sha256=sha(ROOT/'plans/shared.json'),source_plan_repeat=2)
 methods=['original','B1_initial']+[f'{b}_{step}' for step in (1000,2000) for b in sources];cfg['methods']=methods;cfg['fixed_report_steps']=[2000];cfg['train_fit_steps']=[]
 dump(ROOT/'config.json',cfg)
 proto=load(PREV/'effective_protocol.json');proto.update(schema='v633',experiment=ROOT.name,updates=1000,new_optimizer_updates=3000,initial_adaptation_step=1000,final_adaptation_step=2000,expected_points=4094,logical_points=4094,new_encodes=1794,historical_reused_encodes=2300,baseline_points=2300,methods=methods,new_branches=list(sources),historical_branches=[],fresh_optimizer=False,restore_source_optimizer=True,calibration='read-only v632b completed40; zero new calibration',calibration_sha256=cfg['calibration_sha256'],reference_stats_index_sha256=cfg['reference_stats_index_sha256'],source_plan_repeat=2,checkpoint_steps=cfg['checkpoint_steps'],diagnostic=dict(step=1500,qps=[0,5,9],new_points=414,label='fixed 3-QP diagnostic only'),main=dict(step=2000,qps=list(range(10)),new_points=1380),common_rate=dict(methods=methods,points=100,spacing='ln(bpp)',interpolation='PCHIP',extrapolate=False),B_EMA='N/A; auxiliary pre-update full views are diagnostics only; no retroactive filling of historical gaps',gradient_recording='preclip merge error measured before assigning/clip; separate postclip scaling error')
 for k in ['authority','frozen_runtime_protocol_sha256','all_adaptation_logical_points','seed_model']:proto.pop(k,None)
 dump(ROOT/'protocol.json',proto);dump(ROOT/'effective_protocol.json',proto);dump(ROOT/'branch_manifest.json',dict(branches=sources,restore_optimizer=True,new_updates_per_branch=1000,source_steps=1000,final_steps=2000))
 tasks=[]
 manifest=load(ROOT/'eval_manifest.json')
 for d,vs in manifest.items():
  for v in vs:
   for m in ['original','B1_initial']:
    for q in range(10):tasks.append(dict(stage='baseline',dataset=d,video=v['video_index'],method=m,qp=q))
   for b in sources:
    for step,qs in [(1000,range(10)),(1500,[0,5,9]),(2000,range(10))]:
     for q in qs:tasks.append(dict(stage='historical' if step==1000 else 'new',dataset=d,video=v['video_index'],method=f'{b}_{step}',qp=q,step=step,branch=b))
 assert len(tasks)==4094 and sum(t['stage']=='new' for t in tasks)==1794
 dump(ROOT/'task_manifest.json',dict(expected=4094,historical=2300,new_encodes=1794,tasks=tasks))
 deps=load(PREV/'source_hashes.json')['hashes'];hashes=dict(deps)
 req=['REPORT.md','RESULTS_SUMMARY.md','PROTOCOL_REVISION.md','effective_protocol.json','branch_manifest.json','config.json','protocol.json','training32.py','train.py','evaluate32.py','completion_report32b.py','calibration32b.py','distribution32.py','frames32.py','io32.py','reference_stats/index.json','calibration/final.json','audits/gradient_recording_integrity.json','audits/gradient_recording_repair.json','final_integrity.json','data_isolation_audit.json','geometry_audit.json']
 for n in req:hashes[str(PREV/n)]=sha(PREV/n)
 for e in load(PREV/'reference_stats/index.json')['entries']:hashes[e['path']]=e['sha256']
 for b,info in sources.items():
  hashes[info['path']]=info['sha256'];parent=Path(info['path']).parents[3]
  for n in ['io32.py','train.py','training32.py','distribution32.py','frames32.py','config.json','protocol.json','source_hashes.json']:hashes[str(parent/n)]=sha(parent/n)
  hashes[str(parent/'branches'/b/'checkpoint_index.json')]=sha(parent/'branches'/b/'checkpoint_index.json')
 dump(ROOT/'source_hashes.json',dict(hashes=hashes,read_only=True));dump(ROOT/'audits/historical_before.json',{p:dict(sha256=h) for p,h in hashes.items()})
 (ROOT/'audits/git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=REPO,text=True));head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=REPO,text=True).strip();assert 'ZhangYangyingzi/GVC-RT'.lower() in remote.lower()
 dump(ROOT/'audits/preflight.json',dict(status='PASS',time=time.time(),actual_HEAD=head,reference_commit='abb1f516ef44d48485ddfbb0dcdde7b737a4ca6e',origin=remote,no_applicable_AGENTS_found=True,existing_matching_workers=False,python='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python',disk_free_bytes=shutil.disk_usage(ROOT).free,conservative_new_storage_budget_bytes=100*1024**3,checkpoint_inspection='audits/source_checkpoint_inspection.json'))
 (ROOT/'.gitignore').write_text('*.pt\n*.npz\n*.npy\n*.bin\n*.lock\n*.pid\n*.tmp\n__pycache__/\nfeatures/\nbitstreams/\npublication/\n')
 # A copy documents the precise inherited limitations; it is not a new audit result.
 for n in ['data_isolation_audit.json','geometry_audit.json']:
  shutil.copy2(PREV/n,ROOT/'audits'/('inherited_'+n))
 print('FIXED CONFIGURATION',head,'4094 tasks;3000 additional optimizer updates')
if __name__=='__main__':main()
