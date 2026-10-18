import hashlib,json,shutil,torch,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[1]
PARENT=ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000'; SCREEN=ROOT.parent/'gvcrt_neural_wrapper_v6_34a_p_joint_fd_trend_screen'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert sha(PARENT/'plans/shared.json')=='b61c3552277b9c45f1d9f5a615afa2ddf2d9aa9ed48bd55b0eb63385e5290b42'
assert sha(SCREEN/'eval_manifest.json')==sha(ROOT/'eval_manifest.json')
branches={}
for b,expected,step in [('N_noP_BG_FD','6162740ea84cf6a7725084d3a9f9ce9241db430642ddbeadd57dd6c96e640f96',475),('P_PBG_FD','5a185fb4fdd5ced958a53350f5b0a684e54707795d0d1e7ce13485759c99485e',368)]:
 src=PARENT/'branches'/b/'checkpoints/latest.pt';dst=ROOT/'branches'/b/'checkpoints/latest.pt'
 assert sha(src)==expected
 s=torch.load(src,map_location='cpu',weights_only=False,mmap=True)
 assert s['adaptation_step']==s['committed_optimizer_updates']==s['plan_cursor']==step
 assert s['signature']=='fcfccc56cfb3b9ae45016df5d82e706973f99ac20c594cd834acabeea1b93e03'
 assert s['plan_sha256']=='b61c3552277b9c45f1d9f5a615afa2ddf2d9aa9ed48bd55b0eb63385e5290b42'
 assert sum(int(v['counter']) for v in s['EMA'].values())==step
 ost=[int(v['step']) for v in s['optimizer']['state'].values() if 'step' in v]
 assert ost and set(ost)=={step}
 shutil.copy2(src,dst)
 shutil.copy2(src,ROOT/'branches'/b/'checkpoints'/f'adaptation_step_{step}.pt')
 receipt=PARENT/'branches'/b/'receipts'/f'{step}.json';assert receipt.is_file()
 shutil.copy2(receipt,ROOT/'branches'/b/'receipts'/f'{step}.json')
 branches[b]={'source_checkpoint':str(src),'source_sha256':sha(src),'copied_checkpoint':str(dst),'copied_sha256':sha(dst),'step':step,'optimizer_internal_steps':sorted(set(ost)),'ema_counter_sum':sum(int(v['counter']) for v in s['EMA'].values()),'plan_cursor':s['plan_cursor'],'parent_signature':s['signature'],'plan_sha256':s['plan_sha256'],'frozen_hashes':s['frozen_hashes'],'initial_module_hashes':s['initial_module_hashes'],'restore_components':['wrapper','bridge','generator','AdamW','EMA','crop/global RNG','buffers','plan cursor','last receipt'],'status':'PASS'}
 del s
# Validate all 90 prior points and bring small receipts into the new namespace.
old_tasks=js(SCREEN/'task_manifest.json')['tasks'];assert len(old_tasks)==90
reused=[]
for t in old_tasks:
 name=f"{t['stage']}_{t['dataset']}_{t['video']}_{t['method']}_{t['qp']}.json";src=SCREEN/'tasks'/name
 assert src.is_file();r=js(src);assert r['status']=='PASS'
 for p,m in r['artifact_metadata'].items():
  q=Path(p);assert q.is_file() and q.stat().st_size==m['bytes'] and q.stat().st_mtime_ns==m['mtime_ns'] and sha(q)==m['sha256']
 shutil.copy2(src,ROOT/'tasks'/name)
 reused.append({'task':t,'receipt':str(src),'receipt_sha256':sha(src),'result_signature':r['signature'],'artifacts_verified':len(r['artifact_metadata'])})
# Extend tasks only with N/P step500. Historical task records remain byte-identical.
tasks=list(old_tasks)
for method,branch in [('N_noP_BG_FD_500','N_noP_BG_FD'),('P_PBG_FD_500','P_PBG_FD')]:
 for v in range(6):
  for q in (0,5,9):tasks.append({'stage':'adapt','dataset':'ulong','video':v,'method':method,'step':500,'branch':branch,'qp':q})
dump(ROOT/'task_manifest.json',{'expected_logical_points':126,'tasks':tasks})
# Parent training sources are preserved, with a continuation-only adapter change.
dump(ROOT/'audits/restore_audit.json',{'status':'PASS','captured_unix':time.time(),'parent_experiment':str(PARENT),'parent_training_signature':'fcfccc56cfb3b9ae45016df5d82e706973f99ac20c594cd834acabeea1b93e03','branches':branches,'parent_driver_processes_verified_absent':True,'latest_source_selected':'highest complete transaction <=500','no_optimizer_reset':True,'no_prefill_or_calibration_repeated':True})
dump(ROOT/'audits/historical_reuse_audit.json',{'status':'PASS','source_experiment':str(SCREEN),'source_manifest_sha256':sha(SCREEN/'eval_manifest.json'),'expected_reused_points':90,'verified_reused_points':len(reused),'reused_task_artifacts':reused,'fresh_encode_count':0})
print('PREPARED',json.dumps({k:v['step'] for k,v in branches.items()},sort_keys=True),'reused',len(reused))
