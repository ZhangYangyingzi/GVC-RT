from io32 import *
import subprocess,collections,shutil

def git(*a):return subprocess.check_output(['git',*a],cwd=REPO,text=True).strip()
def main():
 for n in ('audits','plans','logs','results','figures','evaluation','tasks','raw','runtime','branches/D_basicDISTS_noP_500'):(ROOT/n).mkdir(parents=True,exist_ok=True)
 (ROOT/'errors.jsonl').touch(exist_ok=True)
 src=ROOT.parent/'gvcrt_neural_wrapper_v6_34b_p_joint_fd_trend_500/plans/shared.json';assert sha(src)=='b61c3552277b9c45f1d9f5a615afa2ddf2d9aa9ed48bd55b0eb63385e5290b42';shutil.copyfile(src,ROOT/'plans/shared.json')
 checks={
 ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/branches/N_noP_BG_FD/checkpoints/adaptation_step_0.pt':'18394bc0145e4bfc9eb18fb009fdb9dcc7f626536fe2367fb19b95e471f09074',
 V36/'branches/B_basicLoss_noP_500/checkpoints/adaptation_step_500.pt':'6b1e74b0f5dc524bb28fdbfc6ffe5352ec19f967de5adcc788e3c216a6dfffbd',
 V36/'evaluation/checkpoints/B_basicLoss_noP_500_500.pt':'e76fb5688b7bb54c54f6ec4dc9a2c76fc285d2fb050ae59d2f7a209df5198d95',
 ROOT.parent/'gvcrt_neural_wrapper_v6_34b_p_joint_fd_trend_500/branches/N_noP_BG_FD/checkpoints/adaptation_step_500.pt':'b29a1236a5158aa07f2b685f945723064c376404f5564d1f79cb5b35beb839d9',
 ROOT.parent/'gvcrt_neural_wrapper_v6_34b_p_joint_fd_trend_500/evaluation/checkpoints/N_noP_BG_FD_500.pt':'ca2c66aa1d68877779c8646b66d02e4ba06e82fc4f3b3c92c11271fc7ffc19dc'}
 cp=[]
 for p,expected in checks.items():
  actual=sha(p);assert actual==expected,(str(p),actual);cp.append(dict(path=str(p),sha256=actual,bytes=p.stat().st_size))
 for p in (REPO/'checkpoints').glob('GVC-RT_?.pt'):cp.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
 plan=load(src)['updates'][:500];assert len(plan)==500;counts=collections.Counter(x['source_row']['domain'] for x in plan);assert dict(counts)==dict(ulong=250,uvg=250)
 for i,x in enumerate(plan):assert x['step']==i+1 and hashlib.sha256(json.dumps(x['source_row'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==x['source_row_canonical_sha256']
 ev=load(ROOT/'eval_manifest.json');assert {d:len(v) for d,v in ev.items()}==dict(ulong=6,uvg_holdout=2,hevc_b=5)
 train_sources={x['source_row']['source_sha256'] for x in plan};test_sources={v['source_sha256'] for vs in ev.values() for v in vs};assert not train_sources&test_sources
 source_files=set(REPO.glob('src/**/*.py'))
 for name in ('gvcrt_neural_wrapper_v6_36_noP_basic_vs_full_loss_matched_500','gvcrt_neural_wrapper_v6_35_full_pipeline_gain_attribution_500','gvcrt_neural_wrapper_v6_34b_p_joint_fd_trend_500','gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000'):
  folder=ROOT.parent/name;source_files.update(folder.glob('*.py'));source_files.update(folder.glob('*.json'));source_files.update(folder.glob('audits/*.json'))
 source_files.update((ROOT.parent/'gvcrt_neural_wrapper_v6_29_fid_audit_feature_covariance_control').glob('stat_loss.py'));source_files.update((ROOT.parent/'gvcrt_parallel_v5a3_v60').glob('*.py'));source_files.update((ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint').glob('train.py'))
 sources={str(p):dict(sha256=sha(p),git_blob=git('hash-object',str(p))) for p in sorted(source_files)}
 dump(ROOT/'source_manifest.json',dict(HEAD=git('rev-parse','HEAD'),origin_main=git('rev-parse','origin/main'),user_reference_main='2395d42ebd5593bd4f3ab9ae8bc1253fe1dc7482',worktree_status=git('status','--short'),AGENTS_found=[],files=sources,policy='all historical dependencies read-only; new code only in this experiment'))
 dump(ROOT/'source_hashes.json',sources)
 dump(ROOT/'checkpoint_reuse_manifest.json',dict(status='PASS',checkpoints=cp,source_index_path=str(V36/'checkpoint_reuse_manifest.json'),source_index_sha256=sha(V36/'checkpoint_reuse_manifest.json'),evaluation_config_source_sha256=sha(V36/'evaluation_config.json')))
 dump(ROOT/'audits/source_and_checkpoint_verification.json',dict(status='PASS',checkpoints=cp,source_manifest_sha256=sha(ROOT/'source_manifest.json')))
 audit=load(V36/'audits/data_isolation.json');assert audit['status']=='PASS'
 dump(ROOT/'audits/data_isolation.json',dict(status='PASS',inherited_path=str(V36/'audits/data_isolation.json'),inherited_sha256=sha(V36/'audits/data_isolation.json'),training_domains=dict(counts),train_source_hashes=sorted(train_sources),test_source_hashes=sorted(test_sources),overlap=[],official_pretraining_exposure='UNKNOWN; not claimed excluded',new_GT_statistics_pool=False,plan_rows=[dict(step=x['step'],sha256=x['source_row_canonical_sha256']) for x in plan]))
 protocol=dict(name=ROOT.name,methods=METHODS,new_updates=dict(zip(METHODS,[0,0,500,0,0,0,0])),P=False,seed=20261034,steps=500,save_steps=[0,100,250,500],primary_step=500,plan_sha256=sha(src),training_domains=dict(counts),batch=1,crop=256,frames=64,warmup=list(range(1,49)),supervised=list(range(49,64)),reference_detach=True,BPTT=False,optimizer=dict(name='AdamW',bridge_LR=6e-6,generator_LR=1e-6,betas=[.9,.999],eps=1e-8,weight_decay=1e-4,schedule='constant',global_clip=1),loss=dict(LPIPS=1,reconstruction_L1=REC,DISTS=.5,proxy_L1=.01,rate='v6.36 beta(q)*rate/65536',legacy_stat='not computed',FD='not computed'),evaluation=dict(datasets={d:len(v) for d,v in ev.items()},frames=64,QP=QPS,IP=-1,native_I=True,rate_fps=20,latent_alphas=[0,.5,1],planned_RD=273,planned_FID=63,latents='actual18-channel generator input',interpolation_grid=20,interpolation='linear ln(actual bpp), per video then equal video weight',FID='pooled official Inception2048, FP64 ddof1 low-rank cross-SVD'),equivalence=dict(RGB_maxabs=1e-5,RGB_relativeL2=1e-6,metrics_abs=1e-5,FID_abs=1e-3),allowed_GPU=[4,5,6,7])
 dump(ROOT/'protocol.json',protocol);dump(ROOT/'effective_config.json',protocol);dump(ROOT/'config.json',protocol)
 cfg=load(ROOT/'evaluation_config.json');cfg['methods']={'original':[None,None]};cfg['checkpoints']={k:v for k,v in cfg['checkpoints'].items() if k in ('N_oursLoss_noP_500','B_basicLoss_noP_500')};cfg['methods'].update({k:[None,k] for k in cfg['checkpoints']});cfg['methods']['F_fullLoss_noP_500']=[None,'N_oursLoss_noP_500'];cfg['deployment_receiver_hashes']={k:v for k,v in cfg['deployment_receiver_hashes'].items() if k in cfg['checkpoints']};cfg.update(expected_points=273,expected_pooled_FID_points=63,schema='v637_evaluation_with_validated_v634_adapter',experiment='v637_loss_bridge_generator_mechanism_500',protocol_methods=METHODS,rate_accounting_fps=20);dump(ROOT/'evaluation_config.json',cfg)
 rows=read(V36/'results/all_rd_points.csv');assert len(rows)==117
 verified=[]
 for x in rows:
  assert x['method'] in ('O_original','B_basicLoss_noP_500','F_fullLoss_noP_500')
  v=next(v for v in ev[x['dataset']] if v['video_index']==int(x['video']));assert x['source_rgb_sha256']==v['rgb_sha256'] and json.loads(x['source_frame_indices'])==v['source_frame_indices']
  for k in ('bitstream','feature','frame_metrics','transitions'):assert sha(x[k+'_path'])==x[k+'_sha256']
  assert Path(x['bitstream_path']).stat().st_size==int(x['real_bytes'])
  verified.append(dict(dataset=x['dataset'],video=int(x['video']),QP=int(x['QP']),method=x['method'],source_receipt=x['reuse_source_receipt'],source_receipt_sha256=sha(x['reuse_source_receipt']),artifact_hashes={k:x[k+'_sha256'] for k in ('bitstream','feature','frame_metrics','transitions')}))
 write(ROOT/'results/reused_points.csv',rows);pooled=read(V36/'results/pooled_fid.csv');assert len(pooled)==27;write(ROOT/'results/reused_pooled_fid.csv',pooled)
 dump(ROOT/'audits/reuse_pointwise_verification.json',dict(status='PASS',RD=117,FID=27,points=verified,pooled_source_sha256=sha(V36/'results/pooled_fid.csv'),model_config_sha256=sha(V36/'evaluation_config.json')))
 tasks=[dict(dataset=d,video=v['video_index'],qp=q) for d,vs in ev.items() for v in vs for q in QPS];dump(ROOT/'evaluation/task_manifest.json',dict(planned_RD=273,planned_FID=63,source_tasks=tasks,dependency_graph={'gradient_audit':[],'D500':['gradient_audit'],'module_swap':['source_audit'],'D_real_encode':['D500'],'replay_all':['D500','D_real_encode'],'aggregate':['replay_all'],'publish':['aggregate','integrity']},methods=METHODS))
 print('PREPARE PASS',dict(counts),flush=True)
if __name__=='__main__':main()
