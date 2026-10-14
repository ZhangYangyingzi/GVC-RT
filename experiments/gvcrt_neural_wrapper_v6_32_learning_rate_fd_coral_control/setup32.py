"""Freeze exact historical plans, sources, weights, and fixed evaluation matrix."""
from io32 import *
import subprocess,collections
def main():
 import torch
 for n in ('audits','logs','tasks','plans','imports','reference_stats','calibration','branches','evaluation','vendor'):(ROOT/n).mkdir(exist_ok=True)
 cfg=load(V29/'config.json');ec=load(V29/'evaluation/config.json');src=cfg['source_checkpoint']
 assert src['sha256']=='dc76f0e221dc7aaa5e5f45f648722584ce226e65828ce24151f9961985f3f6f7'
 assert sha(src['path'])==src['sha256']
 state=torch.load(src['path'],map_location='cpu',weights_only=True)
 actual={k:tensor_hash(state[k]) for k in ('wrapper','bridge','generator')}
 assert actual==src['module_hashes'] and actual['wrapper']=='08cfe75a5292c9ad874c6ab593a418fbc425c5ac9182ba4edaf4f779a441d0a9'
 dep=ec['checkpoints']['B1_initial'];assert sha(dep['path'])==dep['sha256']
 sd=torch.load(dep['path'],map_location='cpu',weights_only=True);assert {k:tensor_hash(sd[k]) for k in actual}==actual
 hashes={src['path']:src['sha256'],dep['path']:dep['sha256']}
 for origin,names in [(V29,['config.json','evaluation/config.json','protocol.json','qp_semantics_audit.json','calibration/final.json','plans/shared.json','plans/calibration.json','train_manifest_ulong.json','train_manifest_uvg.json','window_training.py','stat_loss.py','train.py','io21.py','adapter.py']), (V30,['old_sources.json','protocol.json','adapter.py','io21.py','fid_statistics.py','audits/cache_closure_integrity.json']), (V31,['probe_worker.py','gradient_cosines31.py','adapter31.py','checkpoint_audit.json','final_integrity.json'])]:
  for n in names:
   p=origin/n;hashes[str(p)]=sha(p);textfile(ROOT/'imports'/origin.name/n,p.read_text())
 for n in ('plans/shared.json','plans/calibration.json','train_manifest_ulong.json','train_manifest_uvg.json','qp_semantics_audit.json'):textfile(ROOT/n,(V29/n).read_text())
 sources={d:load(V29/'manifests'/f'{d}.json')['videos'] for d in ('ulong','uvg_holdout','uvg_validation','hevc_b','mcljcv')}
 closure_sources=load(V30/'old_sources.json')
 assert all(sources[d]==v for d,v in closure_sources.items())
 assert {d:len(v) for d,v in sources.items()}==dict(ulong=8,uvg_holdout=2,uvg_validation=1,hevc_b=5,mcljcv=30)
 for d in sources:
  p=V29/'manifests'/f'{d}.json';hashes[str(p)]=sha(p);textfile(ROOT/'manifests'/f'{d}.json',p.read_text())
 dump(ROOT/'eval_manifest.json',sources)
 plan=load(ROOT/'plans/shared.json');cal=load(ROOT/'plans/calibration.json');assert len(plan['updates'])==1000 and len(cal['updates'])==40
 strata=collections.Counter((r['domain'],r['external_qp']) for r in plan['updates']);assert len(strata)==20 and set(strata.values())=={50}
 videos={d:load(ROOT/f'train_manifest_{d}.json')['videos'] for d in ('ulong','uvg')}
 resolution=[]
 for d,vs in videos.items():
  for v in vs:
   resolution.append(dict(domain=d,video=v.get('filename',v.get('name')),width=v.get('width'),height=v.get('height')))
 dump(ROOT/'audits/resolution_manifest.json',dict(videos=resolution))
 cfg.update(schema='v632',branches=list(BRANCHES),lambda_stat=STAT,seed=plan['seed'],seed_global=20261032,updates=1000,checkpoint_steps=[0,250,500,1000],latest_every=25,expected=dict(committed_updates=4000,prefill=1000,calibration=40,logical_points=3312,new_encodes=2392,baseline=920),trainable_modules=['bridge','generator'],frozen_wrapper=True,no_discriminator=True,D_updates=0,optimizer_effective=dict(betas=[.9,.999],eps=1e-8,weight_decay=1e-4,clip_max_norm=1.),beta_ema=.95,feature_dimension=2048,training_ddof=0,evaluation_ddof=1,training_epsilon_multiplier=1e-4,distribution_backward_scale=4096,reference_commit='230881952522810904c1d549b716f4718132663b',actual_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),branch_lrs={b:[3e-6,5e-7] if b.startswith('A_') else [6e-6,1e-6] for b in BRANCHES})
 cfg['methods']=['original','B1_initial']+[f'{b}_1000' for b in BRANCHES]
 ec.update(datasets=list(sources),expected_points=3312,methods={m:ec['methods'][m] for m in ('original','B1_initial')},checkpoints={'B1_initial':dep},deployment_receiver_hashes={'B1_initial':ec['deployment_receiver_hashes']['B1_initial']},IP=-1)
 dump(ROOT/'config.json',cfg);dump(ROOT/'evaluation_config.json',ec)
 protocol=load(V30/'protocol.json');protocol.update(experiment='v632',IP=-1,frames=64,source_pool='v629 actual official frozen manifests; v630 old_sources',trainable=['bridge','generator'],P_frozen=True,global_view=[1920,1080],canvas=[1920,1088],EMA_beta=.95,training_ddof=0,eval_ddof=1,new_encodes=2392,baseline_points=920,logical_points=3312)
 dump(ROOT/'protocol.json',protocol)
 dump(ROOT/'checkpoint_audit.json',dict(status='PASS',training=src,deployment=dep,actual_saved_dtype_hashes=actual,compression_hash=cfg['source_checkpoint']['compression_hash'],FP16_deployment_receiver_hashes=ec['deployment_receiver_hashes']['B1_initial']))
 dump(ROOT/'branch_manifest.json',dict(branches=[dict(name=b,lrs=cfg['branch_lrs'][b],source_step=1000,additional_updates=1000,loss='C1'+(' + FD' if b.startswith('C_') else ' + mean/CORAL' if b.startswith('D_') else '')) for b in BRANCHES]))
 dump(ROOT/'objective_spec.json',dict(base='LPIPS+.5*DISTS+beta(q)*rate_bpp+.01*proxy_L1',rec=REC,stat=STAT,fused_base_backward=True,independent_rec_stat_scale=4096,normalization=15,distribution_view='full 1920x1080 RGB',FD='regularized Frechet distance, torch float64',CORAL='mean squared / d + covariance squared / (4*d*d)',coefficients='training-only 40 windows once'))
 tasks=[]
 for d,vs in sources.items():
  for v in vs:
   for m in ('original','B1_initial'):
    for q in range(10):tasks.append(dict(stage='baseline',dataset=d,video=v['video_index'],method=m,qp=q))
   for b in BRANCHES:
    for s,qs in ((250,(0,5,9)),(1000,range(10))):
     for q in qs:tasks.append(dict(stage='new',dataset=d,video=v['video_index'],method=f'{b}_{s}',branch=b,step=s,qp=q))
 assert len(tasks)==3312;dump(ROOT/'task_manifest.json',dict(tasks=tasks))
 for p,h in load(V29/'audits/dependencies.json').items():
  if Path(p).suffix=='.py':assert sha(p)==h;hashes[p]=h
 w='/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth';assert sha(w)=='6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2';hashes[w]=sha(w)
 fd=ROOT/'vendor/FD-Loss';commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=fd,text=True).strip();assert commit=='5c03b8112fec8b9432631e4ce053c0d918cc24bc'
 fdsrc={}
 for n in ('frechet_distance/losses.py','frechet_distance/queue.py','LICENSE'):
  p=fd/n;h=sha(p);hashes[str(p)]=h;fdsrc[n]=h;textfile(ROOT/'imports/FD-Loss'/n,p.read_text())
 dump(ROOT/'audits/FD_source.json',dict(status='PASS',repository='https://github.com/Jiawei-Yang/FD-Loss',commit=commit,files=fdsrc,read_sources=True,ported='FP64 moments EMA and regularized symmetric Frechet math only; no main/global settings, no constant fallback'))
 dump(ROOT/'source_hashes.json',dict(hashes=hashes,metadata={p:dict(bytes=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns) for p in hashes}))
 textfile(ROOT/'implementation_differences.md','Independent v6.32 ROOT and output. Historical fused C1 preserved; real stat fields and single loss accumulation. P frozen; B/G trainable. Full-frame distribution view, independent RNG/buffers, float64 EMA moments; fixed gradient-calibrated coefficients. No D or extra landed optimizer probes.\n')
 textfile(ROOT/'RUNBOOK.md',f'```bash\ncd {REPO}\n{PYTHON} -B {ROOT}/launch32.py\n{PYTHON} -B {ROOT}/check_once.py\n{PYTHON} -B {ROOT}/wait_for_terminal.py\n```\nResume uses the same launch command, locks, task signatures and latest optimizer/RNG/EMA transaction.\n')
 dump(ROOT/'audits/setup.json',dict(status='PASS',expected=cfg['expected'],strata={f'{d}:{q}':n for (d,q),n in strata.items()}))
 print('SETUP PASS',flush=True)
if __name__=='__main__':main()
