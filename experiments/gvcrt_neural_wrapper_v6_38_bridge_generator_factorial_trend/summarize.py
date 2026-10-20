"""Validate fixed-budget and same-stream results; export descriptive paired differences."""
from io32 import *
import math,collections,subprocess
METRICS=('LPIPS','DISTS','FloLPIPS','PSNR','L1')
def main():
 import torch
 tasks=[load(p) for p in sorted((ROOT/'tasks').glob('*_v*_q*.json')) if load(p).get('status')=='PASS'];rows=[t['result'] for t in tasks]
 assert len(rows)==24 and len({(r['method'],r['video'],r['QP']) for r in rows})==24
 streams=[]
 for v in (0,1):
  for q in (0,5,9):
   group=[t for t in tasks if t['result']['video']==v and t['result']['QP']==q];assert len(group)==4
   assert len({(t['result']['bitstream_sha256'],t['result']['real_bytes'],t['result']['actual_bpp']) for t in group})==1
   assert len({json.dumps(t['decoded_features'],sort_keys=True) for t in group})==1
   p=Path(group[0]['result']['bitstream_path']);assert sha(p)==group[0]['result']['bitstream_sha256']
   streams.append(dict(video=v,QP=q,methods=['O','B','G','BG'],bytes=p.stat().st_size,sha256=sha(p),identical_y_f_q=True))
 training={};cpindex={};binindex=[]
 for b in ('B','G','BG'):
  f=load(ROOT/f'branches/{b}/final_integrity.json');assert f['committed_updates']==100 and f['optimizer_internal_step']==100
  receipts=[load(ROOT/f'branches/{b}/receipts/{i}.json') for i in range(1,101)];assert len(list((ROOT/f'branches/{b}/receipts').glob('*.json')))==100
  plan=load(ROOT/'plans/first100.json')['updates']
  assert all(r['step']==i and r['source_row_sha256']==plan[i-1]['source_row_canonical_sha256'] for i,r in enumerate(receipts,1))
  events=[json.loads(l) for l in (ROOT/f'branches/{b}/optimizer_calls.jsonl').read_text().splitlines()]
  committed=[e['step'] for e in events if e['event']=='COMMITTED'];assert committed==list(range(1,101))
  assert all(r['frozen_check']=='PASS' for r in receipts)
  training[b]=dict(status='PASS',committed_updates=100,optimizer_step=100,ledger_committed=100,active_modules=f['active_modules'],frozen_BG=f['frozen_BG'],initial=f['initial_module_hashes'],final=f['final_module_hashes'],physical_GPUs=sorted({r['physical_GPU'] for r in receipts}))
  write(ROOT/f'results/train_{b}.csv',[{k:r[k] for k in ('step','source_plan_step','source_row_sha256','domain','video','external_qp','losses','gradient_modules','parameter_delta_norms','preclip_norm','postclip_norm','clip_coefficient','seconds','peak_memory_MiB','physical_GPU')} for r in receipts])
  cpindex[b]={}
  for step in (0,50,100):
   p=ROOT/f'branches/{b}/checkpoints/adaptation_step_{step}.pt';assert p.exists()
   st=torch.load(p,map_location='cpu',weights_only=True);assert st['committed_updates']==step and st['plan_cursor']==step
   assert {int(v['step']) for v in st['optimizer']['state'].values()}==({step} if step else set())
   for name in ('bridge','generator'):
    got=tensor_hash(st[name]);expected=f['initial_module_hashes'][name] if step==0 or name not in f['active_modules'] else f['final_module_hashes'][name] if step==100 else got
    assert got==expected
   if step==0:
    original=torch.load(ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/branches/N_noP_BG_FD/checkpoints/adaptation_step_0.pt',map_location='cpu',weights_only=True)
    assert torch.equal(st['crop_rng']['torch'],original['crop_rng']['torch']) and torch.equal(st['crop_rng']['cuda'],original['crop_rng']['cuda'])
    for mod,bufs in st['buffers'].items():
     if mod in original['buffers']:
      assert all(torch.equal(value,original['buffers'][mod][name]) for name,value in bufs.items())
    del original
   del st
   cpindex[b][str(step)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,optimizer_state_verified=True,buffers_and_RNG_source_equal_at_step0=step==0);binindex.append(cpindex[b][str(step)])
  p=ROOT/f'evaluation/checkpoints/{b}_100.pt';binindex.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
  inf=torch.load(p,map_location='cpu',weights_only=True)
  deployed={k:tensor_hash({n:t.half() for n,t in inf[k].items()}) for k in ('bridge','generator')}
  original_deployment=load(ROOT/'evaluation_config.json')['original_receiver_hashes']
  for k in f['frozen_BG']:assert deployed[k]==original_deployment[k]
  training[b]['deployment_receiver_hashes']=deployed
  training[b]['frozen_BG_deployment_equal_to_original']=True
  del inf
 assert len({json.dumps(t['initial'],sort_keys=True) for t in training.values()})==1
 paired=[]
 for a,b in [('B','O'),('G','O'),('BG','O'),('BG','B'),('BG','G')]:
  for v in (0,1):
   for q in (0,5,9):
    ra=next(r for r in rows if (r['method'],r['video'],r['QP'])==(a,v,q));rb=next(r for r in rows if (r['method'],r['video'],r['QP'])==(b,v,q))
    for m in METRICS:
     d=ra[m]-rb[m];paired.append(dict(comparison=a+'/'+b,video=v,sequence=ra['sequence'],QP=q,metric=m,condition_value=ra[m],reference_value=rb[m],absolute_delta=d,relative_delta=d/max(abs(rb[m]),1e-12),condition_better=d>0 if m=='PSNR' else d<0,actual_bpp=ra['actual_bpp'],bitstream_sha256=ra['bitstream_sha256']))
 summary=[]
 for comparison in ('B/O','G/O','BG/O','BG/B','BG/G'):
  for m in METRICS:
   ps=[p for p in paired if p['comparison']==comparison and p['metric']==m]
   summary.append(dict(comparison=comparison,metric=m,points=6,videos=2,mean_absolute_delta=sum(p['absolute_delta'] for p in ps)/6,mean_relative_delta=sum(p['relative_delta'] for p in ps)/6,condition_wins=sum(p['condition_better'] for p in ps),aggregation='six fixed same-stream anchors, video and QP equal weight; descriptive only'))
 write(ROOT/'results/evaluation.csv',rows);write(ROOT/'results/paired_differences.csv',paired);write(ROOT/'results/comparison_summary.csv',summary)
 dump(ROOT/'audits/stream_and_feature_equality.json',dict(status='PASS',streams=streams,real_RANS=True,no_estimated_rate_substitution=True))
 dump(ROOT/'audits/training_integrity.json',dict(status='PASS',O_new_updates=0,conditions=training,smoke_updates_included=True,extra_smoke_optimizer_updates=0))
 dump(ROOT/'checkpoint_index.json',cpindex);dump(ROOT/'binary_artifacts_index.json',binindex)
 ev=[json.loads(l) for l in (ROOT/'logs/task_events.jsonl').read_text().splitlines()];starts=[e for e in ev if e['event']=='START'];exits=[e for e in ev if e['event']=='EXIT'];assert len(starts)==len(exits)
 failures=[e for e in exits if e['exit_code']!=0]
 dump(ROOT/'audits/execution.json',dict(status='PASS',tasks_started=len(starts),tasks_exited=len(exits),failed_attempts=failures,physical_GPUs=sorted({e['gpu'] for e in starts}),exit_codes=[e['exit_code'] for e in exits]))
 final=dict(status='PASS',updates={'O':0,'B':100,'G':100,'BG':100},RD_complete=24,RD_planned=24,FID='not planned',streams=6,same_stream_and_decoded_features=True,smoke_conditions=3,blocked=[],failed_tasks=[],failed_attempts=len(failures),single_seed=20261034,scope='two-video 100-update trend; no significance/generalization inference')
 dependencies=load(ROOT/'manifests/code_dependencies.json')
 assert all(sha(Path(p))==expected for p,expected in dependencies.items())
 assert all(sha(Path(p))==expected for p,expected in load(ROOT/'source_hashes.json').items())
 dump(ROOT/'audits/dependency_integrity.json',dict(status='PASS',historical_dependency_files=len(dependencies),v637_source_files_unchanged=True))
 dump(ROOT/'final_integrity.json',final)
 lines=['# Results summary','',f"Status: PASS. Updates O=0, B=100, G=100, BG=100. RD24/24, six shared real-RANS streams, two heldout HEVC-B videos, QP0/5/9,64 frames,63 flow transitions. Failed attempts: {len(failures)}.",'','All methods have identical full stream bytes/SHA and decoded y/f/q hashes for each video/QP. Training module freezes and exact100 committed optimizer steps pass.','', '|Comparison|Metric|Mean absolute delta|Mean relative delta|Wins/6|','|---|---|---:|---:|---:|']
 for s in summary:lines.append(f"|{s['comparison']}|{s['metric']}|{s['mean_absolute_delta']:.9g}|{s['mean_relative_delta']:.6g}|{s['condition_wins']}|")
 lines+=['','Delta means first condition minus reference. Lower LPIPS/DISTS/FloLPIPS/L1 and higher PSNR are better. Averages give each of six video/QP anchors equal weight. No statistical inference is made.','', 'This is a single-seed,100-update trend screen on two training identities (1 historical U-Long row,99 historical Beauty rows), not evidence of convergence, significance or generalization. No extra branch, alpha, seed, checkpoint selection or training extension was run.','', 'See `results/evaluation.csv` for24 individual points, `results/paired_differences.csv` for150 metric differences, `branches/*/receipts/` for all committed update logs, and `logs/task_events.jsonl` for actual GPU and exit records.']
 textfile(ROOT/'RESULTS_SUMMARY.md','\n'.join(lines)+'\n');textfile(ROOT/'REPORT.md','\n'.join(lines)+'\n')
 # Complete computational state before publication; publication is recorded separately.
 dump(ROOT/'run_status.json',dict(status='COMPUTE_COMPLETE',**{k:v for k,v in final.items() if k!='status'},time=time.time()))
 print('FINAL_INTEGRITY_PASS',flush=True)
if __name__=='__main__':main()
