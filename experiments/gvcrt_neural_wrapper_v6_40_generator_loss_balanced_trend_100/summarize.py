"""Strict integrity checks and descriptive fixed comparisons; no mechanism attribution."""
from io32 import *
import math,collections
METHODS=('O_original','G_basic','G_dists')
METRICS=('LPIPS','DISTS','FloLPIPS','PSNR','L1','actual_bpp')
PAIRS=(('G_basic','O_original'),('G_dists','O_original'),('G_dists','G_basic'))
def main():
 import torch
 plan=load(ROOT/'plans/first100.json')['updates'];assert len(plan)==100
 assert load(ROOT/'audits/objective_gradient_step0.json')['status']=='PASS'
 counts=collections.Counter((x['source_row']['domain'],x['source_row']['external_qp']) for x in plan)
 assert all(counts[(d,q)]==5 for d in ('uvg','ulong') for q in range(10))
 for i in range(0,100,2):
  a,b=[x['source_row'] for x in plan[i:i+2]];assert {a['domain'],b['domain']}=={'uvg','ulong'} and a['external_qp']==b['external_qp']
 tasks=[]
 for m in METHODS:
  for v in (0,1):
   for q in (0,5,9):
    key=f'{m}_v{v}_q{q}';t=load(ROOT/'tasks'/f'{key}.json');assert t['status']=='PASS' and t['signature']==signature('eval_'+key)
    rr=t['result'];assert rr['code_sha256']==sha(ROOT/'evaluate.py');assert rr['frames']==64 and rr['transitions']==63
    fr=ROOT/'results/frames'/f'{key}.csv';tr=ROOT/'results/transitions'/f'{key}.csv'
    assert sha(fr)==rr['frame_csv_sha256'] and sha(tr)==rr['transition_csv_sha256']
    fs=read(fr);ts=read(tr);assert [int(x['frame']) for x in fs]==list(range(64))
    assert [(int(x['from_frame']),int(x['to_frame'])) for x in ts]==[(i,i+1) for i in range(63)]
    assert abs(rr['PSNR']-(-10*math.log10(max(sum(float(f['pixel_SSE']) for f in fs)/(64*1080*1920*3),1e-15))))<1e-10
    assert all(math.isfinite(float(rr[k])) for k in METRICS)
    assert rr['real_bytes']==Path(rr['bitstream_path']).stat().st_size and sha(rr['bitstream_path'])==rr['bitstream_sha256']
    assert rr['actual_bpp']==rr['real_bytes']*8/(64*1080*1920)
    tasks.append(t)
 assert len(tasks)==18
 streams=[]
 for v in (0,1):
  for q in (0,5,9):
   g=[t for t in tasks if (t['result']['video'],t['result']['QP'])==(v,q)]
   assert len({json.dumps(t['decoded_features'],sort_keys=True) for t in g})==1
   assert len({(t['result']['bitstream_sha256'],t['result']['real_bytes'],t['result']['actual_bpp'],t['bridge_hash']) for t in g})==1
   assert len(g[0]['decoded_features'])==63
   streams.append(dict(video=v,QP=q,methods=list(METHODS),bytes=g[0]['result']['real_bytes'],sha256=g[0]['result']['bitstream_sha256'],identical_y_f_q=True,identical_original_bridge=True,independent_decodes=3))
 dump(ROOT/'audits/same_stream_features_bridge.json',dict(status='PASS',streams=streams,full_real_RANS_bytes=True))
 source_path=ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/branches/N_noP_BG_FD/checkpoints/adaptation_step_0.pt'
 source=torch.load(source_path,map_location='cpu',weights_only=True)
 training={};cpindex={};bins=[];step0s=[]
 for b in METHODS[1:]:
  fin=load(ROOT/f'branches/{b}/final_integrity.json');assert fin['status']=='PASS' and fin['committed_updates']==fin['optimizer_internal_step']==100 and fin['active_modules']==['generator']
  receipts=[load(ROOT/f'branches/{b}/receipts/{i}.json') for i in range(1,101)];assert len(list((ROOT/f'branches/{b}/receipts').glob('*.json')))==100
  for i,rec in enumerate(receipts,1):
   row=plan[i-1]['source_row'];assert rec['step']==i and rec['source_row_sha256']==plan[i-1]['source_row_canonical_sha256']
   assert rec['actual_frame_qps']==row['actual_frame_qps'] and rec['frozen_check']=='PASS' and rec['optimizer_internal_step']==i
   assert rec['reference_detach'] and rec['supervised_frames']==15 and rec['losses']['proxy_L1']==0
   assert rec['parameter_delta_norms']['generator']['norm']>0 and rec['parameter_delta_norms']['bridge']['norm']==0
   assert rec['postclip_norm']<=1.00001 and abs(rec['clip_coefficient']-min(1,1/(rec['preclip_norm']+1e-6)))<1e-12
   assert len(rec['frame_trace'])==15 and [x['position'] for x in rec['frame_trace']]==list(range(49,64))
  ev=[json.loads(l) for l in (ROOT/f'branches/{b}/optimizer_calls.jsonl').read_text().splitlines()]
  assert [e['step'] for e in ev if e['event']=='COMMITTED']==list(range(1,101))
  write(ROOT/f'results/train_{b}.csv',receipts)
  cpindex[b]={}
  for step in (0,50,100):
   p=ROOT/f'branches/{b}/checkpoints/adaptation_step_{step}.pt';st=torch.load(p,map_location='cpu',weights_only=True)
   assert st['committed_updates']==st['plan_cursor']==step and st['signature']==signature(b) and st['branch']==b
   assert {int(v['step']) for v in st['optimizer']['state'].values()}==({step} if step else set())
   assert st['optimizer']['param_groups'][0]['lr']==1e-6 and len(st['optimizer']['param_groups'])==1
   assert tensor_hash(st['bridge'])==source['initial_module_hashes']['bridge']
   assert all(t.dtype==torch.float32 for k in ('bridge','generator') for t in st[k].values() if t.is_floating_point())
   if step==0:
    for k in ('wrapper','bridge','generator'):assert tensor_hash(st[k])==tensor_hash(source[k])
    for mod,bufs in st['buffers'].items():assert all(torch.equal(value,source['buffers'][mod][name]) for name,value in bufs.items())
    assert st['crop_rng']['python']==source['crop_rng']['python']
    for k in ['torch','cuda']:assert torch.equal(st['crop_rng'][k],source['crop_rng'][k])
    assert st['crop_rng']['numpy'][0]==source['crop_rng']['numpy'][0] and torch.equal(st['crop_rng']['numpy'][1],source['crop_rng']['numpy'][1]) and st['crop_rng']['numpy'][2:]==source['crop_rng']['numpy'][2:]
    step0s.append(dict(branch=b,weights_equal=True,buffers_equal=True,full_training_RNG_equal=True,fresh_AdamW=len(st['optimizer']['state'])==0))
   if step==100:assert tensor_hash(st['generator'])==fin['final_module_hashes']['generator']!=source['initial_module_hashes']['generator']
   cpindex[b][str(step)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),committed_updates=step)
   bins.append(cpindex[b][str(step)]);del st
  inf=ROOT/f'evaluation/checkpoints/{b}_100.pt';assert sha(inf)==fin['inference_checkpoint_sha256'];bins.append(dict(path=str(inf),bytes=inf.stat().st_size,sha256=sha(inf)))
  training[b]=dict(status='PASS',committed_updates=100,optimizer_internal_step=100,ledger_commits=100,physical_APPLIED_calls=sum(e['event']=='APPLIED' for e in ev),discarded_replayed_calls=sum(e['event']=='DISCARDED_REPLAY' for e in ev),counts=dict(collections.Counter(x['domain'] for x in receipts)),initial=fin['initial_module_hashes'],final=fin['final_module_hashes'],frozen=fin['frozen_hashes'],only_Generator_changed=True)
 del source
 dump(ROOT/'audits/initialization_tensor_buffer_rng.json',dict(status='PASS',conditions=step0s))
 dump(ROOT/'audits/training_freeze_gradient_updates.json',dict(status='PASS',conditions=training,smoke_updates_counted=True,extra_diagnostic_optimizer_steps=0))
 dump(ROOT/'checkpoint_index.json',cpindex);dump(ROOT/'binary_artifacts_index.json',bins)
 rows=[t['result'] for t in tasks];write(ROOT/'evaluation.csv',rows);write(ROOT/'results/evaluation.csv',rows)
 paired=[]
 for a,b in PAIRS:
  for v in (0,1):
   for q in (0,5,9):
    x=next(r for r in rows if (r['method'],r['video'],r['QP'])==(a,v,q));y=next(r for r in rows if (r['method'],r['video'],r['QP'])==(b,v,q))
    for m in METRICS:
     d=x[m]-y[m];better=(d>1e-12) if m=='PSNR' else (d< -1e-12)
     paired.append(dict(comparison=a+' - '+b,video=v,sequence=x['sequence'],QP=q,metric=m,condition_value=x[m],reference_value=y[m],absolute_delta=d,unit='dB' if m=='PSNR' else 'bpp' if m=='actual_bpp' else 'metric units',relative_delta_percent=None if m=='PSNR' else 100*d/max(abs(y[m]),1e-12),improved=better,tied=abs(d)<=1e-12,actual_bpp=x['actual_bpp']))
 summary=[]
 for a,b in PAIRS:
  for video in (None,0,1):
   for m in METRICS:
    ps=[p for p in paired if p['comparison']==a+' - '+b and p['metric']==m and (video is None or p['video']==video)];n=len(ps)
    summary.append(dict(comparison=a+' - '+b,scope='six_point_equal' if video is None else 'video',video='all' if video is None else video,sequence='both videos' if video is None else ps[0]['sequence'],metric=m,points=n,distinct_videos=2 if video is None else 1,mean_condition_value=sum(p['condition_value'] for p in ps)/n,mean_reference_value=sum(p['reference_value'] for p in ps)/n,mean_absolute_delta=sum(p['absolute_delta'] for p in ps)/n,unit=ps[0]['unit'],mean_relative_delta_percent=None if m=='PSNR' else sum(p['relative_delta_percent'] for p in ps)/n,improved_points=sum(p['improved'] for p in ps),tied_points=sum(p['tied'] for p in ps),aggregation='equal weight per fixed anchor; multiple QPs are not independent videos'))
 for name,rr in [('paired_differences.csv',paired),('summary.csv',summary)]:write(ROOT/name,rr);write(ROOT/'results'/name,rr)
 deps=load(ROOT/'source_hashes.json');assert all(sha(p)==h for p,h in deps.items())
 source_entries={p:dict(path=p,bytes=Path(p).stat().st_size,sha256=h,role='immutable dependency') for p,h in deps.items()}
 for window in load(ROOT/'audits/training_windows.json')['windows']:
  p=window['source_path'];assert sha(p)==window['source_sha256'];source_entries[p]=dict(path=p,bytes=Path(p).stat().st_size,sha256=window['source_sha256'],role='selected raw training video')
  for entry in window['canonical_files']:
   p=entry['path'];assert sha(p)==entry['sha256'];source_entries[p]=dict(path=p,bytes=Path(p).stat().st_size,sha256=entry['sha256'],role='selected canonical training RGB')
 for v in load(ROOT/'eval_manifest.json')['hevc_b']:
  if v['video_index'] not in (0,1):continue
  for i in range(64):
   p=str(Path(v['input_dir'])/f'im{i+1:05d}.png');expected=v['frame_png_sha256'][i];assert sha(p)==expected;source_entries[p]=dict(path=p,bytes=Path(p).stat().st_size,sha256=expected,role='selected test RGB')
 dump(ROOT/'source_manifest.json',list(source_entries.values()))
 dump(ROOT/'audits/historical_readonly.json',dict(status='PASS',verified_files=len(deps),all_hashes_unchanged=True))
 events=[json.loads(l) for l in (ROOT/'logs/task_events.jsonl').read_text().splitlines()];starts=[e for e in events if e['event']=='START'];exits=[e for e in events if e['event']=='EXIT'];assert len(starts)==len(exits)
 assert {e['gpu'] for e in starts}<={4,5,6,7}
 exits=[dict(e,gpu=next(s['gpu'] for s in starts if s['PID']==e['PID'] and s['key']==e['key'])) for e in exits]
 for gpu in (4,5,6,7):
  intervals=sorted([(e['start'],e['end']) for e in exits if e['gpu']==gpu]);assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),gpu
 failed=[e for e in exits if e['exit_code']!=0]
 dump(ROOT/'audits/execution.json',dict(status='PASS',tasks_started=len(starts),tasks_exited=len(exits),failed_attempts=failed,one_experiment_task_per_GPU=True,physical_GPUs=sorted({e['gpu'] for e in starts})))
 final=dict(status='PASS',updates={'O_original':0,'G_basic':100,'G_dists':100},RD_complete=18,RD_planned=18,real_RANS_streams=6,independent_decode_points=18,same_stream_y_f_q_bridge=True,frame_records=18*64,transition_records=18*63,balanced_plan=True,train_video_counts={'Beauty':50,'U-Long':50},each_video_each_QP=5,distinct_training_time_windows=2,seed=20261034,diagnostic_optimizer_updates=0,checkpoint_evaluation='step100 only',FID='not computed',failed_attempts=len(failed),blocked=[])
 dump(ROOT/'final_integrity.json',final)
 lines=['# 基础损失与加入 DISTS 的对照：客观结果','', '计算与完整性检查：PASS。O_original 0 次更新；G_basic、G_dists 各 100 次已提交更新。18/18 实测评测点，6 条完整 RANS 码流，每点 64 帧、63 个真实相邻转换。','', '下表为六点等权平均的条件值减参考值。LPIPS/DISTS/FloLPIPS/L1 越低越好，PSNR 越高越好，PSNR 差值单位为 dB。感知指标相对变化不是码率节省。','', '|比较|指标|平均条件值|平均参考值|平均差值|平均相对变化 %|改善点数|','|---|---|---:|---:|---:|---:|---:|']
 for s in summary:
  if s['scope']=='six_point_equal':
   rel='—' if s['mean_relative_delta_percent'] is None else f"{s['mean_relative_delta_percent']:.6g}"
   lines.append(f"|{s['comparison']}|{s['metric']}|{s['mean_condition_value']:.9g}|{s['mean_reference_value']:.9g}|{s['mean_absolute_delta']:.9g}|{rel}|{s['improved_points']}/6|")
 lines+=['','每视频三点汇总见 summary.csv；全部逐点配对差值见 paired_differences.csv。多个 QP 不构成独立视频样本。','', '训练输入为 Beauty 和指定 U-Long 各一个固定历史窗口，各 50 个不同空间裁剪、每 QP 五次更新。两个条件使用同一个有序计划、相同原始 step0、buffers 和完整训练 RNG。仅 Generator 更新；Bridge、压缩核心和质量网络的冻结检查通过。rate 对 Generator 的梯度为零，Identity proxy_L1 为零；不描述为码率优化。','', '限制：单 seed、100 步、两段固定训练窗口、两个测试视频，属于趋势实验。基础损失不是未公开的官方完整训练目标，不能据此判断官方模型训练不足。不作机制归因或研究创新判断。不把历史 v6.38 不同计划下的 G-only 结果之差解释为 99:1→50:50 的纯因果效应。本报告未额外加入历史比较。','',f'执行记录包含 {len(failed)} 次失败尝试，详见 audits/execution.json 和 logs/。诊断不执行额外 optimizer.step，smoke 的前两次更新计入每分支100次。','', '训练 total loss 的定义不同，不作为两个条件训练优劣的直接可比指标。仅绘制共同分量与实际同码流评测曲线。']
 textfile(ROOT/'REPORT.md','\n'.join(lines)+'\n');textfile(ROOT/'RESULTS_SUMMARY.md','\n'.join(lines)+'\n')
 dump(ROOT/'run_status.json',dict(status='COMPUTE_COMPLETE',**{k:v for k,v in final.items() if k!='status'},time=time.time()))
 print('FINAL_INTEGRITY_PASS',flush=True)
if __name__=='__main__':main()
