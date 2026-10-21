"""Verify training/evaluation, import hash-identical fixed v6.40 controls, and report."""
from io32 import *
import collections,math,shutil
OLD=ROOT.parent/'gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100'
METHODS=('O_original','G_basic','G_L1')
METRICS=('LPIPS','DISTS','FloLPIPS','PSNR','L1','actual_bpp')
PAIRS=(('G_basic','O_original'),('G_L1','O_original'),('G_L1','G_basic'))

def reuse_controls():
    assert load(OLD/'final_integrity.json')['status']=='PASS'
    for name in ('plans/first100.json','evaluation_config.json','eval_manifest.json','manifests/streams.json'):
        assert sha(ROOT/name)==sha(OLD/name),(name,'reference changed')
    deps=load(OLD/'source_hashes.json')
    for path,digest in deps.items():assert Path(path).is_file() and sha(path)==digest,path
    gfin=load(OLD/'branches/G_basic/final_integrity.json');assert gfin['status']=='PASS' and gfin['committed_updates']==100
    gcp=OLD/'evaluation/checkpoints/G_basic_100.pt';assert sha(gcp)==gfin['inference_checkpoint_sha256']
    reused=[]
    for method in METHODS[:2]:
      for v in (0,1):
       for q in (0,5,9):
        key=f'{method}_v{v}_q{q}';src=OLD/'tasks'/f'{key}.json';task=load(src);assert task['status']=='PASS'
        rr=task['result'];sf=OLD/'results/frames'/f'{key}.csv';st=OLD/'results/transitions'/f'{key}.csv'
        assert sha(sf)==rr['frame_csv_sha256'] and sha(st)==rr['transition_csv_sha256']
        assert len(read(sf))==64 and len(read(st))==63
        df=ROOT/'results/frames'/f'{key}.csv';dt=ROOT/'results/transitions'/f'{key}.csv';df.parent.mkdir(parents=True,exist_ok=True);dt.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(sf,df);shutil.copy2(st,dt);assert sha(df)==sha(sf) and sha(dt)==sha(st)
        local=dict(rr);local.update(frame_csv_sha256=sha(df),transition_csv_sha256=sha(dt),
          reused_from_v640=True,reuse_source_task=str(src),reuse_source_task_sha256=sha(src),
          reuse_verification='config/checkpoint/code-dependencies/stream/raw-CSV hashes PASS')
        dump(ROOT/'tasks'/f'{key}.json',dict(status='PASS',signature=signature('eval_'+key),
          bridge_hash=task['bridge_hash'],result=local,decoded_features=task['decoded_features'],
          frozen_model_hashes=task['frozen_model_hashes'],physical_GPU='reused',
          reuse_source_task_sha256=sha(src)))
        reused.append(dict(key=key,task_sha256=sha(src),frame_sha256=sha(sf),transition_sha256=sha(st)))
    dump(ROOT/'audits/v640_result_reuse.json',dict(status='PASS',reference=str(OLD),
      dependency_files_verified=len(deps),evaluation_config_sha256=sha(ROOT/'evaluation_config.json'),
      G_basic_checkpoint_sha256=sha(gcp),tasks=reused,reused_points=12))

def main():
    import torch
    reuse_controls();plan=load(ROOT/'plans/first100.json')['updates'];assert len(plan)==100
    diag=load(ROOT/'audits/objective_gradient_step0.json');assert diag['status']=='PASS' and diag['diagnostic_optimizer_step_calls']==0
    branch=ROOT/'branches/G_L1';fin=load(branch/'final_integrity.json');assert fin['status']=='PASS' and fin['committed_updates']==100
    receipts=[load(branch/'receipts'/f'{i}.json') for i in range(1,101)];assert len(list((branch/'receipts').glob('*.json')))==100
    for i,r in enumerate(receipts,1):
      assert r['step']==r['optimizer_internal_step']==i and r['source_row_sha256']==plan[i-1]['source_row_canonical_sha256']
      assert r['frozen_check']=='PASS' and r['reference_detach'] and r['supervised_frames']==15
      assert r['losses']['proxy_L1']==0 and r['losses']['weighted_LPIPS']==0 and r['losses']['weighted_DISTS']==0
      assert r['gradient_modules']['generator']['norm']>0 and r['parameter_delta_norms']['generator']['norm']>0
      assert r['parameter_delta_norms']['bridge']['norm']==0 and r['postclip_norm']<=1.00001
    events=[json.loads(x) for x in (branch/'optimizer_calls.jsonl').read_text().splitlines()]
    assert [e['step'] for e in events if e['event']=='COMMITTED']==list(range(1,101))
    assert sum(e['event']=='APPLIED' for e in events)==100
    cpindex={'G_L1':{}}
    for step in (0,50,100):
      p=branch/f'checkpoints/adaptation_step_{step}.pt';s=torch.load(p,map_location='cpu',weights_only=True)
      assert s['committed_updates']==s['plan_cursor']==step
      assert {int(v['step']) for v in s['optimizer']['state'].values()}==({step} if step else set())
      cpindex['G_L1'][str(step)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),committed_updates=step)
    write(ROOT/'results/train_G_L1.csv',receipts)
    tasks=[]
    for m in METHODS:
      for v in (0,1):
       for q in (0,5,9):
        key=f'{m}_v{v}_q{q}';t=load(ROOT/'tasks'/f'{key}.json');assert t['status']=='PASS' and t['signature']==signature('eval_'+key)
        rr=t['result'];fr=ROOT/'results/frames'/f'{key}.csv';tr=ROOT/'results/transitions'/f'{key}.csv'
        assert sha(fr)==rr['frame_csv_sha256'] and sha(tr)==rr['transition_csv_sha256']
        fs=read(fr);ts=read(tr);assert len(fs)==64 and len(ts)==63
        assert rr['real_bytes']==Path(rr['bitstream_path']).stat().st_size and sha(rr['bitstream_path'])==rr['bitstream_sha256']
        assert rr['actual_bpp']==rr['real_bytes']*8/(64*1080*1920)
        tasks.append(t)
    streams=[]
    for v in (0,1):
      for q in (0,5,9):
        group=[t for t in tasks if (t['result']['video'],t['result']['QP'])==(v,q)]
        assert len({json.dumps(t['decoded_features'],sort_keys=True) for t in group})==1
        assert len({(t['result']['bitstream_sha256'],t['result']['real_bytes'],t['bridge_hash']) for t in group})==1
        assert len(group[0]['decoded_features'])==63
        streams.append(dict(video=v,QP=q,bytes=group[0]['result']['real_bytes'],sha256=group[0]['result']['bitstream_sha256'],
          identical_y_f_q_and_frozen_bridge_output=True,methods=list(METHODS),independent_G_L1_decode=True))
    dump(ROOT/'audits/same_stream_features_bridge.json',dict(status='PASS',streams=streams))
    rows=[t['result'] for t in tasks];write(ROOT/'evaluation.csv',rows);write(ROOT/'results/evaluation.csv',rows)
    paired=[]
    for a,b in PAIRS:
      for v in (0,1):
       for q in (0,5,9):
        x=next(r for r in rows if (r['method'],r['video'],r['QP'])==(a,v,q));y=next(r for r in rows if (r['method'],r['video'],r['QP'])==(b,v,q))
        for m in METRICS:
          d=x[m]-y[m]
          paired.append(dict(comparison=f'{a} - {b}',video=v,sequence=x['sequence'],QP=q,metric=m,
            condition_value=x[m],reference_value=y[m],absolute_delta=d,unit='dB' if m=='PSNR' else 'bpp' if m=='actual_bpp' else 'metric units',
            relative_delta_percent=None if m in ('PSNR','actual_bpp') else 100*d/max(abs(y[m]),1e-12)))
    write(ROOT/'paired_differences.csv',paired);write(ROOT/'results/paired_differences.csv',paired)
    summary=[]
    for a,b in PAIRS:
      for video in (None,0,1):
       for m in METRICS:
        ps=[p for p in paired if p['comparison']==f'{a} - {b}' and p['metric']==m and (video is None or p['video']==video)];n=len(ps)
        summary.append(dict(comparison=f'{a} - {b}',scope='six_point_equal' if video is None else 'video',video='all' if video is None else video,
          sequence='both videos' if video is None else ps[0]['sequence'],metric=m,points=n,distinct_videos=2 if video is None else 1,
          mean_condition_value=sum(p['condition_value'] for p in ps)/n,mean_reference_value=sum(p['reference_value'] for p in ps)/n,
          mean_absolute_delta=sum(p['absolute_delta'] for p in ps)/n,unit=ps[0]['unit'],
          mean_relative_delta_percent=None if m in ('PSNR','actual_bpp') else sum(p['relative_delta_percent'] for p in ps)/n,
          aggregation='equal weight per video/QP anchor; six anchors come from two videos'))
    write(ROOT/'summary.csv',summary);write(ROOT/'results/summary.csv',summary)
    dump(ROOT/'audits/training_freeze_gradient_updates.json',dict(status='PASS',committed_updates=100,applied_optimizer_calls=100,
      smoke_updates_counted=True,diagnostic_optimizer_calls=0,only_generator_changed=True,final_hashes=fin['final_module_hashes'],
      frozen_hashes=fin['frozen_hashes'],all_receipts_verified=True))
    dump(ROOT/'checkpoint_index.json',cpindex)
    bins=[x for x in cpindex['G_L1'].values()];inf=ROOT/'evaluation/checkpoints/G_L1_100.pt';bins.append(dict(path=str(inf),bytes=inf.stat().st_size,sha256=sha(inf)))
    dump(ROOT/'binary_artifacts_index.json',bins)
    source_entries=[]
    for p in [ROOT/'plans/first100.json',ROOT/'protocol.json',ROOT/'evaluation_config.json',OLD/'protocol.json',OLD/'evaluation/checkpoints/G_basic_100.pt']:
      source_entries.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    for x in load(ROOT/'manifests/streams.json'):
      p=Path(x['bitstream_path']);source_entries.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    dump(ROOT/'source_manifest.json',source_entries)
    lines=['# G_L1 与 G_basic 固定对照结果','',
      '完整性检查：PASS。G_L1 恰好 100 次已提交更新；18/18 评测点完成。O_original 与 G_basic 的 12 点经依赖、checkpoint、码流和原始 CSV 哈希核验后复用 v6.40；G_L1 六点独立解码。','',
      '六点等权汇总如下；六点来自两个视频的三个 QP，不能视为六个独立视频样本。PSNR 差值单位为 dB；感知指标相对变化不表示码率节省。','',
      '|比较|指标|条件均值|参考均值|平均差值|平均相对变化 %|','|---|---|---:|---:|---:|---:|']
    for s in summary:
      if s['scope']=='six_point_equal':
        rel='—' if s['mean_relative_delta_percent'] is None else f"{s['mean_relative_delta_percent']:.8g}"
        lines.append(f"|{s['comparison']}|{s['metric']}|{s['mean_condition_value']:.10g}|{s['mean_reference_value']:.10g}|{s['mean_absolute_delta']:.10g}|{rel}|")
    lines += ['', '逐点差值见 paired_differences.csv；逐视频三点汇总见 summary.csv。未计算 FID。', '',
      '训练条件：同一原始 step0、buffers、完整训练 RNG 和 first100 固定计划；batch 1，256×256，I0 与 P1–P48 无梯度，P49–P63 共 15 帧监督，detached reference，无 BPTT。G_L1 仅更新 Generator，使用 0.1383589248954108×reconstruction_L1 + beta_q(schedule_s1p0)×rate_bpp + 0.01×proxy_L1；LPIPS 和 DISTS 均不进入训练目标。Identity proxy 和冻结压缩核心对应的 Generator 梯度审计为零，因此不声称码率优化。', '',
      '限制：单 seed、100 次更新、两个固定训练窗口、两个评测视频。结果仅记录该固定实验条件下的数字；不作机制归因、统计显著性、泛化或创新性判断。G_basic 是 v6.40 固定基础目标，不代表官方完整训练目标。']
    textfile(ROOT/'REPORT.md','\n'.join(lines)+'\n')
    textfile(ROOT/'README.md','# v6.41 Generator L1 vs LPIPS balanced trend (100 updates)\n\n执行产物与客观结果见 `REPORT.md`。协议见 `PROTOCOL.md` 和 `protocol.json`。\n')
    textfile(ROOT/'PROTOCOL.md','# Protocol\n\nG_L1 从指定原始 step0 开始，只训练 Generator 100 次；复用 v6.40 的不可变 first100 计划、优化器设置与评测码流。目标为 `0.1383589248954108 * reconstruction_L1 + beta_q(schedule_s1p0) * rate_bpp + 0.01 * proxy_L1`，不含 LPIPS 或 DISTS。O_original 与 G_basic 固定结果来自经哈希核验的 v6.40。\n')
    textfile(ROOT/'OBJECTIVE_COMPARISON.md','# Objective comparison\n\n`G_basic` 使用 v6.40 已完成的固定基础目标。`G_L1` 的唯一有意训练目标差异是去掉 LPIPS；其余数据、初始化、计划、优化器与反向实现相同。该基础目标不称为 GVC-RT 官方完整训练目标。\n')
    final=dict(status='PASS',updates={'O_original':0,'G_basic':100,'G_L1':100},RD_complete=18,RD_planned=18,
      reused_verified_points=12,new_independent_decode_points=6,real_RANS_streams=6,frame_records=18*64,transition_records=18*63,
      same_stream_y_f_q_bridge=True,plan_sha256=sha(ROOT/'plans/first100.json'),diagnostic_optimizer_updates=0,FID='not computed',blocked=[])
    dump(ROOT/'final_integrity.json',final);dump(ROOT/'run_status.json',dict(status='COMPUTE_COMPLETE',**{k:v for k,v in final.items() if k!='status'}))
    excluded={'artifact_index.json','publication_status.json','run_status.json','audits/publication_content_verification.json'}
    public=[]
    for p in sorted(ROOT.rglob('*')):
      if not p.is_file():continue
      rel=str(p.relative_to(ROOT))
      if rel in excluded or p.suffix in ('.pt','.npz','.npy','.bin','.lock','.tmp') or '__pycache__' in p.parts:continue
      public.append(dict(relative_path=rel,bytes=p.stat().st_size,sha256=sha(p)))
    dump(ROOT/'artifact_index.json',dict(status='PASS',public=public,public_files=len(public)))
    print('FINAL_INTEGRITY_PASS',len(rows),flush=True)
if __name__=='__main__':main()
