import shutil
from v5_utils import ROOT,V41,V4,BRANCHES,sha,load,dump

def main():
    for name in ('parts','logs','training_logs','checkpoints','bitstreams','features','rd_curves'):(ROOT/name).mkdir(parents=True,exist_ok=True)
    assert load(V41/'final_integrity.json')['status']=='PASS'
    selection=load(V41/'selected_checkpoint.json');assert selection['step']==20000 and not selection['used_final_test']
    source=V41/'checkpoints/beta_high/step_20000.pt';assert sha(source)==selection['checkpoint_sha256']
    config=load(V41/'config.json')
    config.update(schema='gvcrt_v5a_clip_ablation',source_checkpoint=str(source),initial_checkpoint=str(source),source_step=20000,
                  branches=BRANCHES,optimized_P_frame_budget=21000,gpus=[4,5,6,7],scheduler=None,
                  selection='none; evaluate both fixed matched-budget endpoints',BPTT=False,
                  loss='(LPIPS+DISTS+beta*R_est_bpp+0.01*proxy_L1)/(clip_length-1)',
                  validation_methods=['original','v41_20000','clip4_control','clip8'],
                  result_gate_split='final; validation gates also recorded; no checkpoint selection')
    for k in ('updates','target_step','checkpoint_steps','clip_length'):config.pop(k,None)
    if (ROOT/'config.json').exists():assert load('config.json')==config
    else:dump('config.json',config)
    for split in ('train','validation','test'):
        src=V41/f'{split}_manifest.json';dst=ROOT/src.name
        if not dst.exists():shutil.copy2(src,dst)
        assert sha(src)==sha(dst)
    metric=V41/'metric_implementation_audit.json';assert load(metric)['status']=='PASS'
    if not (ROOT/metric.name).exists():shutil.copy2(metric,ROOT/metric.name)
    assert sha(metric)==sha(ROOT/metric.name)
    for path,digest in load(metric)['source_and_weight_sha256'].items():assert sha(path)==digest
    sources={str(p):sha(p) for base in (V41,V4) for p in base.iterdir() if p.suffix in ('.py','.json','.csv')}
    sources[str(source)]=sha(source)
    if (ROOT/'source_snapshot.json').exists():assert all(sha(p)==h for p,h in load('source_snapshot.json').items())
    else:dump('source_snapshot.json',sources)
    splits={s:{v.get('sha256',v.get('source_sha256')) for v in load(f'{s}_manifest.json')['videos']} for s in ('train','validation','test')}
    assert all(None not in values for values in splits.values())
    assert [len(splits[s]) for s in ('train','validation','test')]==[256,6,8]
    assert not (splits['train']&splits['validation'] or splits['train']&splits['test'] or splits['validation']&splits['test'])
    dump('split_audit.json',dict(status='PASS',train_videos=256,validation_videos=6,test_videos=8,disjoint=True,
         manifest_sha256={s:sha(ROOT/f'{s}_manifest.json') for s in splits}))
    dump('metric_protocol_reuse_audit.json',dict(status='PASS',source=str(metric),protocol_sha256=sha(metric),
         source_metric_runtime_sha256=sha(V41/'metric_runtime.py'),protocol_unchanged=True))
    print('PREPARE PASS',flush=True)

if __name__=='__main__':main()
