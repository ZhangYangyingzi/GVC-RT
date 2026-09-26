import json
import shutil
from experiment_utils import ROOT, V4, sha, dump

for name in ('parts','logs','training_logs','checkpoints/beta_high','bitstreams','features','rd_curves'):
    (ROOT/name).mkdir(parents=True,exist_ok=True)
cfg=json.loads((V4/'config.json').read_text())
source=V4/'checkpoints/beta_high/step_10000.pt'
cfg.update(schema='gvcrt_v4_1_high_convergence_metrics', source_checkpoint=str(source),
           source_step=10000, target_step=20000, checkpoint_steps=[10000,15000,20000],
           beta=cfg['branches']['beta_high']['beta'], scheduler=None,
           metric_use='evaluation only', equal_rate_grid_points=100, rate_axis='log(kbps)',
           branches={'beta_high':cfg['branches']['beta_high']})
cfg['initial_checkpoint']=str(source)
if (ROOT/'config.json').exists():
    assert json.loads((ROOT/'config.json').read_text())==cfg
else: dump('config.json',cfg)
snapshot={}
for name in ('train_manifest.json','validation_manifest.json','test_manifest.json'):
    src=V4/name; dst=ROOT/name
    if not dst.exists(): shutil.copy2(src,dst)
    assert sha(src)==sha(dst); snapshot[str(src)]=sha(src)
snapshot[str(source)]=sha(source)
dst=ROOT/'checkpoints/beta_high/step_10000.pt'
if not dst.exists():
    temporary=dst.with_suffix('.tmp'); shutil.copyfile(source,temporary); temporary.replace(dst)
assert sha(dst)==snapshot[str(source)]
for src in V4.iterdir():
    if src.suffix in ('.json','.csv','.py'): snapshot[str(src)]=sha(src)
if (ROOT/'source_snapshot.json').exists():
    old=json.loads((ROOT/'source_snapshot.json').read_text())
    assert all(sha(p)==h for p,h in old.items())
else: dump('source_snapshot.json',snapshot)
splits={}
for name,count in [('train',256),('validation',6),('test',8)]:
    records=json.loads((ROOT/f'{name}_manifest.json').read_text())['videos']
    assert len(records)==count
    splits[name]={r.get('sha256',r.get('source_sha256')) for r in records}
    assert len(splits[name])==count and None not in splits[name]
assert not(splits['train'] & splits['validation'] or splits['train'] & splits['test'] or splits['validation'] & splits['test'])
dump('split_audit.json',dict(train_videos=256,validation_videos=6,test_videos=8,same_as_v4=True,disjoint=True,status='PASS'))
print('PREPARE PASS')
