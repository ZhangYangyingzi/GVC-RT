"""Freeze prescribed inputs, checkpoint evidence, and fixed task identities."""
from audit31 import *
import inspect,subprocess
def main():
 import torch
 cfg=load(V29/'config.json');ec=load(V29/'evaluation/config.json')
 for n in ('logs','audits','tasks','imports'): (ROOT/n).mkdir(exist_ok=True)
 old_files=['evaluation/config.json','CONFIG_RESOLVED.json','protocol.json','qp_semantics_audit.json','calibration/final.json','plans/calibration.json','config.json','train.py','window_training.py','stat_loss.py','io21.py','adapter.py','train_manifest_ulong.json','train_manifest_uvg.json']
 hashes={}
 for rel in old_files:
  p=V29/rel;assert p.exists(),('missing logical source',str(p));hashes[str(p)]=sha(p)
  textfile(ROOT/'imports/v629'/rel,p.read_text())
 for rel in ('audits/training_deployment_checkpoints.json','audits/cache_closure_integrity.json','old_sources.json','protocol.json','final_integrity.json','worker.py','adapter.py','io21.py','fid_statistics.py'):
  p=V30/rel;hashes[str(p)]=sha(p);textfile(ROOT/'imports/v630'/rel,p.read_text())
 assert load(V30/'audits/cache_closure_integrity.json')['status']=='PASS'
 assert cfg['source_checkpoint']['sha256']=='dc76f0e221dc7aaa5e5f45f648722584ce226e65828ce24151f9961985f3f6f7'
 audits={}
 for name,cp in [('training',cfg['source_checkpoint']),('deployment',ec['checkpoints']['B1_initial'])]:
  assert sha(cp['path'])==cp['sha256'];hashes[cp['path']]=cp['sha256']
  state=torch.load(cp['path'],map_location='cpu',weights_only=True)
  actual={k:tensor_hash(state[k]) for k in ('wrapper','bridge','generator')};assert actual==cp['module_hashes'];del state
  audits[name]=dict(cp,actual_module_hashes=actual)
 assert audits['training']['actual_module_hashes']==audits['deployment']['actual_module_hashes']
 assert audits['training']['actual_module_hashes']['wrapper']=='08cfe75a5292c9ad874c6ab593a418fbc425c5ac9182ba4edaf4f779a441d0a9'
 sources=load(V30/'old_sources.json');assert len(sources['hevc_b'])==5 and len(sources['mcljcv'])==30
 selected={d:sorted([v for v in vs if d!='mcljcv' or 'videoSRC17' not in v['name']],key=lambda v:v['video_index'])[:(3 if d=='hevc_b' else 6)] for d,vs in sources.items()}
 plan=load(V29/'plans/calibration.json');rows=[dict(r,calibration_index=i) for i,r in enumerate(plan['updates']) if r['external_qp'] in QPS]
 assert [r['calibration_index'] for r in rows]==[0,1,10,11,18,19,20,21,30,31,38,39]
 assert load(V29/'calibration/final.json')['lambda_stat']==LAM
 # Effective historical AdamW constructor omits betas/eps. Resolve the installed signature,
 # and verify source omission explicitly rather than assuming an undocumented default.
 src=(V29/'train.py').read_text();constructor=next(s for s in src.splitlines() if 'opt=torch.optim.AdamW(' in s)
 sig=inspect.signature(torch.optim.AdamW);betas=sig.parameters['betas'].default;eps=sig.parameters['eps'].default
 effective=dict(betas=list(betas),eps=eps,source=str(V29/'train.py'),constructor=constructor,installed_signature=str(sig),torch_version=torch.__version__,declaration='omitted in historical constructor; installed default is effective')
 cfg.update(schema='v631',updates=0,D_updates=0,temporary_updates=72,lambda_stat=LAM,seed_probe=load(plan and V29/'plans/calibration.json')['seed'],optimizer_effective=effective,external_qps=list(QPS),methods=list(METHODS),cells=list(CELLS),expected=dict(color=210,intervention=135,reused=54,new=81,probe=72),reference_commit='1bb6d29cfb017a1be7e1017edf4b3c6e68e406c3')
 dump(ROOT/'config.json',cfg);dump(ROOT/'evaluation_config.json',ec)
 dump(ROOT/'checkpoint_audit.json',dict(status='PASS',**audits,runtime_FP16_receiver_hashes=ec['deployment_receiver_hashes']['B1_initial'],original_receiver_hashes=ec['original_receiver_hashes'],effective_optimizer=effective,hash_algorithm='state_dict key insertion order + contiguous native dtype bytes SHA256'))
 protocol=load(V30/'protocol.json');protocol.update(experiment='v631',QPs=list(QPS),IP=-1,color_diagnostic='BT709 centered chroma, HSV saturation',correction='oracle fit P1-31, eval P32-63',P_frozen=True,trainable_probe=['bridge','generator'],formal_updates=0,temporary_updates=72)
 dump(ROOT/'protocol.json',protocol);dump(ROOT/'eval_manifest.json',sources)
 dump(ROOT/'intervention_manifest.json',dict(selection='video_index ascending; MCL excludes videoSRC17; first3 HEVC/first6 MCL',videos=selected,cells=list(CELLS)))
 dump(ROOT/'probe_manifest.json',dict(source=str(V29/'plans/calibration.json'),source_sha256=sha(V29/'plans/calibration.json'),updates=rows))
 dump(ROOT/'qp_semantics_audit.json',load(V29/'qp_semantics_audit.json'))
 tasks=[]
 for d,vs in sources.items():
  for v in vs:
   for m in METHODS:
    for q in QPS:
     t=dict(stage='color',dataset=d,video=v['video_index'],method=m,qp=q);tasks.append(t)
     p=oldpoint(d,m,v['video_index'],q);r=load(p);hashes[str(p)]=sha(p)
     for key in ('bitstream','feature'):
      assert sha(r[key+'_path'])==r[key+'_sha256'];hashes[r[key+'_path']]=r[key+'_sha256']
     closure=V30/'tasks'/f'closure_{d}_{v["video_index"]}_{m}_{q}.json';cr=load(closure);assert cr['status']=='PASS' and cr['reconstruction_hash_match'];hashes[str(closure)]=sha(closure)
     assert r['frames']==64 and r['source_frame_indices']==v['source_frame_indices'];assert r['actual_qps']==load(V29/'qp_semantics_audit.json')['actual_qps'][str(q)][:64]
 for d,vs in selected.items():
  for v in vs:
   for c in CELLS:
    for q in QPS:tasks.append(dict(stage='intervention',dataset=d,video=v['video_index'],method=c,qp=q))
 for r in rows:
  for c in range(6):tasks.append(dict(stage='probe',dataset=r['domain'],video=r['calibration_index'],method=f'C{c}',qp=r['external_qp']))
 dump(ROOT/'task_manifest.json',dict(tasks=tasks))
 # Snapshot all imported numerical code plus core model/metric dependencies.
 deps=load(V29/'audits/dependencies.json')
 for p,h in deps.items():
  if Path(p).suffix=='.py':assert sha(p)==h;hashes[p]=h
 dump(ROOT/'source_hashes.json',dict(hashes=hashes,metadata={p:dict(bytes=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns) for p in hashes}))
 dump(ROOT/'audits/preflight.json',dict(status='PASS',task_counts=cfg['expected'],optimizer=effective,formal_updates=0,D_updates=0))
 print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()
