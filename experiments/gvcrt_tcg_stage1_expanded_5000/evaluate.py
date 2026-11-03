"""Fixed four-checkpoint heldout evaluation, after successful training only."""
import csv,importlib.util,inspect,math,os,subprocess,sys,time,traceback
from common_expanded import *
METRICS=('bpp','kbps','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError')
def write(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fields=fields or list(dict.fromkeys(k for r in rows for k in r));tmp=path.with_suffix('.tmp')
 with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
 tmp.replace(path)
def legacy():
 p=REPO/'experiments/gvcrt_tcg_stage1_evaluation/evaluate.py'
 spec=importlib.util.spec_from_file_location('original_tcg_evaluation',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 # Legacy module fixes GPU 5 at import; the worker immediately restores its own
 # explicitly assigned physical GPU before torch is imported/initialized.
 return m
def worker(rank):
 base=legacy();gpu=4+rank
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch
 assert torch.cuda.is_available();torch.cuda.set_device(0)
 sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'experiments/gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
 sys.path.insert(0,str(REPO/'experiments/gvcrt_parallel_v5a3_v60'))
 sys.path.insert(0,str(REPO/'experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100'))
 import engine
 from adapter35 import frames_for
 from src.utils.common import set_torch_env
 set_torch_env();torch.set_num_threads(2)
 quality=engine.core.quality_models(torch.device('cuda:0'))
 flow,perceptual=engine.metrics.flo.load_models(torch.device('cuda:0'))
 base.ROOT=ROOT/'evaluation';base.LARGE=ROOT/'large_artifacts/evaluation'
 base.LARGE.mkdir(parents=True,exist_ok=True)
 # Reuse the previously audited real-RANS implementation and add only the
 # prescribed warp measurement; no encoded bytes or decoded outputs are reused.
 code=inspect.getsource(base.run_point)
 code=code.replace('def run_point(engine,runtime,quality,flow,perceptual,frames,v,qp,step,cp,key):',
                   'def run_point(engine,runtime,quality,flow,perceptual,frames,v,qp,step,cp,key,gpu):')
 target="trans.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=value))"
 assert code.count(target)==1
 code=code.replace(target,"warped=sys.modules['pwcnet'].backwarp(previous_out,flow(out,previous_out))\n                warp_error=float((gt-warped).abs().mean())\n                assert math.isfinite(warp_error)\n                rows[-1]['WarpingError']=warp_error\n                trans.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=value,WarpingError=warp_error))")
 code=code.replace("FloLPIPS=sum(r['FloLPIPS'] for r in trans)/63,","FloLPIPS=sum(r['FloLPIPS'] for r in trans)/63,WarpingError=sum(r['WarpingError'] for r in trans)/63,")
 code=code.replace('physical_GPU=5','physical_GPU=gpu')
 ns=dict(base.__dict__);ns['METRICS']=METRICS;ns['write_csv']=write
 exec(compile(code,str(ROOT/'evaluate.py'),'exec'),ns);run_point=ns['run_point']
 videos=[v for vs in load(ROOT/'evaluation/manifest.local.json').values() for v in vs]
 # Held-out native ULong rows are not training-manifest entries. Bind metadata
 # from the audited held-out manifest, while retaining PlanFrames RGB hashes.
 import frames35
 frames35._SOURCES['ulong']={v['source_row']['video']:dict(path=v['source_row']['source_path'],
      sha256=v['source_row']['source_sha256'],width=v['width'],height=v['height'])
      for v in videos if v['dataset']=='ulong'}
 tasks=[(s,v,q) for s in (0,1000,2500,5000) for v in videos for q in (0,5,9)]
 originals=torch.load(SOURCE,map_location='cpu',weights_only=True)['models']
 results=[]
 for index,(step,v,qp) in enumerate(tasks):
  if index%4!=rank:continue
  path=ROOT/f'checkpoints/step_{step}.pt';st=torch.load(path,map_location='cpu',weights_only=True)
  assert st['committed_global_updates']==step and st['protocol_sha256']==sha(ROOT/'protocol.json')
  payload={k:dict(s) for k,s in originals.items()};payload['generator'].update(st['tcg_model_state'])
  class Runtime(engine.Runtime):
   def __init__(self):self.device=torch.device('cuda:0');self.wrapper=None;self.config=dict(force_zero_thres=.12);self.payload=payload
  frames=frames_for(v);runtime=Runtime();key=f"step{step}_{v['dataset']}_{v['name']}_qp{qp}"
  result=run_point(engine,runtime,quality,flow,perceptual,frames,v,qp,step,
                   dict(filename=path.name,sha256=sha(path)),key,gpu)
  results.append(result);write(ROOT/f'evaluation/worker_{rank}_metrics.csv',results)
  event('EVALUATION_POINT_PASS',worker=rank,step=step,sequence=v['name'],QP=qp)
  if hasattr(frames,'close'):frames.close()
  del frames,runtime,st
 dump(ROOT/f'evaluation/worker_{rank}_status.json',dict(status='PASS',points=len(results),physical_GPU=gpu))
def main():
 assert load(ROOT/'audits/training_integrity.json')['status']=='PASS'
 assert load(ROOT/'training_summary.json')['committed_global_updates']==5000
 protocol=load(ROOT/'protocol.json');held=load(ROOT/'evaluation/manifest.local.json')
 dump(ROOT/'evaluation/protocol.json',dict(training_protocol_sha256=sha(ROOT/'protocol.json'),
       checkpoint_steps=[0,1000,2500,5000],primary_checkpoint=5000,QP=[0,5,9],frames=64,
       video_count=sum(len(x) for x in held.values()),domains=list(held),no_validation_selection=True,
       fresh_encode_per_checkpoint=True,cross_checkpoint_stream_reuse=False,
       independent_decode=True,TCG_sequence_GOP_resets=True,
       warping_error=protocol['warping_error'],warping_direction=protocol['warping_direction'],
       warping_sampler='existing official-compatible pwcnet.backwarp, bilinear/zero padding/align_corners=False and existing binary sampling mask'))
 ps=[]
 for rank in range(4):
  f=(ROOT/f'logs/evaluation_worker_{rank}.log').open('a')
  ps.append((subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'evaluate.py'),'--worker',str(rank)],cwd=REPO,stdout=f,stderr=subprocess.STDOUT),f))
 codes=[p.wait() for p,f in ps]
 for p,f in ps:f.close()
 if any(codes):
  dump(ROOT/'evaluation/run_status.json',dict(status='FAILED',worker_exit_codes=codes));raise RuntimeError('Evaluation worker failed')
 rows=[]
 for rank in range(4):
  with (ROOT/f'evaluation/worker_{rank}_metrics.csv').open() as f:
   for r in csv.DictReader(f):
    r.update(step=int(r['step']),QP=int(r['QP']),frames=int(r['frames']),real_bytes=int(r['real_bytes']))
    for k in METRICS:r[k]=float(r[k])
    rows.append(r)
 expected=4*3*sum(len(vs) for vs in held.values());assert len(rows)==expected
 rows.sort(key=lambda r:(r['domain'],r['sequence'],r['QP'],r['step']))
 deltas=[]
 for r in rows:
  if r['step']==0:continue
  a=next(x for x in rows if x['sequence']==r['sequence'] and x['domain']==r['domain'] and x['QP']==r['QP'] and x['step']==0)
  deltas.append(dict(domain=r['domain'],sequence=r['sequence'],QP=r['QP'],step=r['step'],
        **{f'delta_{k}':r[k]-a[k] for k in METRICS}))
 write(ROOT/'metrics.csv',rows);write(ROOT/'deltas.csv',deltas)
 write(ROOT/'warping_error.csv',[{k:r[k] for k in ('domain','sequence','QP','step','WarpingError')} for r in rows])
 dump(ROOT/'evaluation/run_status.json',dict(status='PASS',completed_points=expected,primary_checkpoint=5000))
 dump(ROOT/'final_integrity.json',dict(status='PASS',training=load(ROOT/'audits/training_integrity.json'),
       evaluation_points=expected,heldout_split=load(ROOT/'split_audit.json')['status'],fresh_encode=True,independent_decode=True))
 lines=['# TCG Stage 1 expanded 5000','',
        'Training: PASS. 5000 committed global updates; four ranks; effective batch 4.',
        'Training identities: 940. Planned rank samples: 20000. Exact video/window/crop/QP repeats: 0.',
        'The schedule reuses 4999 audited video/window/crop content keys across distinct QPs: four QPs per source row, with one shared content key appearing at eight distinct QPs. This is controlled content reuse, not 20000 unique frame windows.',
        'Detached 64-frame causal training, 48 P-frame warmup and 15 supervised P frames.',
        'Runtime revisions: fixed-plan input prefetch and exact frozen-device-byte checks from update 2456; separate frozen RAFT target-flow processes from update 2624. The optimizer, batch, plan, precision, objective and 5000-update budget are unchanged.',
        'Runtime protocol and numerical replay audits: runtime_protocol_addendum.json and audits/.',
        'Primary checkpoint: step 5000. Additional fixed checkpoints: step 0, 1000, 2500.',
        f'Evaluation: PASS; {expected} fresh encode/independent decode points.',
        'Numerical details: metrics.csv, deltas.csv, warping_error.csv and evaluation/frames, evaluation/transitions.',
        'Limits: one seed, fixed existing audited training pool and heldout manifest; no parameter or checkpoint selection.','',
        '| Domain | Sequence | QP | Step | '+' | '.join(METRICS)+' |',
        '|---|---|---:|---:|'+ '|'.join(['---:']*len(METRICS))+'|']
 for r in rows:lines.append('| '+r['domain']+' | '+r['sequence']+' | '+str(r['QP'])+' | '+str(r['step'])+' | '+' | '.join(f'{r[k]:.9f}' for k in METRICS)+' |')
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 try:
  if '--worker' in sys.argv:worker(int(sys.argv[-1]))
  else:main()
 except Exception as e:
  traceback.print_exc();sys.exit(1)
