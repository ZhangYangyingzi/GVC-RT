"""Finish requested component-vs-other-base cosines without landing any update.

Raw initial gradients are invariant across the six configurations: only fixed
weights, clip and learning rates differ. Reuse one exact pre-step reference per
prescribed window, then associate the same raw cosine with its six existing cases.
"""
from audit31 import *
import argparse,ast,subprocess,traceback,textwrap
def reference(t,gpu):
 source=(ROOT/'probe_worker.py').read_text();tree=ast.parse(source)
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 stop=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='opt' and n.value.func.attr=='step')
 fn.body=fn.body[:stop]
 tail='''
 records=[]
 for name,buff in zip(component_names[:4],component):
  weight=weights[name]
  if name=='stat':
   other=base;g=buff
  else:
   g=[a*weight for a in buff];other=[a-b for a,b in zip(base,g)]
  ng,no=norm(g),norm(other)
  cosine=dot(g,other)/max(ng*no,1e-30)
  records.append(dict(component=name,raw_cosine_with_other_base=cosine,raw_gradient_norm=norm(buff),other_base_norm=no,finite=math.isfinite(cosine)))
 assert initial['wrapper']==v4.module_hash(models['wrapper']) and core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality]
 assert all(initial[k]==v4.module_hash(models[k]) for k in ('bridge','generator'))
 assert not opt.state
 hook.remove()
 dump(ROOT/'audits/cosines'/f'window_{t["video"]:02d}.json',dict(status='PASS',calibration_index=t['video'],domain=t['dataset'],QP=t['qp'],records=records,source_checkpoint_sha256=cfg['source_checkpoint']['sha256'],probe_source_sha256=sha(ROOT/'probe_worker.py'),AST_reference_stops_before_optimizer_step=True,optimizer_updates=0,formal_updates=0,D_updates=0,input_plan=row,frozen_hashes=dict(wrapper=initial['wrapper'],compression=core,quality=qh),peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
 print('GRADIENT_REFERENCE_PASS',t['video'],flush=True)
'''
 fn.body+=ast.parse(textwrap.dedent(tail)).body
 ref_ast=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
 probe=module('v631_readonly_probe_source',ROOT/'probe_worker.py');ns=dict(probe.__dict__);ns['valid']=lambda t:False
 ns['receipt']=lambda t:ROOT/'audits/cosines'/('ref_'+uid(t)+'.json')
 exec(compile(ref_ast,str(ROOT/'gradient_cosines31.py'),'exec'),ns);ns['main'](t,gpu)
def launch():
 rows=load(ROOT/'probe_manifest.json')['updates'];(ROOT/'audits/cosines').mkdir(parents=True,exist_ok=True)
 from run_all import resources
 for row in rows:
  p=ROOT/'audits/cosines'/f'window_{row["calibration_index"]:02d}.json'
  if p.exists():
   r=load(p);assert r['status']=='PASS' and r['optimizer_updates']==0 and r['probe_source_sha256']==sha(ROOT/'probe_worker.py');continue
  t=dict(stage='probe',dataset=row['domain'],video=row['calibration_index'],method='C0',qp=row['external_qp'])
  while True:
   free,host=resources();need=load(receipt(t));g=max(free,key=free.get)
   if free[g]>=1.25*need['peak_memory_MiB']+2048 and host>=1.25*need['peak_RSS_MiB']+8192:break
   time.sleep(60)
  argv=[PYTHON,'-B','-u',str(ROOT/'gradient_cosines31.py'),'--task',json.dumps(t),'--gpu',str(g)]
  with (ROOT/'logs/gradient_references.log').open('a') as log:
   subprocess.run(argv,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');ap.add_argument('--task');ap.add_argument('--gpu',type=int,choices=(4,5,6,7));a=ap.parse_args()
 try:launch() if a.launch else reference(json.loads(a.task),a.gpu)
 except Exception:error(dict(gradient_reference=True,traceback=traceback.format_exc()));raise
