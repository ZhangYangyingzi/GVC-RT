"""Offline GT-source PWC scoring; decoder/TCG inputs never receive GT flow."""
from ab_common import *
import csv,importlib.util,inspect,io,traceback
METRICS=('bpp','kbps','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC')
def write(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with temp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
 temp.replace(path)
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def setup(rank):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(4+rank))
 for d in ('','experiments/gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies','experiments/gvcrt_parallel_v5a3_v60','experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100'):
  sys.path.insert(0,str(REPO/d))
 # Read-only non-Git dependencies retain the exact parent source path and hash.
 sys.path.append(str(ORIGINAL/'expericent_generation_input/expericent_interface_causal_controls_v9/src'))
 import torch
 def no_download(*a,**kw):raise RuntimeError('LOCAL_MODEL_CACHE_REQUIRED; downloading weights is prohibited')
 torch.hub.download_url_to_file=no_download
 from src.utils.common import set_torch_env
 set_torch_env();torch.cuda.set_device(0);torch.set_num_threads(2)
 import engine
 quality=engine.core.quality_models(torch.device('cuda:0'))
 flow,perceptual=engine.metrics.flo.load_models(torch.device('cuda:0'))
 flo=engine.metrics.flo
 fixed_metric_files={Path(flo.__file__).name:Path(flo.__file__),'pwcnet.py':flo.OFFICIAL_ROOT/'pwcnet.py',flo.PWC_WEIGHTS.name:flo.PWC_WEIGHTS}
 for record in load(ROOT/'audits/GTFlow_sources.json')['source_files']:
  path=fixed_metric_files[record['filename']]
  assert path.stat().st_size==record['size_bytes'] and sha(path)==record['sha256'],('FIXED_PWC_OR_SAMPLER_SOURCE_CHANGED',record['filename'])
 dump(ROOT/f'audits/GTFlow_models_rank{rank}.json',dict(status='PASS',PWC_module_hash=engine.core.module_hash(flow),FloLPIPS_module_hash=engine.core.module_hash(perceptual),PWC_weight_filename=flo.PWC_WEIGHTS.name,PWC_weight_sha256=sha(flo.PWC_WEIGHTS),PWC_source_sha256=sha(flo.OFFICIAL_ROOT/'pwcnet.py'),compat_source_sha256=sha(Path(flo.__file__)),official_commit=flo.OFFICIAL_COMMIT))
 return torch,engine,quality,flow,perceptual
def numerical_checks(torch):
 import torch.nn.functional as F
 backwarp=sys.modules['pwcnet'].backwarp
 h,w=9,11;x=torch.arange(w,device='cuda').view(1,1,1,w).float().expand(1,1,h,w).clone()
 y=torch.arange(h,device='cuda').view(1,1,h,1).float().expand_as(x);image=torch.cat((x,y,x+10*y),1)
 out=[]
 for dx in (0.,1.,-1.,20.):
  f=torch.zeros(1,2,h,w,device='cuda');f[:,0]=dx
  actual=backwarp(image,f)
  xx=torch.linspace(-1+1/w,1-1/w,w,device='cuda').view(1,1,1,w).expand(1,1,h,w)
  yy=torch.linspace(-1+1/h,1-1/h,h,device='cuda').view(1,1,h,1).expand(1,1,h,w)
  grid=torch.cat((xx+dx/((w-1)/2),yy),1).permute(0,2,3,1)
  sampled=F.grid_sample(torch.cat((image,torch.ones_like(x)),1),grid,mode='bilinear',padding_mode='zeros',align_corners=False)
  mask=(sampled[:,-1:]>.999).float();expected=sampled[:,:-1]*mask
  assert torch.equal(actual,expected)
  if dx==0.:assert float((actual-image).abs().max())<1e-4
  out.append(dict(dx=dx,center_input=float(x[0,0,4,5]),center_output=float(actual[0,0,4,5]),corner_output=float(actual[0,0,0,0]),valid_pixels=int(mask.sum()),maxabs_against_explicit_sampling=float((actual-expected).abs().max())))
 assert out[1]['center_output']>out[0]['center_output']>out[2]['center_output']
 assert out[3]['valid_pixels']==0
 return dict(status='PASS',tests=out,convention='Existing pixel-center linspace(-1+1/W,1-1/W) base grid, flow normalized by (W-1)/2,(H-1)/2; grid_sample align_corners=False; positive dx samples to the right; bilinear zero padding; sampled appended-one mask >0.999. Zero flow preserves the image within floating interpolation precision. Pixel flow normalization uses W-1/H-1 as in the existing implementation; preserved explicitly.',denominator='All H*W*3 image elements; invalid positions have zero warped reconstruction, not removed from the L1 denominator',S2VC_exact_implementation_reproduction_claim=False)
def build_point(base,old_point=None):
 source=inspect.getsource(base.run_point)
 source=source.replace('def run_point(engine,runtime,quality,flow,perceptual,frames,v,qp,step,cp,key):','def run_point(engine,runtime,quality,flow,perceptual,frames,v,qp,step,cp,key,gt_cache,old_point=None,sample_dir=None):')
 source=source.replace('encoded_hashes=[];actual=[];buffer=io.BytesIO()',"encoded_hashes=[];encoder_trace=[];actual=[];buffer=io.BytesIO()")
 source=source.replace("            encoded_hashes.append(eco.tensor_sha(decoded['x_hat']))","            encoded_hashes.append(eco.tensor_sha(decoded['x_hat']))\n            encoder_trace.append(dict(frame=i,actual_qp=q,payload_sha256=eco.sha256_bytes(encoded['bit_stream']),payload_bytes=len(encoded['bit_stream']),reference_frame_sha256=None if pe.dpb[0].frame is None else eco.tensor_sha(pe.dpb[0].frame),reference_feature_sha256=None if pe.dpb[0].feature is None else eco.tensor_sha(pe.dpb[0].feature)))")
 start=source.index('        ie,pe=runtime.models');end=source.index('        ia,pa=runtime.models')
 encoded=source[start:end]
 reuse="""        stream_path=P/'large_artifacts/evaluation/bitstreams'/old_point['result']['bitstream_file']
        assert sha(stream_path)==old_point['result']['bitstream_sha256']
        data=stream_path.read_bytes();actual=old_point['actual_qps'];encoded_hashes=None
"""
 source=source[:start]+('        if old_point is None:\n'+''.join('    '+line+'\n' for line in encoded.splitlines())+'        else:\n'+''.join('    '+line+'\n' for line in reuse.splitlines()))+source[end:]
 source=source.replace("hashes.append(eco.tensor_sha(decoded['x_hat']));assert hashes[-1]==encoded_hashes[i]","hashes.append(eco.tensor_sha(decoded['x_hat']))\n            if encoded_hashes is not None:assert hashes[-1]==encoded_hashes[i]")
 source=source.replace("            spatial=eco.frame_metrics(out,gt,quality)","            if sample_dir is not None:\n                save_lossless_tensor(out,sample_dir/i.__str__())\n            spatial=eco.frame_metrics(out,gt,quality)")
 source=source.replace("            if previous_gt is not None:","            rows[-1].update(source_frame_RGB_sha256=v['frame_rgb_sha256'][i],checkpoint_sha256=cp['sha256'],reconstruction_tensor_sha256=hashes[-1],decoded_RGB_tensor_sha256=eco.tensor_sha(out))\n            if previous_gt is not None:")
 source=source.replace("trans.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=value))","""recon_warp=sys.modules['pwcnet'].backwarp(previous_out,flow(out,previous_out))
                warp_rec=float((gt-recon_warp).abs().mean())
                if i not in gt_cache:
                    target=flow(gt,previous_gt).detach().cpu().contiguous()
                    gt_cache[i]=dict(flow=target,sha256=hashlib.sha256(target.numpy().tobytes()).hexdigest(),source_RGB_sha256=v['rgb_sha256'])
                cached=gt_cache[i];assert cached['source_RGB_sha256']==v['rgb_sha256']
                assert hashlib.sha256(cached['flow'].numpy().tobytes()).hexdigest()==cached['sha256']
                gt_warp=sys.modules['pwcnet'].backwarp(previous_out,cached['flow'].cuda())
                warp_gt=float((gt-gt_warp).abs().mean())
                assert math.isfinite(warp_gt) and math.isfinite(warp_rec)
                rows[-1].update(WarpingError=warp_rec,WarpingError_ReconFlow_PWC=warp_rec,WarpingError_GTFlow_PWC=warp_gt,reconstruction_tensor_sha256=hashes[-1])
                trans.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=value,WarpingError=warp_rec,WarpingError_ReconFlow_PWC=warp_rec,WarpingError_GTFlow_PWC=warp_gt,GT_flow_sha256=cached['sha256']))""")
 source=source.replace("FloLPIPS=sum(r['FloLPIPS'] for r in trans)/63,","FloLPIPS=sum(r['FloLPIPS'] for r in trans)/63,**{k:sum(r[k] for r in trans)/63 for k in ('WarpingError','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC')},")
 source=source.replace('fresh_encode=True,','fresh_encode=old_point is None,input_stream_reused=old_point is not None,')
 source=source.replace("physical_GPU=5","physical_GPU=int(os.environ['CUDA_VISIBLE_DEVICES'])")
 source=source.replace("model_hashes_unchanged=True,TCG_reset_audits=resets,","model_hashes_unchanged=True,receiver_module_hashes=runtime.loaded_receiver_hashes,source_sha256=v['source_sha256'],native_geometry=[w,h],metrics_input='unquantized independently decoded native RGB tensors',TCG_reset_audits=resets,")
 source=source.replace("        write_csv(ROOT/'frames'/f'{key}.csv',rows)","        if old_point is None:write_csv(ROOT/'encoder_reference'/f'{key}.csv',encoder_trace)\n        write_csv(ROOT/'frames'/f'{key}.csv',rows)")
 validation="        if old_point is not None:\n            assert eco.sha256_bytes(''.join(hashes).encode())==old_point['reconstruction_sha256']\n            for metric in ('bpp','kbps','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError'):\n                assert abs(result[metric]-old_point['result'][metric])<1e-6,(metric,result[metric],old_point['result'][metric])\n"
 source=source.replace("        dump(ROOT/'points'",validation+"        dump(ROOT/'points'")
 ns=dict(base.__dict__,P=P,METRICS=METRICS,write_csv=write,save_lossless_tensor=save_lossless_tensor)
 exec(compile(source,str(ROOT/'gtflow.py'),'exec'),ns);return ns['run_point']
def save_lossless_tensor(out,path):
 # Explicitly lossy viewing previews; all neural metrics use unquantized tensors.
 import numpy as np
 from PIL import Image
 path=Path(path).with_suffix('.jpg');path.parent.mkdir(parents=True,exist_ok=True)
 a=(out[0].detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
 Image.fromarray(a).save(path,quality=95,subsampling=0)
 (path.parent/'PREVIEW_ONLY.txt').write_text('Lossy viewing preview: RGB rounded to 8 bit, JPEG quality 95, no chroma subsampling. All reported metrics use original independently decoded FP32 tensors, not this preview.\n')
 if sum(p.stat().st_size for p in (ROOT/'large_artifacts').rglob('*') if p.is_file())>20*2**30:raise RuntimeError('LARGE_ARTIFACT_BUDGET_20GiB_EXCEEDED')
 disk()
def worker(rank):
 base=load_module('tcg_legacy_evaluation',REPO/'experiments/gvcrt_tcg_stage1_evaluation/evaluate.py')
 torch,engine,quality,flow,perceptual=setup(rank)
 dump(ROOT/f'audits/GTFlow_sampler_rank{rank}.json',numerical_checks(torch))
 base.ROOT=ROOT/'evaluation/parent_gtflow';base.LARGE=ROOT/'large_artifacts/parent_gtflow'
 run_point=build_point(base)
 from adapter35 import frames_for
 import frames35
 videos=[v for vs in load(P/'evaluation/manifest.local.json').values() for v in vs]
 frames35._SOURCES['ulong']={v['source_row']['video']:dict(path=v['source_row']['source_path'],sha256=v['source_row']['source_sha256'],width=v['width'],height=v['height']) for v in videos if v['dataset']=='ulong'}
 originals=torch.load(ORIGINAL/'experiments/gvcrt_tcg_stage1_pilot/checkpoints/adaptation_step_0.pt',map_location='cpu',weights_only=True)['models']
 results=[]
 for index,v in enumerate(videos):
  if index%4!=rank:continue
  frames=frames_for(v);cache={}
  for step in (0,1000,2500,5000):
   path=P/f'checkpoints/step_{step}.pt';st=torch.load(path,map_location='cpu',weights_only=True)
   cp=next(c for c in load(P/'checkpoints_manifest.json')['checkpoints'] if c['filename']==path.name);assert sha(path)==cp['sha256']
   payload={k:dict(s) for k,s in originals.items()};payload['generator'].update(st['tcg_model_state'])
   class Runtime(engine.Runtime):
    def __init__(self):self.device=torch.device('cuda:0');self.wrapper=None;self.config=dict(force_zero_thres=.12);self.payload=payload
   for qp in (0,5,9):
    key=f"step{step}_{v['dataset']}_{v['name']}_qp{qp}";old=load(P/f'evaluation/points/{key}.json');assert old['result']['checkpoint_sha256']==cp['sha256']
    dest=ROOT/f'evaluation/parent_gtflow/points/{key}.json'
    if dest.exists():
     point=load(dest);assert point['status']=='PASS' and point['input_stream_reused'] and not point['fresh_encode'];results.append(point['result']);continue
    event('GT_FLOW_POINT_STARTED',point=key,physical_GPU=4+rank)
    result=run_point(engine,Runtime(),quality,flow,perceptual,frames,v,qp,step,cp,key,cache,old_point=old)
    results.append(result);write(ROOT/f'evaluation/parent_gtflow/worker_{rank}_metrics.csv',results)
    event('GT_FLOW_POINT_PASS',point=key,completed=len(results))
  if hasattr(frames,'close'):frames.close()
  del frames,cache
 dump(ROOT/f'evaluation/parent_gtflow/worker_{rank}_status.json',dict(status='PASS',completed_points=len(results)))
def main():
 with lock('gtflow.lock'):
  dump(ROOT/'evaluation/parent_gtflow/run_status.json',dict(status='RUNNING',completed_points=len(list((ROOT/'evaluation/parent_gtflow/points').glob('*.json'))),expected_points=156,input_stream_reused=True,fresh_encode=False,independent_decode=True,PID=os.getpid()))
  for n in ('GVC-RT_I.pt','GVC-RT_P.pt'):
   f=REPO/'checkpoints'/n
   if not f.exists():f.symlink_to(ORIGINAL/'checkpoints'/n)
   assert sha(f)==next(x['sha256'] for x in load(P/'source_manifest.json')['model_file_sources'] if x['filename']==n)
  # Read-only dependency link required by historical metric_runtime import.
  dep=REPO/'expericent_generation_input';target=ORIGINAL/'expericent_generation_input'
  if not dep.exists():dep.symlink_to(target,target_is_directory=True)
  ps=[]
  for rank in range(4):
   f=(ROOT/f'logs/parent_gtflow_rank{rank}.log').open('a')
   ps.append((subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'gtflow.py'),'--worker',str(rank)],cwd=REPO,stdout=f,stderr=subprocess.STDOUT),f))
  codes=[p.wait() for p,f in ps]
  for p,f in ps:f.close()
  if any(codes):
   dump(ROOT/'evaluation/parent_gtflow/run_status.json',dict(status='FAILED',exit_codes=codes));raise RuntimeError('GT_FLOW_WORKER_FAILURE')
  rows=[load(f)['result'] for f in sorted((ROOT/'evaluation/parent_gtflow/points').glob('*.json'))]
  assert len(rows)==156 and len({(r['step'],r['domain'],r['sequence'],r['QP']) for r in rows})==156
  write(ROOT/'evaluation/parent_gtflow/metrics.csv',rows)
  dump(ROOT/'evaluation/parent_gtflow/run_status.json',dict(status='PASS',completed_points=156,expected_points=156,input_stream_reused=True,fresh_encode=False,independent_decode=True))
if __name__=='__main__':
 try:
  if '--worker' in sys.argv:worker(int(sys.argv[-1]))
  else:main()
 except Exception:traceback.print_exc();sys.exit(1)
