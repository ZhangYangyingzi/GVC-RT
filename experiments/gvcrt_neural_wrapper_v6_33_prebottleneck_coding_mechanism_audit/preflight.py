"""Read-only provenance/input checks; freeze exact dependencies and experiment signatures."""
import subprocess,ast,platform,time
from audit_io import *

def main():
 old=EXPS/'gvcrt_neural_wrapper_v6_31a_latent_fidelity_vs_generative_utility_audit'
 protocol=load(ROOT/'protocol.json');datasets=load(ROOT/'dataset_manifest.json')['datasets'];assert {d:len(v) for d,v in datasets.items()}==protocol['populations']
 assert sha(ROOT/'dataset_manifest.json')==sha(old/'dataset_manifest.json')
 sources=load(old/'source_hashes.json')
 # Only immutable code/model/data dependencies are inherited; runtime status is not a signature.
 bad=[p for p,h in sources.items() if sha(p)!=h];assert not bad,bad
 inventory={}
 for name,p in [('original_I',REPO/'checkpoints/GVC-RT_I.pt'),('original_P',REPO/'checkpoints/GVC-RT_P.pt'),('B1',B1)]:inventory[name]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
 assert inventory['B1']['sha256']=='dc76f0e221dc7aaa5e5f45f648722584ce226e65828ce24151f9961985f3f6f7'
 for p in [old/'audit_io.py',old/'proxy_runtime.py',old/'worker.py',ROOT/'dataset_manifest.json',REPO/'src/models/common_model.py',REPO/'src/models/video_model_gvcrt.py',REPO/'src/models/entropy_models.py',REPO/'src/layers/cuda_inference.py']:
  sources[str(p)]=sha(p)
 frames=0
 for d,vs in datasets.items():
  for v in vs:
   assert len(v['source_frame_indices'])==64
   for i in range(64):
    p=Path(v['input_dir'])/(f'im{i+1:05d}.png' if d=='hevc_b' else f'frame_{i:06d}.png')
    assert sha(p)==v['frame_png_sha256' if d=='hevc_b' else 'frame_file_sha256'][i];sources[str(p)]=sha(p);frames+=1
 dump('checkpoint_inventory.json',inventory);dump('source_hashes.json',sources)
 dump('tensor_semantics.json',dict(analysis_y='P.compress -> self.enc(x,ctx,q_encoder) -> compress_prior_2x first argument, before in-place reciprocal scaling',y_hat='compress_prior_2x returned last tensor: sums of centered integer symbols + per-group means, times clamped q_dec; input to actual latent decoder',symbols='process_with_mask returned second tensor, actual centered rounded/clipped/skip-forced y_q; nonoverlapping masks merged; exactly packed for RANS',boundaries='half-integers in y/q_dec - actual conditional mean for active non-saturated entries; force_zero_thres scale boundary separate; no fixed half-integer boundary in raw y',entropy='packed int16 at native entropy_coder.encode_y: signed high8 symbol,low8 actual CDF index; use installed quantized CDF. Tail/escape bypass bits unavailable, so CDF entry bits are partial diagnostic.',prediction='unavailable: feature_extractor produces conditional context, not a defined RGB or analysis-y prediction',codec_sources={str(p):sha(p) for p in [REPO/'src/models/common_model.py',REPO/'src/models/video_model_gvcrt.py',REPO/'src/models/entropy_models.py',REPO/'src/layers/cuda_inference.py']}))
 # CPU mathematical check of exact strength endpoints and real process_with_mask fallback.
 import torch
 from proxy_runtime import backend
 backend()
 from src.layers.cuda_inference import process_with_mask
 x=torch.linspace(0,1,101);p=(x+.1*torch.sin(x*8)).clamp(0,1)
 assert torch.equal((x+0*(p-x)).clamp(0,1),x)
 assert torch.allclose((x+(p-x)).clamp(0,1),p)
 y=torch.tensor([[[[.49,.51,2.,-2.,200.]]]]);m=torch.ones_like(y);sc=torch.tensor([[[[1.,1.,.1,1.,1.]]]]);mean=torch.zeros_like(y)
 res,sym,hat,_=process_with_mask(y,sc,mean,m,.12)
 assert sym.flatten().tolist()==[0.,1.,0.,-2.,127.]
 assert torch.equal(hat,sym)
 for f in ROOT.glob('*.py'):ast.parse(f.read_text())
 dump('selftest.json',dict(status='PASS',alpha_endpoints=True,quantizer_round_skip_clip=True,optimizer_steps=0))
 dump('preflight.json',dict(status='PASS',checked_frames=frames,videos=sum(map(len,datasets.values())),expected_points=3220,source_dependencies=len(sources),teacher='paper_compatible_teacher_proxy',HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),python=sys.executable,time=time.time(),checkpoint_updates=0))
 print('PREFLIGHT_PASS',frames,flush=True)
if __name__=='__main__':main()
