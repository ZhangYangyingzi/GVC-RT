"""Explicit CPU-only checks; never schedules GPU tasks."""
from preflight import *
def main():
 import torch,ast
 from src.entropy_coder import GaussianRansCodec
 from rate_backend import table_bits
 from bootstrap import BRANCHES,V1,SOURCE_SHA
 from runtime import Sender
 source=V1/'final_P/C_latent_P_1000.pt';assert sha(source)==SOURCE_SHA
 p=Sender('C_latent_P');p.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['P'],strict=True);assert state_hash(p.state_dict())==load_json(R/'audits/source_experiment.json')['P_module_sha256']
 coder=GaussianRansCodec();symbols=torch.tensor([0,1,-1,10,-10,255,256,-256,32000,-32000]*64,dtype=torch.float32);scales=torch.tensor([1e-5,.1,1.,3.,100.]*128);bits,stats=table_bits(symbols,scales,coder);assert torch.isfinite(bits).all() and stats['escape_symbols']>0
 coder.reset_encoder();coder.encode(symbols,scales);payload=coder.finish_encoder();coder.start_decoder(payload);decoded=coder.decode(scales,'cpu',torch.float32);assert torch.equal(symbols,decoded)
 assert abs(len(payload)*8-float(bits.sum()))<128,('CDF and escape accounting disagreement',len(payload)*8,float(bits.sum()))
 files=list(R.glob('*.py'))
 for f in files:ast.parse(f.read_text(),filename=str(f))
 save('audits/cpu_implementation_checks.json',dict(status='PASS',CPU_only=True,source_module_hash=state_hash(p.state_dict()),branch_coefficients=BRANCHES,synthetic_escape_and_scale_clamp_test=True,coder_table_bits=float(bits.sum()),actual_payload_bits=len(payload)*8,residual='finite rANS state/byte alignment',symbols_decoded_exactly=True,stats=stats,python_sources_parsed=len(files),formal_updates=0,formal_evaluation_points=0))
 print('CPU IMPLEMENTATION CHECKS PASS',flush=True)
def load_json(path):return json.loads(Path(path).read_text())
if __name__=='__main__':main()
