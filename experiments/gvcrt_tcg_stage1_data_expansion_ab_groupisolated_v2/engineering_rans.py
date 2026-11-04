"""Pre-training engineering replay of one already-scored historical parent point.

This is not development/test scoring or an optimizer update. It checks the new
receipt instrumentation against the fixed E1 result before the long training.
"""
from ab_common import *
from gtflow import setup,load_module,build_point,numerical_checks,METRICS
import csv
def main():
 base=load_module('engineering_real_rans',REPO/'experiments/gvcrt_tcg_stage1_evaluation/evaluate.py')
 torch,engine,quality,flow,perceptual=setup(1)  # fixed physical GPU5, no fallback
 sampler=numerical_checks(torch)
 base.ROOT=ROOT/'audits/rans_roundtrip_probe';base.LARGE=ROOT/'large_artifacts/engineering_rans_probe'
 videos=[v for vs in load(P/'evaluation/manifest.local.json').values() for v in vs]
 video=next(v for v in videos if v['name']=='Kimono1');step=5000;q=5
 oldkey=f"step{step}_{video['dataset']}_{video['name']}_qp{q}"
 old=load(ROOT/f'evaluation/parent_gtflow/points/{oldkey}.json');assert old['status']=='PASS'
 cp=dict(filename='step_5000.pt',sha256=sha(P/'checkpoints/step_5000.pt'));assert cp['sha256']==PARENT_SHA==old['result']['checkpoint_sha256']
 payload=torch.load(ORIGINAL/'experiments/gvcrt_tcg_stage1_pilot/checkpoints/adaptation_step_0.pt',map_location='cpu',weights_only=True)['models']
 state=torch.load(P/'checkpoints/step_5000.pt',map_location='cpu',weights_only=True);payload['generator'].update(state['tcg_model_state'])
 class Runtime(engine.Runtime):
  def __init__(self):self.device=torch.device('cuda:0');self.wrapper=None;self.config=dict(force_zero_thres=.12);self.payload=payload
 from adapter35 import frames_for
 frames=frames_for(video);key='engineering_'+oldkey
 try:result=build_point(base)(engine,Runtime(),quality,flow,perceptual,frames,video,q,step,cp,key,{})
 finally:
  if hasattr(frames,'close'):frames.close()
 receipt=load(base.ROOT/f'points/{key}.json')
 assert receipt['real_RANS'] and receipt['fresh_encode'] and not receipt['input_stream_reused'] and receipt['independent_decode']
 assert receipt['reconstruction_sha256']==old['reconstruction_sha256'] and result['bitstream_sha256']==old['result']['bitstream_sha256'] and result['real_bytes']==old['result']['real_bytes']
 for metric in METRICS:assert abs(result[metric]-old['result'][metric])<1e-6,('ENGINEERING_REPLAY_METRIC_MISMATCH',metric)
 def rows(path):
  with path.open() as f:return list(csv.DictReader(f))
 transitions=rows(base.ROOT/f'transitions/{key}.csv');old_transitions=rows(ROOT/f'evaluation/parent_gtflow/transitions/{oldkey}.csv')
 assert len(transitions)==63 and [r['GT_flow_sha256'] for r in transitions]==[r['GT_flow_sha256'] for r in old_transitions]
 trace=rows(base.ROOT/f'encoder_reference/{key}.csv');assert len(trace)==64
 assert trace[0]['reference_frame_sha256'] and not trace[0]['reference_feature_sha256']
 assert all(not r['reference_frame_sha256'] and r['reference_feature_sha256'] for r in trace[1:])
 fr=rows(base.ROOT/f'frames/{key}.csv');assert len(fr)==64 and all(len(r['decoded_RGB_tensor_sha256'])==64 and r['checkpoint_sha256']==PARENT_SHA for r in fr)
 dump(ROOT/'audits/rans_engineering.json',dict(status='PASS',scope='One fixed already-scored historical parent point; engineering only, excluded from formal development/final point counts',physical_GPU=5,source_point=oldkey,source_receipt_SHA256=sha(ROOT/f'evaluation/parent_gtflow/points/{oldkey}.json'),parent_checkpoint_SHA256=PARENT_SHA,stream_SHA256=result['bitstream_sha256'],reconstruction_SHA256=receipt['reconstruction_sha256'],sampler_compatibility=sampler,independent_decode=True,fresh_encode=True,metrics_match_previous=True,GT_flow_hashes_match_previous=True,nullable_encoder_reference_trace_checked=True,no_optimizer_updates=True,no_new_dev_or_test_quality_read=True,implementation_SHA256=sha(ROOT/'engineering_rans.py')))
if __name__=='__main__':main()
