from io32 import *
REF=ROOT.parent/'gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100'
V37=ROOT.parent/'gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500'
CP=REF/'evaluation/checkpoints/G_basic_100.pt'
CP_SHA='89c1a5baa195c8ffc9695af495d43389e00021b926eae13a0337d3b2eb117fd4'
METHODS=('O_original','G_basic')
METRICS=('LPIPS','DISTS','FloLPIPS','PSNR','L1','actual_bpp')
def signature42():
 return hashlib.sha256(json.dumps({n:sha(ROOT/n) for n in ('protocol.json','visual_manifest.json','data_manifest.json','stream_manifest.json','model_manifest.json','evaluate42.py','common42.py','adapter35.py','io32.py','io35.py','training35.py')},sort_keys=True).encode()).hexdigest()
def frame_path(v,i):return Path(v['input_dir'])/(f'im{i+1:05d}.png' if v['dataset']=='hevc_b' else f'frame_{i:06d}.png')
