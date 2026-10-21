from io32 import *
REF=ROOT.parent/'gvcrt_neural_wrapper_v6_42_gbasic_content_visual_validation'
CP=ROOT.parent/'gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100/evaluation/checkpoints/G_basic_100.pt'
CP_SHA='89c1a5baa195c8ffc9695af495d43389e00021b926eae13a0337d3b2eb117fd4'
TARGETS=[8,16,24,32,40,48]
def signature43():
 names=['protocol.json','data_manifest.json','baseline_manifest.json','model_manifest.json','evaluation_config.json','evaluate43.py','common43.py','adapter35.py','io32.py','io35.py','training35.py','source_manifest.json']
 return hashlib.sha256(json.dumps({n:sha(ROOT/n) for n in names},sort_keys=True).encode()).hexdigest()
def key(t):return f"{t['dataset']}/{t['sequence']}/{t['variant']}"
def result_path(t):return ROOT/'tasks'/key(t)/'result.json'
def validate_result(t):
 p=result_path(t)
 if not p.exists():return False
 r=load(p);assert r['signature']==signature43() and r['task']==t
 for f in r['artifacts']:assert Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256'],f
 assert r['status']=='PASS'
 return True
