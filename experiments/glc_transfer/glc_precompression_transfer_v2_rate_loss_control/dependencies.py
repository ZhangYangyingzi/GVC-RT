"""Actual installed dependencies, weights and original source fingerprint check."""
from preflight import *
from bootstrap import V1
from data import load
import importlib.util,importlib.metadata
def main():
 from runtime import quality
 old=load(V1/'audits/reference_dependencies.json');metric=load(V1/'audits/metric_dependencies.json');rows=[]
 for x in old['files']:
  assert sha(x['source'])==x['sha256'],x['source'];rows.append(x)
  if x.get('snapshot'):
   target=R/x['snapshot'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(x['source']).read_bytes())
 assert sha('/Huang_group/zyyz/home_dir/.conda/envs/glc/weights.pt')==old['DISTS_weights_sha256']
 q=quality('cpu');models=[]
 for m,x in zip(q,metric['quality']):
  assert sha(x['path'])==x['source_sha256'] and state_hash(m.state_dict())==x['module_hash'],x['path'];models.append(dict(x,actual_module_hash=state_hash(m.state_dict())))
 for package,version in metric['versions'].items():assert importlib.metadata.version(package)==version,(package,version)
 assert sha('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')==metric['Inception_weights_sha256']
 path=Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback/flolpips_compat.py');spec=importlib.util.spec_from_file_location('v2_readonly_flow_fingerprints',path);flow=importlib.util.module_from_spec(spec);spec.loader.exec_module(flow)
 flowrows=[]
 for path,h in [(flow.OFFICIAL_ROOT/'flolpips.py',flow.OFFICIAL_SOURCE_SHA256),(flow.OFFICIAL_ROOT/'pwcnet.py',flow.OFFICIAL_PWC_SOURCE_SHA256),(flow.OFFICIAL_ROOT/'weights/v0.1/alex.pth',flow.OFFICIAL_ALEX_WEIGHT_SHA256),(flow.PWC_WEIGHTS,flow.PWC_WEIGHT_SHA256)]:assert sha(path)==h;flowrows.append(dict(path=str(path),sha256=h))
 for x in rows:
  if x['source'].endswith('MLCodec_rans.cpython-312-x86_64-linux-gnu.so'):assert Path(x['source']).stat().st_size==x['bytes']
 save('audits/reference_dependencies.json',old);save('audits/metric_dependencies.json',dict(status='PASS',quality=models,versions=metric['versions'],Inception_weights_sha256=metric['Inception_weights_sha256'],FloLPIPS=flowrows,existing_v1_source_enforced_same_weight_fingerprints=True))
 save('audits/server_preflight.json',dict(status='PASS',python=sys.executable,disk_free_bytes=__import__('shutil').disk_usage(R).free,GLC_commit=subprocess.check_output(['git','-C',str(GLC),'rev-parse','HEAD'],text=True).strip(),GLC_worktree=subprocess.check_output(['git','-C',str(GLC),'status','--porcelain=v1'],text=True),GVC_commit=subprocess.check_output(['git','-C',str(GVC),'rev-parse','HEAD'],text=True).strip(),AGENTS_search='no applicable AGENTS.md found in ancestor directories or GLC/GVC repository search',GPU_allowed=[4,5,6,7],frozen_source_fingerprints=rows))
 print('DEPENDENCIES PASS',flush=True)
if __name__=='__main__':main()
