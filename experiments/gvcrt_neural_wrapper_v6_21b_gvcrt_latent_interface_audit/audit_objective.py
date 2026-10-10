"""Search available source for exact original losses; no proxy metrics."""
import json,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
B=ROOT
REPO=ROOT.parents[1]
V20=ROOT.parent/'gvcrt_neural_wrapper_v6_20_native_i_reference_adaptation'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def bd(name,v):
 p=B/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def objective_search():
 hits=[];files=[]
 for p in sorted(REPO.rglob('*.py')):
  if any(x in p.parts for x in ('cache','__pycache__','.git')) or p.is_relative_to(ROOT) or p.is_relative_to(B):continue
  txt=p.read_text(errors='replace');matches=[]
  for i,line in enumerate(txt.splitlines(),1):
   if re.search(r'cosine_similarity|cosine_embedding|L_cos\b|L_margin\b|margin_loss|cos_loss',line,re.I):matches.append(dict(line=i,text=line))
  if p.is_relative_to(REPO/'src') or matches:files.append(dict(path=str(p),sha256=sha(p)))
  if matches:hits.append(dict(path=str(p),sha256=sha(p),matches=matches))
 # The public codec distribution supplies inference modules, not the original training objective.
 codec_hits=[h for h in hits if Path(h['path']).is_relative_to(REPO/'src')]
 assert not codec_hits,('original codec loss candidate needs inspection',codec_hits)
 bd('config.json',dict(no_training=True,seed=20261010,methods=list(load(V20/'config.json')['methods']),external_qps=[0,4,9],frames=64,expected_points=240))
 bd('protocol.json',dict(no_training=True,no_proxy_replacement=True,original_loss_required=True,checkpoint_source=str(V20),status='BLOCKED',missing_definitions=['original GVC-RT cosine alignment training loss','original GVC-RT margin training loss']))
 bd('audits/gvcrt_objective_source.json',dict(status='BLOCKED',search_root=str(REPO),searched_python_files=len(list(REPO.rglob('*.py'))),source_files=files,candidate_hits=hits,original_codec_hits=codec_hits,exact_formulas=None,exact_tensor_names=None,reason='Original GVC-RT training cosine/margin objective implementation is absent from the available codec source. Adaptation/proxy and teacher LFQ losses are not substituted.'))
 (B/'audits/gvcrt_objective_source_excerpt.txt').write_text('\n'.join(h['path']+'\n'+'\n'.join(str(m['line'])+': '+m['text'] for m in h['matches']) for h in hits)+'\n')
 for name in ['gvcrt_objective_integrity.json','latent_probe_integrity.json','codec_integrity.json','evaluation_integrity.json']:
  bd(name,dict(status='BLOCKED',no_training=True,reason='Exact original loss definition unavailable; no probe or evaluation using proxy losses.'))
 bd('blocking_error.json',dict(status='BLOCKED',reason='Original GVC-RT cosine alignment and margin training loss definitions unavailable in repository source',no_proxy_replacement=True,training_performed=False))
 bd('final_integrity.json',dict(status='BLOCKED',no_training=True,no_proxy_replacement=True,exact_gvcrt_alignment_loss_reused=False,completed_points=0,expected_points=240,blocking_error='blocking_error.json'))
 (B/'README_NUMBERS_ONLY.md').write_text('# Protocol\n\nNo training. Exact original GVC-RT loss required.\n\n| Item | Status |\n|---|---|\n| Original cosine/margin objective source | BLOCKED |\n| Completed points | 0 / 240 |\n\nFiles: audits/gvcrt_objective_source.json, audits/gvcrt_objective_source_excerpt.txt, blocking_error.json.\n')

if __name__=='__main__':objective_search()
