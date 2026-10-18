import json,hashlib,os
from pathlib import Path
R=Path(__file__).resolve().parent
EX={'branches','evaluation','bitstreams','features','raw','reference_stats','__pycache__'}
EXACT={'plans/shared.json','train_manifest_ulong.json','audits/workspace_state.json'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
files=[];large=[]
for p in sorted(R.rglob('*')):
 rel=p.relative_to(R)
 if p.is_symlink() or not p.is_file():continue
 if str(rel) in EXACT:continue
 if rel.name in ('file_index.json','server_artifacts_index.json'):continue
 if any(x in EX for x in rel.parts) and (rel.parts[0] in EX):continue
 if rel.parts[0]=='branches' and 'checkpoints' in rel.parts:continue
 if rel.parts[:2]==('evaluation','checkpoints'):continue
 if p.suffix in ('.pyc','.tmp') or p.name.endswith('.lock'):continue
 size=p.stat().st_size;rec={'path':str(rel),'bytes':size,'sha256':sha(p)};files.append(rec)
 if size>512*1024:large.append(rec)
(R/'file_index.json').write_text(json.dumps({'indexed_files':len(files),'excluded_large_server_artifacts':['branches/*/checkpoints/**','evaluation/checkpoints/**','bitstreams/**','features/**','raw/**','reference_stats/**'],'files':files,'files_over_512KiB':large,'index_excludes_itself':True},indent=2)+'\n')
# Server-side binary index uses the checkpoint/evaluation receipts' verified hashes.
items={}
def add(path,digest=None,bytes_=None,kind=None):
 p=Path(path)
 if not p.is_absolute():p=(R/p).resolve()
 if not p.exists():return
 items[str(p)]={'path':str(p),'bytes':int(bytes_ if bytes_ is not None else p.stat().st_size),'sha256':digest or sha(p),'kind':kind or 'server_binary'}
for b in ('N_noP_BG_FD','P_PBG_FD'):
 ix=json.load(open(R/'branches'/b/'checkpoint_index.json'))
 for step,e in ix.items():
  if 'path' in e:add(e['path'],e.get('sha256'),kind='training_checkpoint_step_'+step)
  if 'inference' in e:add(e['inference']['path'],e['inference']['sha256'],kind='inference_checkpoint_step_'+step)
 add(R/'branches'/b/'checkpoints/latest.pt',kind='training_checkpoint_latest')
for p in sorted((R/'tasks').glob('*.json')):
 try:r=json.load(open(p))
 except Exception:continue
 if r.get('status')!='PASS':continue
 for art,meta in r.get('artifact_metadata',{}).items():
  q=Path(art)
  if q.exists() and q.suffix in ('.bin','.npz'):add(q,meta.get('sha256'),meta.get('bytes'),kind='bitstream' if q.suffix=='.bin' else 'feature_cache')
(R/'server_artifacts_index.json').write_text(json.dumps({'server_artifacts':len(items),'artifacts':list(items.values())},indent=2)+'\n')
print('INDEXED',len(files),'files',len(items),'server artifacts','large_text',len(large))
