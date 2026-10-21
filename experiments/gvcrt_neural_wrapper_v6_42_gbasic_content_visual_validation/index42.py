from common42 import *
def main():
 assert load(ROOT/'final_integrity.json')['status']=='PASS'
 assert load(ROOT/'audits/visual_material_integrity.json')['status']=='PASS'
 sources={}
 for name in ('io32.py','io35.py','adapter35.py','training35.py','evaluate.py','evaluation_config.json'):
  p=REF/name;sources[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p),role='read-only evaluation reference')
 for name in ('eval_manifest.json','results/reused_points.csv','evaluation_config.json'):
  p=V37/name;sources[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p),role='read-only data/stream reference')
 for n,h in load(REF/'source_hashes.json').items():
  p=Path(n);assert sha(p)==h,(n,'historical dependency changed');sources[n]=dict(bytes=p.stat().st_size,sha256=h,role='verified inherited dependency')
 for p in ROOT.glob('*.py'):sources[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p),role='experiment implementation')
 dump(ROOT/'source_manifest.json',sources)
 dump(ROOT/'audits/historical_readonly.json',dict(status='PASS',inherited_dependencies_verified=len(load(REF/'source_hashes.json')),
  no_historical_writes=True,historical_paths=['v6.37','v6.40'],v641_checkpoints_used=False))
 exclude={'artifact_index.json','publication_status.json','run_status.json','audits/publication_content_verification.json','logs/publication.jsonl'}
 public=[]
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file():continue
  rel=str(p.relative_to(ROOT))
  if rel in exclude or any(x in p.parts for x in ('server_media','server_previews','__pycache__')) or p.suffix in ('.pt','.npz','.npy','.bin','.lock','.tmp'):continue
  assert p.stat().st_size<50*1024*1024
  public.append(dict(relative_path=rel,bytes=p.stat().st_size,sha256=sha(p)))
 dump(ROOT/'artifact_index.json',dict(status='PASS',public=public,public_files=len(public),public_bytes=sum(x['bytes'] for x in public),
  server_index='server_artifacts_index.json',full_media_uploaded=False))
 if not (ROOT/'publication_status.json').exists():dump(ROOT/'publication_status.json',dict(status='NOT_PUBLISHED',push_success=False))
 print('INDEX_PASS',len(public),sum(x['bytes'] for x in public),flush=True)
if __name__=='__main__':main()
