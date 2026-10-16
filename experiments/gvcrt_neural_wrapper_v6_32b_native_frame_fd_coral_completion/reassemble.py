"""Reconstruct text shards and verify the original SHA256 (no third-party imports)."""
import argparse,hashlib,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--index',default=str(Path(__file__).resolve().parent/'file_index.json'));ap.add_argument('--output',required=True);a=ap.parse_args();idx=Path(a.index);root=idx.parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
index=json.loads(idx.read_text());records=index.get('sharded_texts',[])
for part in index.get('index_parts',[]):
 b=(root/part['relative_path']).read_bytes();assert len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256']
 for line in b.splitlines():
  e=json.loads(line)
  if e['kind']=='sharded_text':records.append(e['record'])
for r in records:
 p=out/r['original_relative_path'];p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('wb') as f:
  for part in r['parts']:
   b=(root/part['relative_path']).read_bytes();assert len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256'];f.write(b)
 assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 print('VERIFIED',r['original_relative_path'])
