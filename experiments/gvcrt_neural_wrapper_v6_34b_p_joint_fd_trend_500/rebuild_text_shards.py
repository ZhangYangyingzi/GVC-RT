import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
INDEX=ROOT/'text_shards'/'shard_index.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 idx=json.loads(INDEX.read_text())
 for item in idx['files']:
  out=ROOT/item['source_path'];out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+'.rebuild.tmp')
  with tmp.open('wb') as w:
   for s in item['shards']:w.write((ROOT/s['path']).read_bytes())
  if tmp.stat().st_size!=item['bytes'] or sha(tmp)!=item['sha256']:
   tmp.unlink(missing_ok=True);raise SystemExit('rebuild hash mismatch: '+item['source_path'])
  tmp.replace(out)
  print('PASS',item['source_path'],item['bytes'],item['sha256'])
if __name__=='__main__':main()
