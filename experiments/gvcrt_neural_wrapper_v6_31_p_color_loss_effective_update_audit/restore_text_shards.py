"""Restore this experiment's logical UTF8 files using verified publication indices."""
import argparse,hashlib,json
from pathlib import Path
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--destination',type=Path);a=ap.parse_args()
 root=a.root.resolve();dest=(a.destination or root).resolve();index=json.loads((root/'file_index.json').read_text());entries=index.get('files',[])
 def safe(base,rel):
  p=(base/rel).resolve();assert p.is_relative_to(base),('unsafe indexed path',rel);return p
 for p in index.get('index_parts',[]):
  f=safe(root,p['path']);assert digest(f)==p['sha256'];entries+=json.loads(f.read_text())['files']
 for item in entries:
  if not item.get('shards'):continue
  content=[]
  for s in item['shards']:
   p=safe(root,s['path']);assert p.stat().st_size==s['bytes'] and digest(p)==s['sha256'];content.append(p.read_bytes())
  data=b''.join(content);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
  out=safe(dest,item['path'])
  if out.exists():assert digest(out)==item['sha256'];continue
  out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+'.restore.tmp');tmp.write_bytes(data);tmp.replace(out)
 print('RESTORE PASS')
if __name__=='__main__':main()
