"""Reassemble only this published directory; verify and preserve differing files."""
from io32 import *
def main():
 index=load(ROOT/'file_index.json');entries=index.get('files',[])
 for item in index.get('index_parts',[]):
  p=ROOT/item['path'];assert sha(p)==item['sha256'];entries.extend(load(p)['files'])
 for item in entries:
  if 'shards' not in item:continue
  p=ROOT/item['path'];assert p.resolve().is_relative_to(ROOT)
  if p.exists():assert sha(p)==item['sha256'],('existing differing file preserved',str(p));continue
  chunks=[]
  for x in item['shards']:
   q=ROOT/x['path'];assert q.resolve().is_relative_to(ROOT) and sha(q)==x['sha256'];chunks.append(q.read_bytes())
  b=b''.join(chunks);assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 print('TEXT SHARDS VERIFIED',flush=True)
if __name__=='__main__':main()
