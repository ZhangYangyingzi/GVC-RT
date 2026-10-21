"""Index server-side binaries and public artifacts without self-referential hashes."""
from io32 import *
def main():
 binary=[];public=[]
 mutable={'artifact_index.json','publication_status.json','run_status.json'}
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.lock','.tmp'):continue
  if p.name in mutable or p.name.startswith('publication') or p.name in ('finish.log','index.log'):continue
  row=dict(path=str(p),relative_path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
  if p.suffix in ('.pt','.bin','.npz','.npy') or 'artifacts' in p.relative_to(ROOT).parts:binary.append(row)
  else:public.append(row)
 dump(ROOT/'artifact_index.json',dict(status='PASS',public=public,server_binary=binary,excluded_mutable_files=sorted(mutable),weights_and_raw_video_remain_server_side=True))
 print('ARTIFACT_INDEX_PASS',len(public),len(binary),flush=True)
if __name__=='__main__':main()
