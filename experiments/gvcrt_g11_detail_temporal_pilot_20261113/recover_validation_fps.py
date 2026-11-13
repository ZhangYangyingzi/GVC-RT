"""Resolve only six predeclared validation sources; never scan dataset media or final_test."""
from util import *
import zipfile
rows=videos('validation');out=[];missing=[]
for r in rows:
 p=Path(r['source_path'])
 if p.exists():assert sha(p)==r['source_sha256'];out.append((r,p))
 else:missing.append(r)
for archive in sorted(Path('/Huang_group/zyyz/datasets/UltraVideo-Long/clips_long_1920').glob('*.zip')):
 if not missing:break
 with zipfile.ZipFile(archive) as z:
  names={Path(n).name:n for n in z.namelist()}
  for r in list(missing):
   name=r['sequence']+'.mp4'
   if name not in names:continue
   raw=z.read(names[name]);assert hashlib.sha256(raw).hexdigest()==r['source_sha256'];p=ROOT/'cache/validation_source_identity'/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(raw);out.append((r,p));missing.remove(r)
assert not missing,[r['sequence'] for r in missing]
result=[]
for r,p in out:
 rate=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate','-of','json',str(p)],capture_output=True,text=True,check=True).stdout)['streams'][0]['r_frame_rate'];result.append(dict(sequence=r['sequence'],old_path=r['source_path'],verified_path=str(p),sha256=sha(p),fps=rate))
dump(ROOT/'sources/validation_source_recovery.json',dict(status='PASS',sources=result,identity_unchanged=True))
print('VALIDATION_FPS_VERIFIED',len(result))
