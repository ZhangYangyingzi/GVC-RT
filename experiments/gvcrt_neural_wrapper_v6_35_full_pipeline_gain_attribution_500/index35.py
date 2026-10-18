import hashlib,json,platform
from pathlib import Path
R=Path(__file__).resolve().parent
try:
 import matplotlib,numpy
 plotting={'python':platform.python_version(),'executable':'/usr/bin/python3','matplotlib':matplotlib.__version__,'numpy':numpy.__version__}
except Exception as e:plotting={'python':platform.python_version(),'error':repr(e)}
(R/'plot_environment.json').write_text(json.dumps(plotting,indent=2)+'\n')
skip_parts={'bitstreams','features','cache','__pycache__'};files=[]
for p in sorted(R.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(R)
 if any(part in skip_parts for part in rel.parts) or (rel.parent.name=='tasks' and p.suffix=='.lock') or rel.as_posix()=='file_index.json':continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 files.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(R/'file_index.json').write_text(json.dumps({'root':str(R),'self_indexed':False,'excluded_server_binary_directories':sorted(skip_parts),'file_count':len(files),'files':files},indent=2)+'\n')
print('INDEXED',len(files),'files')
