"""Initialize only experiment-local v3 outputs, then exec the active controller."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
for relative in ('logs','evaluation_v3/batches','large_artifacts/evaluation_v3','large_artifacts/viewing_previews_v3','migrations/online_earlystop_v3'):
 path=ROOT/relative;assert path.resolve().is_relative_to(ROOT);path.mkdir(parents=True,exist_ok=True)
if '--resume' in sys.argv:
 import json,time
 path=ROOT/'control_v3.json';old=json.loads(path.read_text());old.update(pause_requested=False,explicit_resume_unix=time.time());temp=path.with_suffix('.tmp');temp.write_text(json.dumps(old,indent=2)+'\n');temp.replace(path)
os.execv(sys.executable,[sys.executable,'-B','-u',str(ROOT/'v3_controller.py'),*[a for a in sys.argv[1:] if a!='--resume']])
