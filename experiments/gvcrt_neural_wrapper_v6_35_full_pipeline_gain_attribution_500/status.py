import json,time
from pathlib import Path
R=Path(__file__).resolve().parent
p=R/'pipeline_status.json'
print(p.read_text() if p.exists() else json.dumps({'status':'NOT_STARTED'}))
