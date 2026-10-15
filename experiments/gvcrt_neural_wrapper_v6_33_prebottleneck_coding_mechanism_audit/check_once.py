"""Small read-only state; never import torch or inspect large artifacts."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent
for n in ('launch.json','status.json','publication_status.json'):
 if (r/n).exists():print(n,json.dumps(json.loads((r/n).read_text()),ensure_ascii=False))
