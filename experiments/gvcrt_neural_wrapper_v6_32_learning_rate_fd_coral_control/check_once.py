"""Read only the small runtime status; never import torch or scan artifacts."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent
if __name__=='__main__':
 def read(name):
  try:return json.loads((R/name).read_text())
  except (OSError,ValueError):return {}
 s=read('status.json');launch=read('launch.json');parallel=read('parallel_evaluation_status.json');pub=read('publication_status.json')
 print(json.dumps(dict(session=launch.get('session'),PID=s.get('driver_PID',launch.get('PID')),stage=s.get('status'),terminal=s.get('terminal'),training=s.get('training'),evaluation=s.get('evaluation'),failed=s.get('failed'),pending=s.get('pending'),heartbeat_unix=s.get('heartbeat_unix'),parallel_PID=parallel.get('PID'),parallel_terminal=parallel.get('terminal'),report=s.get('report'),publication=pub.get('status',s.get('publication'))),ensure_ascii=False))
