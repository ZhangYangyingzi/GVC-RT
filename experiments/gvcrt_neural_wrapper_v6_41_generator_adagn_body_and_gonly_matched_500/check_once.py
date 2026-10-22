"""Lightweight v6.41 progress snapshot; never imports Torch or scans caches."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def get(path):
 p=ROOT/path
 try:return json.loads(p.read_text())
 except FileNotFoundError:return {}
def main():
 a=get('run_status.json');b=get('branches/G_fullLoss_Gonly_500/training_status.json')
 c=get('run_gonly_status.json');d=get('final_integrity.json');pub=get('publication_status.json')
 print(json.dumps(dict(A_RD=f"{a.get('complete',0)}/39 tasks",A_status=a.get('status','NOT_STARTED'),
  G_only_steps=f"{b.get('committed_updates',0)}/500",B_status=b.get('status','NOT_STARTED'),
  G_only_eval=f"{c.get('complete',0)}/39 tasks",G_only_eval_status=c.get('status','NOT_STARTED'),
  computation=d.get('status','PENDING'),publication=pub.get('status','PENDING'),
  A_failures=len(a.get('failures',{})),B_eval_failures=len(c.get('failures',{}))),ensure_ascii=False))
if __name__=='__main__':main()
