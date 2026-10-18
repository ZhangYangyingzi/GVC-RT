"""Minute-spaced read-only monitor, no GPU or process control."""
import json,time
from pathlib import Path
R=Path(__file__).resolve().parent
def read(path):
 try:return json.loads((R/path).read_text())
 except (OSError,ValueError):return {}
def main():
 end=time.monotonic()+96*3600
 while time.monotonic()<end:
  s=read('status.json');pub=read('publication_status.json');worker=read('worker_status.json');report=read('audits/report_progress.json')
  out=dict(status=s.get('status'),phase=s.get('phase'),PID=s.get('PID'),samples=read('preparation_status.json'),paired_rate=len(list((R/'audits/rate_pairs').glob('*.json'))),paired_rate_expected=120,gradient_audits=len(list((R/'audits/component_gradients').glob('*.json'))),gradient_audits_expected=80,updates={b:read(f'branches/{b}/status.json').get('updates',0) for b in ('H00_continue','H10_rate2','H01_dists025','H11_rate2_dists025')},evaluation_completed=len(list((R/'results/points').glob('*.json'))),evaluation_expected=2760,new_evaluation_expected=1840,worker_count=len(worker.get('running',[])),FID_progress=report,publication_phase=pub.get('phase'),commit=pub.get('commit'),time=time.time())
  if s.get('error'):out['error']=s['error']
  print(json.dumps(out,ensure_ascii=False),flush=True)
  if s.get('terminal'):return
  time.sleep(60)
if __name__=='__main__':main()
