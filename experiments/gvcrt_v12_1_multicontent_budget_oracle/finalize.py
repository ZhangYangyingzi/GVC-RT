#!/usr/bin/env python3
import csv,glob,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(pattern):
    out=[]
    for p in sorted(glob.glob(str(ROOT/'parts'/pattern))):
        with open(p,newline='') as f:out.extend(csv.DictReader(f))
    return out
def write(name,rows):
    rows=list(rows);p=ROOT/'raw'/name;p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

baseline=read('*_baseline.csv');candidates=read('*_candidates.csv');selected=read('*_selected.csv');frames=read('*_frames.csv');audit=read('*_audit.csv');sync=read('*_sync.csv');rd=read('*_rd.csv')
write('baseline_cells.csv',baseline);write('all_candidates.csv',candidates);write('selected_budget_feasible.csv',selected);write('frame_metrics.csv',frames);write('bitstream_audit.csv',audit);write('state_sync.csv',sync);write('rd_points.csv',rd)
done=[];failed=[]
for p in sorted((ROOT/'parts').glob('*_done.json')):
    x=json.loads(p.read_text());done.append(x)
for p in sorted((ROOT/'parts').glob('*_failed.json')):
    failed.append({'file':str(p),'payload':json.loads(p.read_text())})
expected=12
checks={'expected_cells':expected,'completed_cells':sum(x.get('status')=='PASS' for x in done),'all_state_sync':all(r.get('status')=='PASS' for r in sync),'all_candidate_decode':all(r.get('decode_status')=='PASS' for r in audit),'all_finite':all(r.get('finite_status','PASS')=='PASS' for r in candidates),'all_model_unchanged':all(x.get('model_unchanged') for x in done),'failed_records':failed}
checks['status']='PASS' if checks['completed_cells']==expected and all(checks[k] for k in ('all_state_sync','all_candidate_decode','all_finite','all_model_unchanged')) else 'FAIL'
(ROOT/'sanity'/'final_integrity.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
