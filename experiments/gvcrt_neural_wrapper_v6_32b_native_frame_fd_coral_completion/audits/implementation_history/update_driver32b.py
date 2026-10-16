from pathlib import Path
p=Path('driver32b.py');s=p.read_text()
s=s.replace("procs={};logs={}","""# Reconcile start records that were fsynced before a controller stopped but before
 # its next state snapshot. Live workers keep their full pending reservations.
 known={w['PID'] for w in running};events=ROOT/'gpu_schedule.jsonl'
 if events.exists():
  for line in events.read_text().splitlines():
   ev=json.loads(line)
   if ev.get('event')=='START' and ev.get('PID') not in known and alive(ev['PID'],ev.get('start_ticks')):
    assert str(ROOT) in identity(ev['PID'])['command'];ev=dict(ev,adopted=True);running.append(ev);known.add(ev['PID'])
 procs={};logs={} """)
s=s.replace("free,host=resources();inflight={w['gpu']:sum(x['reserved_MiB'] for x in running if x['gpu']==w['gpu']) for w in running if w['gpu'] is not None}","""free,host=resources()
  # Count actual residency once, and retain the unallocated part of every live
  # reservation. Model-loading tasks therefore reserve their full predicted peak.
  resident={}
  try:
   lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()
   for line in lines:
    a=line.split(',')
    if a[0].strip().isdigit() and a[1].strip().isdigit():resident[int(a[0])]=float(a[1])
  except Exception:pass
  observed=[]
  for w in running:
   if w['category']=='train':
    tp=ROOT/'branches'/w['branch']/'training_status.json'
    if tp.exists():
     tr=load(tp)
     if tr.get('PID')==w['PID'] and tr.get('committed_updates',0)>=1:observed.append(tr['peak_memory_MiB'])
  if observed:measured['train']=max(observed)*1.1
  for w in running:
   if w['category']=='train' and observed:
    tp=ROOT/'branches'/w['branch']/'training_status.json'
    if tp.exists() and load(tp).get('PID')==w['PID']:
     reserve=measured['train']*1.2+2048
     if reserve!=w['reserved_MiB']:
      append(ROOT/'gpu_schedule.jsonl',dict(event='RESERVATION_MEASURED_UPDATE',time=time.time(),task=w['id'],PID=w['PID'],physical_GPU=w['gpu'],previous_reserved_MiB=w['reserved_MiB'],reserved_MiB=reserve,measured_full_update_peak_MiB=load(tp)['peak_memory_MiB']))
      w['reserved_MiB']=reserve
  inflight={g:sum(max(0.,w['reserved_MiB']-resident.get(w['PID'],0.)) for w in running if w['gpu']==g) for g in (4,5,6,7)}""")
s=s.replace("# Current free already deducts actual occupancy. Also deduct ALL live reservations;\n   # this is deliberately conservative and protects workers still loading models.","# Free memory deducts residency. Subtract only unrealized reservation plus margin;\n   # every loading worker with no allocation still reserves its entire peak.")
p.write_text(s)
import ast
ast.parse(s)
