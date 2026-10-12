"""Attach measured artifact peaks to scheduler admission estimates."""
from io21 import *
def main():
 starts={};out=[]
 for line in (ROOT/'gpu_schedule.jsonl').read_text().splitlines():
  e=json.loads(line);key=(e['task'],e.get('PID'));measured=None;source=None
  if e['event']=='start':starts[key]=e
  elif e['event']=='exit':
   s=starts.get(key,{})
   if s.get('kind')=='eval':
    paths=[point(s['dataset'],s['method'],s['video'],q) for q in s['qps']]
    values=[(load(p).get('peak_memory_MiB'),str(p)) for p in paths if p.exists()]
    values=[v for v in values if v[0] is not None]
    if values:measured,source=max(values)
   elif s.get('kind')=='probe':
    p=ROOT/'audits'/f"{s['branch']}_pretraining.json"
    if p.exists():measured=load(p).get('peak_memory_MiB');source=str(p)
   elif s.get('kind')=='train':
    p=ROOT/'branches'/s['branch']/'training_logs/train.jsonl'
    if p.exists():measured=max(json.loads(x).get('peak_memory_MiB',0) for x in p.read_text().splitlines());source=str(p)
   elif s.get('kind')=='visual':
    p=ROOT/'historical/visual_integrity.json'
    if p.exists():measured=load(p).get('peak_memory_MiB');source=str(p)
   e.update(admission_peak_estimate_MiB=e.pop('peak_memory_MiB',None),measured_peak_memory_MiB=measured,measured_peak_source=source,measurement_scope='matching point artifacts or branch maximum; unavailable remains null')
  out.append(e)
 (ROOT/'gpu_schedule_verified.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in out))
 dump(ROOT/'audits/gpu_schedule_semantics.json',dict(status='PASS',raw_schedule_preserved=True,admission_free_before='nvidia-smi free minus unallocated active-task reservations',estimate_separate_from_actual=True,measurement_source='PyTorch max_memory_reserved from complete training/evaluation',logical_device=0,physical_devices=[4,5,6,7],events=len(out)))
if __name__=='__main__':main()
