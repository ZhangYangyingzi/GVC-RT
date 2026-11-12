"""No duplicate launch; resume a fixed-budget run or verify an already completed run."""
from util import *
import argparse

def verify():
 check_protocol();checks=['audits/preflight.json','audits/extended_checks.json','audits/aligner_frozen_reference.json','audits/calibration.json']
 for p in checks:assert (ROOT/p).exists(),('MISSING',p)
 align=readtensor(ROOT/'checkpoints/aligner_0500.pt');assert align['committed_updates']==500 and align['protocol_sha256']==check_protocol()
 groups={}
 for g in ['S_current','U_unaligned','T_aligned']:
  p=ROOT/'checkpoints'/g/'step_0300.pt';c=readtensor(p);assert c['committed_updates']==300 and c['pretrained_aligner_sha256']==sha(ROOT/'checkpoints/aligner_0500.pt') and c['protocol_sha256']==check_protocol();assert load(p.parent/'freeze_audit.json')['status']=='PASS';groups[g]=300
 if (ROOT/'receipts/evaluation.complete.json').exists():
  r=load(ROOT/'receipts/evaluation.complete.json');assert all((ROOT/'metrics'/f"{__import__('evaluate').tag(j)}.complete.json").exists() for j in r['jobs'])
 return dict(status='VERIFIED',aligner_updates=500,groups=groups,protocol_sha256=check_protocol())

a=argparse.ArgumentParser();a.add_argument('action',choices=['status','resume','verify']);p=a.parse_args()
if p.action=='status':print(json.dumps(load(ROOT/'status.json'),indent=2))
elif p.action=='verify':print(json.dumps(verify(),indent=2))
else:
 if (ROOT/'receipts/local_delivery.complete.json').exists():print(json.dumps(verify()|{'resume':'NO_OP_ALREADY_COMPLETE'},indent=2))
 else:
  # lock is tested in runner; active workers are attached by PID instead of repeated.
  subprocess.run([PYTHON,'-B','runner_next.py'],cwd=ROOT,check=True)
