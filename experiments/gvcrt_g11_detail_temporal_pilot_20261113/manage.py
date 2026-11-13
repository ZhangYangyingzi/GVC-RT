from util import *
import argparse
from runner import alive
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['status','resume','verify']);a=p.parse_args()
 if a.action=='status':print(json.dumps(load(ROOT/'status.json'),indent=2))
 elif a.action=='verify':
  check=load(ROOT/'final_integrity.json');assert check['status']=='PASS';assert read(ROOT/'checkpoints/latest.pt')['protocol_sha256']==protocol_sha();print(json.dumps(check,indent=2))
 elif (ROOT/'logs/publication_remote_final.json').exists():print(json.dumps(dict(status='COMPLETE',resume='NO_OP_ALREADY_COMPLETE')))
 else:
  s=ROOT/'receipts/runner_process.json'
  if s.exists() and alive(load(s)['pid']):print(json.dumps(dict(status='RUNNING',pid=load(s)['pid'],attached=True)))
  else:subprocess.run([PYTHON,'-B','runner.py'],cwd=ROOT,check=True)
