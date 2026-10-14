"""Only this experiment; independent latest-main worktree and ordinary fork push."""
from io32 import *
import subprocess,traceback
def main():
 ns=dict(legacy_io().__dict__);ns.update(ROOT=ROOT,dump=dump,load=load,sha=sha)
 src=(V30/'publish_github.py').read_text().replace('from io21 import *','').replace('Add v6.30 FID cache closure and sample robustness audit results','Add v6.32 learning-rate FD and CORAL control execution results').replace('v630_','v632_').replace('V630_PUBLICATION_PHASE','V632_PUBLICATION_PHASE')
 exec(compile(src,str(ROOT/'publish32.py'),'exec'),ns)
 ns['EXCLUDED'].update({'transactions','temporary','vendor'})
 dump(ROOT/'pipeline_status.json',dict(status='BLOCKED',calculation='A/B and eligible B1 work completed; C/D resolution blockers recorded',report='COMPLETE'))
 os.environ['V632_PUBLICATION_PHASE']='blocked_execution_record'
 ns['main']()
if __name__=='__main__':
 try:main()
 except Exception:
  previous=load(ROOT/'publication_status.json') if (ROOT/'publication_status.json').exists() else {};previous.update(status='FAIL',error=traceback.format_exc());dump(ROOT/'publication_status.json',previous);raise
