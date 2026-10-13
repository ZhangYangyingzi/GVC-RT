"""Isolated latest-main ordinary fork publication, verified remote blobs."""
from audit31 import *
import traceback
def main():
 ns=dict(legacy_io().__dict__);ns.update(ROOT=ROOT,dump=dump,load=load,sha=sha)
 src=(V30/'publish_github.py').read_text().replace('from io21 import *','')
 src=src.replace('Add v6.30 FID cache closure and sample robustness audit results','Add v6.31 P color and effective loss update audit results').replace('v630_','v631_').replace('V630_PUBLICATION_PHASE','V631_PUBLICATION_PHASE')
 exec(compile(src,str(ROOT/'publish31.py'),'exec'),ns)
 ns['EXCLUDED'].update({'clones','cpu_tasks'})
 dump(ROOT/'pipeline_status.json',load(ROOT/'status.json'));os.environ['V631_PUBLICATION_PHASE']='final_integrity_verified'
 ns['main']()
if __name__=='__main__':
 try:main()
 except Exception:
  dump(ROOT/'publication_status.json',dict(status='FAIL',error=traceback.format_exc()));raise
