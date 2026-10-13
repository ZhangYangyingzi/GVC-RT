"""Preserve computations after supplying the frozen plotting QP tuple."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from audit_io import *
old=load(ROOT/'audits/report_repair/migration.json')['new_code_hashes']
new={p.name:sha(p) for p in ROOT.glob('*.py')}
source=(ROOT/'plot30.py').read_text()
assert hashlib.sha256(source.replace(';QPS=(0,5,9)','').encode()).hexdigest()==old['plot30.py']
assert all(new[n]==h for n,h in old.items() if n!='plot30.py') and set(new)==set(old)
rows=[]
for p in sorted((ROOT/'tasks').glob('*.json')):
    r=load(p);t=r['task'];assert r['signature']==signature(t,old)
    assert all(Path(a).is_file() and sha(a)==h for a,h in r['artifacts'].items())
    previous=r['signature'];digest=sha(p);r['signature']=signature(t)
    r['plot_inputs_signature_migration']=dict(old_receipt_sha256=digest,original_signature=previous,audit='audits/report_repair/plot_inputs_migration.json',numerical_code_unchanged=True)
    dump(p,r);assert valid(t);rows.append(dict(task=task_id(t),old_signature=previous,new_signature=r['signature'],old_receipt_sha256=digest))
dump(ROOT/'audits/report_repair/plot_inputs_migration.json',dict(status='PASS',fix='provide exactly QPS=(0,5,9) to standalone plotting entry',old_code_hashes=old,new_code_hashes=new,tasks=rows,numerical_code_changed=False,scientific_calculations_repeated=0))
print('PLOT INPUT MIGRATION PASS',len(rows))
