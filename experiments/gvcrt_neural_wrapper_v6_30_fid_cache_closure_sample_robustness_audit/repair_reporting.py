"""Verify reporting-only changes and retain all completed numerical tasks."""
from audit_io import *
def main():
    folder=ROOT/'audits/report_repair';old=load(folder/'old_code_hashes.json')
    before=(folder/'old_finalize30.py').read_text();assert hashlib.sha256(before.encode()).hexdigest()==old['finalize30.py']
    start=before.index('    if pooled:\n');end=before.index('    integrity.update(report_sha256',start);block=before[start:end]
    replacement="    if pooled:\n        import subprocess\n        dump(ROOT/'audits/plot_payload.json',dict(pooled=pooled,newpoints=newpoints,methods=METHODS,datasets=DATASETS))\n        subprocess.run(['/data1/anaconda3_new/anaconda_program/bin/python','-B',str(ROOT/'plot30.py')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))\n"
    assert before[:start]+replacement+before[end:]==(ROOT/'finalize30.py').read_text(), 'non-plot aggregation changed'
    helper=(ROOT/'plot30.py').read_text();extracted='\n'.join(line[4:] if line.startswith('    ') else line for line in block.split('\n')).rstrip()+'\n'
    assert helper.endswith(extracted),'plot values or rendering changed'
    before=(folder/'old_run_all.py').read_text();assert hashlib.sha256(before.encode()).hexdigest()==old['run_all.py']
    expected=before.replace("    env=dict(os.environ,V630_PUBLICATION_PHASE='final_integrity_verified')", "    phase='final_integrity_verified' if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json')['status']=='PASS' else 'execution_failure_findings'\n    env=dict(os.environ,V630_PUBLICATION_PHASE=phase)")
    expected=expected.replace("        if (ROOT/'config.json').exists():publish()\n        raise", "        if (ROOT/'config.json').exists():publish()\n        setstate(status='FAILED',phase='finished',running=0)\n        raise")
    assert expected==(ROOT/'run_all.py').read_text(),'unexpected scheduler change'
    before=(folder/'old_publish_github.py').read_text();assert hashlib.sha256(before.encode()).hexdigest()==old['publish_github.py']
    addition="        # Archive obsolete owned terminal markers locally, and remove only their stale remote copies.\n        # Historical failure evidence is retained in audits/archived_terminal_states/.\n        for name in ('FAILED.json','BLOCKED.json','DONE.json'):\n            if not (ROOT/name).exists() and (dest/name).exists():\n                run(['git','rm','--',str(relative/name)],work)\n"
    expected=before.replace('        for off in range(0,len(paths),100):',addition+'        for off in range(0,len(paths),100):')
    assert expected==(ROOT/'publish_github.py').read_text(),'unexpected publisher change'
    now={p.name:sha(p) for p in ROOT.glob('*.py')};changed={'run_all.py','publish_github.py','finalize30.py'}
    assert set(now)==set(old)|{'plot30.py','repair_reporting.py'}
    assert all(now[n]==h for n,h in old.items() if n not in changed),'numerical implementation changed'
    rows=[]
    for p in sorted((ROOT/'tasks').glob('*.json')):
        r=load(p);t=r['task']
        if valid(t):continue
        assert r['signature']==signature(t,old),('frozen input or prior code mismatch',p)
        assert r['execution_complete'] and all(Path(a).is_file() and sha(a)==h for a,h in r['artifacts'].items())
        digest=sha(p);previous=r['signature'];r['signature']=signature(t)
        r['reporting_only_signature_migration']=dict(original_receipt_sha256=digest,original_signature=previous,audit='audits/report_repair/migration.json',all_numerical_code_and_artifacts_unchanged=True)
        dump(p,r);assert valid(t);rows.append(dict(task=task_id(t),original_receipt_sha256=digest,original_signature=previous,new_signature=r['signature']))
    dump(folder/'migration.json',dict(status='PASS',cause='matplotlib absent in gvc-rt environment',existing_plot_python='/data1/anaconda3_new/anaconda_program/bin/python',plotting_package_version='3.9.2',third_party_packages_modified=False,changed_numerical_code=[],old_code_hashes=old,new_code_hashes=now,verified_tasks=len(rows),tasks=rows,repeated_decodes_encodes_features_and_FID=0,old_failure_evidence_retained=True))
    print('REPORTING MIGRATION PASS',len(rows),flush=True)
if __name__=='__main__':main()
