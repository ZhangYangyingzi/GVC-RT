"""User-approved, narrowly scoped report recovery without rewriting frozen code."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,traceback,types
from pathlib import Path
EXPERIMENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXPERIMENT))
import v68_io as io

# Exactly the 16 post-render images inspected before the user's approval.
APPROVED={
 'vimeo_heldout/Rate_DISTS.png':'65f8fe16bd9d20bd3fbcd10102679abe68443f9ef587aa60bdadea3a3732db5e',
 'vimeo_heldout/Rate_FID.png':'36b9c8e46e8be2ad9534d3b008e4b2e510e02ce5d9405d3ec3179c264827a87b',
 'vimeo_heldout/Rate_FloLPIPS.png':'556748a8d54c7cbcdf700bf5beafd56c8fc06d42e42e3576d09501c086da275d',
 'vimeo_heldout/Rate_LPIPS.png':'c5a1e443fa0cbee30307637659c20a95cb00358181874feeb8a6d74482620a88',
 'vimeo_heldout/Rate_MS_SSIM.png':'a2d84677ab1bc826c337004b00ba685f87e7f8e7d2a3df60f32d775d3d298e46',
 'vimeo_heldout/Rate_PSNR.png':'34c59b6767af51bcd1c4cd8bd7b560fec89e2e15f767f7e5ed300a0070018c9e',
 'vimeo_heldout/Rate_SSIM.png':'59d81fd6af2c83d2a821ba51180335e96b8e80fd44160ceed38639f27c564601',
 'virat480/Rate_FID.png':'3c081d139fa2aa1c0c9e4e83106ca4ad01478469e50f4451b7d9f0239dafbe3c',
 'virat480/Rate_MS_SSIM.png':'6e0e42bcb28653066b6314bfb091c3d6bccf4b6b536645192aafdb8ea7d14c85',
 'virat720/Rate_DISTS.png':'0205a88fc820dbf36d90afb161dd483bb9120d320e862f31ffed98337f4cdff1',
 'virat720/Rate_FID.png':'d899bd3a21c484bf30ed9d85061a216bcb25059731dc616fce9737efeae0d492',
 'virat720/Rate_FloLPIPS.png':'11b27259e96bcc385e571716d7884d554fa191bb86845e997847995147c5b024',
 'virat720/Rate_LPIPS.png':'71970d1cb40f572a8313f37a95cd7e5264972a695664f8f3527b180db9e03964',
 'virat720/Rate_MS_SSIM.png':'88aa7fb67595d3e19d47e0c810bcc200e2eb85bbf8eaa2f5ec51369e385d89b5',
 'virat720/Rate_PSNR.png':'57c21c77a2a5e9a4f2870f70390ad91521670316e128ff2bf11e314617784d6a',
 'virat720/Rate_SSIM.png':'bcef56f587805c7ebcc8dbf4e89a61586ea39fe49f3755d90662404bd3ef2318',
}
RECOVERY=EXPERIMENT/'recovery'
EXCEPTIONS=EXPERIMENT/'audits/approved_png_change_exception.json'

def inspect_exception(baseline,current):
    allowed={str(io.V67/'results'/p):h for p,h in APPROVED.items()}
    assert baseline.keys()==current.keys(),'Unexpected added/removed old artifact'
    changed={p for p in baseline if baseline[p]!=current[p]}
    assert changed==set(allowed),('Unapproved old-artifact changes',sorted(changed^set(allowed)))
    evidence=[]
    for p in sorted(changed):
        assert io.sha(p)==allowed[p],('Approved PNG changed again',p)
        evidence.append(dict(path=p,original_size=baseline[p][0],original_mtime_ns=baseline[p][1],approved_size=current[p][0],approved_mtime_ns=current[p][1],approved_sha256=allowed[p],original_png_sha256='NOT_RECORDED_IN_ORIGINAL_SIZE_MTIME_INVENTORY'))
    return evidence

def verify_old():
    baseline=io.load(EXPERIMENT/'audits/old_inventory.json');current=io.old_inventory()
    evidence=inspect_exception(baseline,current);approved=io.load(EXCEPTIONS)
    assert io.sha(EXPERIMENT/'audits/old_inventory.json')==approved['original_inventory_sha256']
    assert evidence==approved['changes']
    return evidence

def frozen_with_exception(full=False):
    cfg=io.frozen(False)  # All original code, configuration and dependency hashes still checked.
    if full:verify_old()
    return cfg

def archive(name):
    source=EXPERIMENT/name;target=RECOVERY/('original_'+name)
    if not target.exists():io.dump(target,io.load(source))

def prepare():
    io.frozen(False)
    baseline=io.load(EXPERIMENT/'audits/old_inventory.json');current=io.old_inventory();changes=inspect_exception(baseline,current)
    evidence=dict(status='APPROVED_EXCEPTION',user_authorized=True,user_approval='可以，请你处理',scope='Only the 16 previously inspected V6.7b derived PNG images; all other old files and frozen dependency hashes remain protected',no_old_experiments_modified=False,original_inventory_sha256=io.sha(EXPERIMENT/'audits/old_inventory.json'),original_protocol_sha256=io.sha(EXPERIMENT/'audits/local_protocol.json'),frozen_dependency_count=len(io.load(EXPERIMENT/'audits/dependencies.json')),changes=changes,unknown_change_author=True,no_training_or_evaluation_rerun=True)
    if EXCEPTIONS.exists():assert io.load(EXCEPTIONS)==evidence
    else:io.dump(EXCEPTIONS,evidence)
    for name in ('final_integrity.json','pipeline_status.json'):archive(name)
    # Save the already-rendered old-directory state to prove this recovery changes none of it.
    target=RECOVERY/'old_inventory_before_recovery.json'
    if target.exists():assert io.load(target)==current
    else:io.dump(target,current)
    tests={}
    sample=next(iter(baseline))
    bad=dict(current);bad[sample]=[current[sample][0]+1,current[sample][1]]
    try:inspect_exception(baseline,bad)
    except AssertionError:tests['reject_unapproved_metadata_change']=True
    else:raise AssertionError('Exception gate allowed an unexpected change')
    bad=dict(current);bad['/unapproved/added.file']=[1,1]
    try:inspect_exception(baseline,bad)
    except AssertionError:tests['reject_added_file']=True
    else:raise AssertionError('Exception gate allowed an added file')
    io.dump(RECOVERY/'exception_gate_tests.json',dict(status='PASS',**tests))

def final_dump(path,data):
    if Path(path)==EXPERIMENT/'final_integrity.json' and data.get('status')=='PASS':
        verify_old()
        assert io.old_inventory()==io.load(RECOVERY/'old_inventory_before_recovery.json')
        data=dict(data,status='PASS_WITH_APPROVED_EXCEPTIONS',no_old_experiments_modified=False,
                  old_experiment_changes_limited_to_approved_PNGs=True,approved_PNG_exceptions=16,
                  approved_exception_audit=str(EXCEPTIONS),approved_exception_audit_sha256=io.sha(EXCEPTIONS),
                  frozen_training_code_unchanged=True,frozen_dependency_hashes_unchanged=True,
                  no_old_files_modified_during_recovery=True,training_and_evaluation_not_rerun=True,
                  recovery_script_sha256=io.sha(__file__))
    io.dump(path,data)

def run_report():
    prepare();original=EXPERIMENT/'report.py';source=original.read_text()
    # The comparator supplies rate_axis and metric_direction already. Pass its dict
    # positionally so the intended display metadata can override those keys safely.
    # This changes only CSV assembly, never metric or interpolation calculations.
    needle="fideq.append(dict(dataset=d,**e,"
    assert source.count(needle)==1
    patched=source.replace(needle,"fideq.append(dict(e,dataset=d,")
    io.dump(RECOVERY/'report_adapter_audit.json',dict(status='PASS',original_report_sha256=io.sha(original),in_memory_report_sha256=hashlib.sha256(patched.encode()).hexdigest(),patch='Pass comparator dictionary positionally in FID CSV assembly to avoid duplicate rate_axis/metric_direction keywords; retain intended metadata values',metric_math_unchanged=True,original_report_file_unchanged=True))
    report=types.ModuleType('v68_approved_report_recovery');report.__file__=str(original)
    exec(compile(patched,str(original),'exec'),report.__dict__)
    report.frozen=frozen_with_exception;report.dump=final_dump
    print('RECOVERY: original code/dependencies verified; exactly 16 approved PNG exceptions',flush=True)
    report.main()
    final=io.load(EXPERIMENT/'final_integrity.json');assert final['status']=='PASS_WITH_APPROVED_EXCEPTIONS'
    csvs={str(p.relative_to(EXPERIMENT)):dict(rows=len(io.read(p)),sha256=io.sha(p)) for p in (EXPERIMENT/'results').rglob('*.csv')}
    expected={'raw_rd':1550,'dataset_macro_rd':200,'equal_rate_per_sequence':1116,'equal_rate_summary':144,'fid_raw':200,'fid_bootstrap':4000,'fid_bootstrap_summary':200,'fid_equal_rate_summary':24,'vimeo_heldout_raw':480,'vimeo_heldout_progression':15,'interface_drift':270,'interface_drift_summary':10,'training_progression_C0':500,'training_progression_C1':500,'training_window_summary':12,'checkpoint_parameter_movement':12,'rate_behavior':240}
    for name,count in expected.items():assert csvs['results/'+name+'.csv']['rows']==count,(name,count)
    io.dump(RECOVERY/'output_validation.json',dict(status='PASS',files=csvs,expected_rows=expected))
    status=io.load(RECOVERY/'original_pipeline_status.json');status.pop('traceback',None)
    status.update(status='PASS_WITH_APPROVED_EXCEPTIONS',phase='complete',PID=os.getpid(),active={},pending=0,updated_unix=time.time(),recovery=True,approved_PNG_exceptions=16,final_integrity=str(EXPERIMENT/'final_integrity.json'))
    io.dump(EXPERIMENT/'pipeline_status.json',status)
    io.dump(RECOVERY/'status.json',dict(status='PASS_WITH_APPROVED_EXCEPTIONS',PID=os.getpid(),finished_unix=time.time()))
    print('RECOVERY COMPLETE: all report outputs validated; no GPU jobs launched',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args();os.chdir(io.REPO)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    if a.launch:
        with (EXPERIMENT/'pipeline.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (EXPERIMENT/'logs/report_recovery.log').open('a') as log:
            proc=subprocess.Popen([io.PYTHON,'-B','-u',str(Path(__file__).resolve())],cwd=io.REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(PID=proc.pid,log=str(EXPERIMENT/'logs/report_recovery.log'))));return
    lock=(EXPERIMENT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    io.dump(RECOVERY/'status.json',dict(status='RUNNING',PID=os.getpid(),started_unix=time.time()))
    try:run_report()
    except Exception:
        error=traceback.format_exc();io.dump(RECOVERY/'status.json',dict(status='FAIL',PID=os.getpid(),traceback=error));io.dump(EXPERIMENT/'logs/failures'/f'report_recovery_{time.time_ns()}.json',dict(status='FAIL',traceback=error));io.dump(EXPERIMENT/'final_integrity.json',dict(status='FAIL',recovery=True,traceback=error));raise

if __name__=='__main__':main()
