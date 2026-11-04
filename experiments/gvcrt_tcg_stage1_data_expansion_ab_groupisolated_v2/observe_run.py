"""CPU-only status observer; never launches training, changes settings, or commits updates."""
from ab_common import *
import contextlib,datetime,io,traceback

def optional(relative):
 path=ROOT/relative
 return load(path) if path.exists() else {}

def held(relative):
 path=ROOT/relative
 if not path.exists():return False
 with path.open('a') as lease:
  try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
 return False

def snapshot():
 import current_status
 output=io.StringIO()
 with contextlib.redirect_stdout(output):current_status.main()
 result=json.loads(output.getvalue());raw=optional('run_status.json')
 names=['runner.lock','equivalence.lock','evaluation.lock','evaluation_development_workers.lock','evaluation_final_workers.lock','integrity.lock','publication.lock']
 names += [f'{a}/{name}' for a in ('A_clean_old_pool','B_expanded_ulong') for name in ('training.lock','rank0_active.lock')]
 leases={name:held(name) for name in names}
 result['snapshot_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 result['coordinator_lease_held']=leases['runner.lock'];result['active_stage_leases']=[n for n,v in leases.items() if v]
 if result['status']=='RUNNING' and not any(leases.values()):
  result['status']='STOPPED_UNEXPECTEDLY';result['failure_stage']=raw.get('stage');result['observer_note']='No same-task coordinator/stage lifetime lease is held; original run_status was RUNNING.'
 result['engineering_checks']={key:optional(path).get('status','NOT_RUN') for key,path in [('runtime_equivalence','audits/runtime_equivalence.json'),('historical_RANS_roundtrip','audits/rans_engineering.json'),('focused_native_geometry','repairs/native_geometry_v1/focused_validation.json'),('A_first_two_updates','A_clean_old_pool/smoke.json'),('B_first_two_updates','B_expanded_ulong/smoke.json')]}
 result['final_integrity_status']=optional('final_integrity.json').get('status','NOT_RUN')
 result['repair_audit_paths']=['ENGINEERING_REPAIR.md','repairs/native_geometry_v1/transition.json','repairs/native_geometry_v1/source_revision.json','repairs/native_geometry_v1/focused_validation.json']
 result['resume_command']=f'env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 {PYTHON} -B -u {ROOT}/launch_repaired.py'
 result['resume_precondition']='Track a held coordinator/stage lease. After an actual exit, resolve its recorded error and resume the exact committed checkpoint; never repeat updates or alter the frozen data protocol.'
 return result

def persist(record):
 # Mutable observer output is excluded by publish.py's logs allowlist. Stop
 # touching REPORT before the evaluator/summarizer/publication owns that file.
 dump(ROOT/'logs/live_status.json',record)
 with (ROOT/'logs/status_observer.log').open('a') as log:log.write(json.dumps(record,ensure_ascii=False)+'\n')
 final_report_owned=record['current_stage'] in ('new_evaluation','integrity','ready_for_publication','publication','complete') or optional('summary.json').get('status')=='PASS'
 if not final_report_owned:
  body=['# TCG source-group-isolation-v2 execution snapshot','',f"Snapshot UTC: {record['snapshot_UTC']}",'This is a dated engineering snapshot, not final experiment acceptance. Live observed state: logs/live_status.json; authoritative updates: atomic latest checkpoints and per-update receipts.','', '```json',json.dumps(record,indent=2,ensure_ascii=False),'```','', 'Data plans are frozen: N_A923, N_B8470, S12710 per arm; final absolute optimizer step17710. Each arm uses4 synchronized ranks on physical GPUs4–7,2 UVG/2 U-Long per update; A then B. LR/loss/precision/QP/detach/TCG-only configuration remains as registered in protocol.json.','', 'All30447 unique plan windows were accepted (25484 native decoded,4963 trusted source/window/crop matches). No full data-plan decoding was repeated for startup repair. Source-group evidence is in graph/ and split_audit.json; historical exposure remains in historical_exposure.json. Sparse screens do not exhaust all offsets/crops/reencodes or establish foundation-pretraining exclusion.','', 'Parent156 GT-flow points are hash-verified E1 reuse,9984 frames/9828 transitions; they are not new A/B evaluations. New development/final point counts appear above. Historical references remain separate from new held-out tests.','', 'Directory initialization is isolated in launch_repaired.py; original directory-failure logs are retained. A stopped before update11 because15 old-pool native heights differ from erroneously appended1080 metadata. The original20-file manifest and the3 revised source predecessors are archived; unchanged protocol/plans/splits and exact checkpoint10 resume authority are documented in ENGINEERING_REPAIR.md and repairs/native_geometry_v1/. Full native geometry, RGB/crop hashes and original numerical tolerances remain checked.','', 'Existing sampler convention is retained: pixel-center grid,align_corners=False,W−1/H−1 normalization,invalid zero/full-image denominator; width11/dx1 gives1.1 sampling pixels. No exact S2VC implementation claim.','', 'Final integrity is not inferred from a running job or engineering check. Publication status and verified commits are recorded only after the actual checks and push. This snapshot contains no scientific interpretation.']
  target=ROOT/'REPORT.md';temp=target.with_name('REPORT.observer.tmp');temp.write_text('\n'.join(body)+'\n');temp.replace(target)

def main():
 with lock('status_observer.lock'):
  dump(ROOT/f'audits/status_observer_start_{int(time.time())}.json',dict(scope='CPU-only observation and dated report snapshots; no training/model/optimizer change',source_sha256=sha(ROOT/'observe_run.py'),source_manifest_sha256=sha(ROOT/'source_manifest.json'),started_unix=time.time(),actual_invocation=[sys.executable,*sys.argv]))
  while True:
   record=snapshot();persist(record)
   print(json.dumps({k:record[k] for k in ('status','current_stage','snapshot_UTC','A','B','engineering_checks','final_integrity_status')},ensure_ascii=False),flush=True)
   if '--watch' not in sys.argv:return 0
   if record['status'] in ('FAILED','BLOCKED','STOPPED_UNEXPECTEDLY'):return 1
   if record['status']=='PASS' and record['github']['remote_verified'] and record['github']['core_blobs_verified']:return 0
   time.sleep(30)

if __name__=='__main__':
 try:sys.exit(main())
 except BlockingIOError:print('An existing same-task status observer holds its lease; no duplicate observer started.',flush=True)
 except Exception:traceback.print_exc();sys.exit(1)
