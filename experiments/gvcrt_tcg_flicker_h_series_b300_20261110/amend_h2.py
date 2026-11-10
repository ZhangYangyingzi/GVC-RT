"""Apply explicit user instruction withdrawing the assistant-added H2 cutoff."""
from common_s2 import *
import copy,ast

def main():
 env();request=load(ROOT/'audits/H2_user_start_request.json');assert request['status']=='ACCEPTED'
 history=ROOT/'audits/protocol_revisions/H2_user_directed_start';history.mkdir(parents=True,exist_ok=True)
 if (history/'amendment.json').exists():print('AMENDMENT_ALREADY_APPLIED');return
 old_protocol=load(ROOT/'protocol.json');old_protocol_sha=sha(ROOT/'protocol.json');old_manifest=load(ROOT/'source_manifest.json');old_gate=load(ROOT/'audits/H2_gate.json')
 dump(history/'protocol_before.json',old_protocol);dump(history/'source_manifest_before.json',old_manifest);dump(history/'H2_gate_before.json',old_gate)
 code_before={n:sha(ROOT/n) for n in ['train.py','pipeline.py','runner.py','selection.py','media.py','integrity.py','results.py','preflight.py','freeze.py']}
 # Numeric losses/model/data/training/selection remain byte-for-byte in JSON values.
 protocol=copy.deepcopy(old_protocol);gate_cfg=protocol['H2_gate'];gate_cfg.pop('ratio_threshold',None);gate_cfg.pop('failure_action',None);gate_cfg.pop('pass_rule',None);gate_cfg.pop('pre_registered_before_measurement',None)
 gate_cfg.update(training_allowed=True,decision_authority='explicit user instruction: 不行，你不能有自己的门槛，你应该启动h2',control_errors='diagnostic only; no assistant-defined numeric launch cutoff',required_implementation_checks=['same-dtype B300 step0 equality','finite losses/gradients','stateless auxiliary target/shifted paths','main-history object/value/gradient-connection preservation','correct sparse four-call/window and globalbatch normalization'],historical_gate_record=str(history/'H2_gate_before.json'),assistant_added_cutoff_withdrawn=True)
 assert {k:v for k,v in old_protocol.items() if k!='H2_gate'}=={k:v for k,v in protocol.items() if k!='H2_gate'}
 dump(ROOT/'protocol.json',protocol)
 new_gate=copy.deepcopy(old_gate);new_gate['previous_status']=old_gate['status'];new_gate['status']='DIAGNOSTIC_ONLY_USER_DIRECTED_TRAINING';new_gate['withdrawn_assistant_threshold']=new_gate.pop('pre_registered_threshold',.1);new_gate['H2_training_allowed']=True;new_gate['decision_authority']='explicit user instruction';new_gate['original_measurement_preserved']=True;new_gate['historical_record']=str(history/'H2_gate_before.json');new_gate['amended_at']=time.time();dump(ROOT/'audits/H2_gate.json',new_gate)
 (ROOT/'execution_policy.py').write_text('''"""User-authorized execution changes; original numeric experiment settings preserved."""
from common_s2 import *
def h2_training_allowed():
    return load(ROOT/'protocol.json')['H2_gate'].get('training_allowed',False) is True

def verify_checkpoint_protocol(saved_hash):
    current=sha(ROOT/'protocol.json')
    if saved_hash==current:return True
    amendment=load(ROOT/'audits/protocol_revisions/H2_user_directed_start/amendment.json')
    old=Path(amendment['prior_protocol_path'])
    assert sha(old)==amendment['prior_protocol_SHA256']==saved_hash
    before=load(old);now=load(ROOT/'protocol.json')
    assert {k:v for k,v in before.items() if k!='H2_gate'}=={k:v for k,v in now.items() if k!='H2_gate'}
    assert h2_training_allowed() and amendment['numeric_model_loss_data_optimizer_selection_unchanged']
    return True
''')
 p=ROOT/'train.py';s=p.read_text().replace('from common_s2 import *','from common_s2 import *\nfrom execution_policy import h2_training_allowed,verify_checkpoint_protocol',1).replace("if group==GROUPS[2]:assert load(ROOT/'audits/H2_gate.json')['status']=='PASS'","if group==GROUPS[2]:assert h2_training_allowed()")
 s=s.replace("assert a['protocol_SHA256']==sha(ROOT/'protocol.json') and a['plan_SHA256']==sha(ROOT/'plans/index.json')", "assert verify_checkpoint_protocol(a['protocol_SHA256']) and a['plan_SHA256']==sha(ROOT/'plans/index.json')");p.write_text(s)
 p=ROOT/'pipeline.py';s=p.read_text().replace('from common_s2 import *','from common_s2 import *\nfrom execution_policy import h2_training_allowed',1).replace("if gate.exists() and load(gate)['status']=='MECHANICAL_CHECK_FAILED':out[GROUPS[2]]=400", "if gate.exists() and not h2_training_allowed():out[GROUPS[2]]=400")
 a=s.index('def task_list(');b=s.index('\ndef produce(',a)
 s=s[:a]+'''def task_list(rows,cfg,pos):
    cursors=sorted({n for n in pos.values() if n<400});safe=min(pos.values());requested=[]
    for p in sorted((ROOT/'cache/requests').glob('*.json'),key=lambda p:p.stat().st_mtime):
        a=load(p)
        if a['training'] and a['row'].get('plan_slot',400)<safe:p.unlink();continue
        if not ready(a['row'],a['qp'],a['training']):requested.append(a)
    # Bounded, fair lookahead for groups at different committed cursors.
    # H2 backfill does not prevent H0/H1 next-batch preparation.
    limit=max(4,cfg['training_prefetch_windows']//max(1,len(cursors)))
    planned=[]
    for offset in range(limit):
        for cursor in cursors:
            index=cursor+offset
            if index>=400:continue
            row=rows[index]
            if not ready(row,row['external_qp'],True):planned.append(dict(row=row,qp=row['external_qp'],training=True))
    urgent={rows[i]['sample_key'] for cursor in cursors for i in range(cursor,min(400,cursor+4))}
    native=[a for a in requested if not a['training']];extra=[a for a in requested if a['training'] and a['row']['sample_key'] not in {x['row']['sample_key'] for x in planned}]
    early=[a for a in planned+extra if a['row']['sample_key'] in urgent];later=[a for a in planned+extra if a['row']['sample_key'] not in urgent]
    enough=all(sum(ready(r,r['external_qp'],True) for r in rows[n:min(400,n+16)])>=min(16,400-n) for n in cursors)
    return early+(native+later if enough else later+native)
''' +s[b:];p.write_text(s)
 p=ROOT/'runner.py';s=p.read_text().replace('from common_s2 import *','from common_s2 import *\nfrom execution_policy import h2_training_allowed',1).replace("load(ROOT/'audits/H2_gate.json')['status']!='PASS'","not h2_training_allowed()").replace("load(ROOT/'audits/H2_gate.json')['status']=='PASS'","h2_training_allowed()")
 # Smoke H2 on GPU2 while already-running H0/H1 keep their own processes.
 p.write_text(s)
 for name in ['selection.py','media.py','integrity.py']:
  p=ROOT/name;s=p.read_text().replace('from common_s2 import *','from common_s2 import *\nfrom execution_policy import h2_training_allowed',1)
  s=s.replace("gate['status']!='PASS'","not h2_training_allowed()").replace("load(ROOT/'audits/H2_gate.json')['status']!='PASS'","not h2_training_allowed()").replace("load(ROOT/'audits/H2_gate.json')['status']=='PASS'","h2_training_allowed()")
  s=s.replace('H2_mechanical_failure_explicit=True','H2_user_authority_and_original_controls_recorded=True');p.write_text(s)
 p=ROOT/'preflight.py';s=p.read_text();a=s.index(' gate_pass=statistics.median(ratios)<');b=s.index(" dump(ROOT/'audits/preflight.json'",a)
 s=s[:a]+''' write_csv(ROOT/'equivariance_controls.csv',controls)
 dump(ROOT/'audits/H2_gate.json',dict(status='DIAGNOSTIC_ONLY_USER_DIRECTED_TRAINING',median_16px_to_8px_error_ratio=statistics.median(ratios),ratios=ratios,unchanged_padding=True,crop64_not_guarantee_boundary_eliminated=True,H2_training_allowed=True,decision_authority='explicit user instruction',time=time.time()))
''' +s[b:];s=s.replace("print('H_PREFLIGHT_PASS H2_GATE',gate_pass,statistics.median(ratios),flush=True)","print('H_PREFLIGHT_PASS H2_CONTROLS_DIAGNOSTIC',statistics.median(ratios),flush=True)");p.write_text(s)
 p=ROOT/'freeze.py';s=p.read_text().replace("files=['common_s2.py'","files=['execution_policy.py','common_s2.py'");p.write_text(s)
 p=ROOT/'results.py';s=p.read_text().replace("Fixed threshold0.1 operationalizes separation by more than one order of magnitude; no padding/crop/target tuning after measurements.","The assistant-added0.1 launch cutoff was withdrawn by explicit user instruction. Original controls remain unchanged and diagnostic; H2 is trained with the original calibrated objective, padding and crop.")
 s=s.replace("H2 missing rows are due to its mechanical gate, not replacement by another model.","H2 started later after the user withdrew the assistant-added cutoff; its matched-step rows are populated from actual completed evaluations.")
 p.write_text(s)
 amendment=dict(status='USER_AUTHORIZED',user_instruction=request['user_instruction'],prior_protocol_path=str(history/'protocol_before.json'),prior_protocol_SHA256=old_protocol_sha,new_protocol_SHA256=sha(ROOT/'protocol.json'),numeric_model_loss_data_optimizer_selection_unchanged=True,H0_H1_processes_and_progress_preserved=request['H0_H1_unchanged'],H2_initialization='same complete B300, new optimizer, same RNG seed and400-window plan; zero previous formal H2 updates',coefficient_source=dict(path=str(ROOT/'gradient_calibration.json'),SHA256=sha(ROOT/'gradient_calibration.json')),original_controls_preserved=dict(path=str(history/'H2_gate_before.json'),SHA256=sha(history/'H2_gate_before.json')),time=time.time());dump(history/'amendment.json',amendment)
 manifest=copy.deepcopy(old_manifest);manifest['protocol_sha256']=sha(ROOT/'protocol.json')
 changed=[]
 for n,h in code_before.items():
  if sha(ROOT/n)!=h:
   if n in manifest['code']:assert manifest['code'][n]==h;manifest['code'][n]=sha(ROOT/n)
   changed.append(dict(file=n,before_SHA256=h,after_SHA256=sha(ROOT/n)))
 manifest['code']['execution_policy.py']=sha(ROOT/'execution_policy.py');manifest.setdefault('user_authorized_amendments',[]).append(dict(record=str(history/'amendment.json'),SHA256=sha(history/'amendment.json'),code_changes=changed));dump(ROOT/'source_manifest.json',manifest)
 for p in ROOT.glob('*.py'):ast.parse(p.read_text())
 print('H2_USER_AMENDMENT_APPLIED',sha(ROOT/'protocol.json'),len(changed))
if __name__=='__main__':main()
