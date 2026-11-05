"""Explicit logging-only migration at committed Router250; never performs an update."""
from common_s2 import *
import ast,io,torch

def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    env()
    with lock('runner.lock'),lock('engineering.lock'):
        before=load(ROOT/'audits/event_time_repair_before.json')
        current_manifest=ROOT/'source_manifest.json'
        if sha(current_manifest)!=before['old_source_manifest_sha256']:
            audit=load(ROOT/'audits/event_time_repair.json')
            assert audit['status']=='PASS' and audit['new_source_manifest_sha256']==sha(current_manifest)
            verify_sources();print('EVENT_TIME_REPAIR_ALREADY_APPLIED',flush=True);return
        old=load(current_manifest)
        changes=[name for name,h in old['code'].items() if sha(ROOT/name)!=h]
        assert changes==['runner.py'],('UNEXPECTED_SOURCE_CHANGE',changes)
        archived=(ROOT/'audits/code_versions/v4_before_event_time_fix/runner.py').read_text()
        expected=archived.replace("event('VALIDATION_COMPLETE',**state['checks'][-1])","event('VALIDATION_COMPLETE',validation_check=state['checks'][-1])")
        assert (ROOT/'runner.py').read_text()==expected and expected!=archived
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
        raw=(ROOT/'checkpoints/latest.pt').read_bytes();assert hashlib.sha256(raw).hexdigest()==before['old_latest_sha256']
        saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        control=load(ROOT/'stopping_state.json')
        assert saved['committed_router_updates']==250 and saved['next_cursor']==1750
        assert saved['validation_state']==control and control['last_validation_step']==250
        assert len(control['checks'])==1 and control['bad_check_count']==1
        assert sha(ROOT/'stopping_state.json')==before['old_stopping_sha256']
        assert all(int(v['step'])==250 for v in saved['optimizer']['state'].values())
        assert load(ROOT/'receipts/update_00250.json.gz')==saved['last_committed_receipt']
        # Exercise the real logger with the saved timestamp-bearing validation payload.
        event('EVENT_TIME_REPAIR_PROBE',validation_check=control['checks'][-1])
        last=json.loads((ROOT/'logs/events.jsonl').read_text().splitlines()[-1])
        assert last['event']=='EVENT_TIME_REPAIR_PROBE' and last['validation_check']==control['checks'][-1]
        assert isinstance(last['time'],float) and last['validation_check']['time']==control['checks'][-1]['time']
        for filename,h in before['immutable_checkpoints'].items():assert sha(ROOT/'checkpoints'/filename)==h
        new=dict(old,revision='router_only_code_freeze_v5_validation_event_logging',time=time.time(),
            previous_manifest_sha256=before['old_source_manifest_sha256'],
            logging_only_migration='audits/event_time_repair.json',
            code={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py')) if not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','large_artifacts'])})
        assert old['protocol_sha256']==sha(ROOT/'protocol.json') and old['plan_index_sha256']==sha(ROOT/'plans/index.json')
        dump(current_manifest,new)
        after=dict(saved)
        after['source_manifest_sha256']=sha(current_manifest)
        after['source_manifest_migration_chain']=[*saved.get('source_manifest_migration_chain',[]),dict(record='audits/event_time_repair.json',
            old_source_sha256=before['old_source_manifest_sha256'],new_source_sha256=sha(current_manifest),formal_updates=250,
            change='Validation event payload nested to prevent duplicate time keyword; numerical code unchanged')]
        atomic_torch(ROOT/'checkpoints/latest.pt',after)
        reloaded=torch.load(ROOT/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
        changed_fields=[k for k in saved if not same(saved[k],reloaded[k])]
        assert set(changed_fields)=={'source_manifest_sha256','source_manifest_migration_chain'}
        assert same(saved['router_state'],reloaded['router_state']) and same(saved['optimizer'],reloaded['optimizer'])
        assert same(saved['all_rank_RNG'],reloaded['all_rank_RNG']) and saved['validation_state']==reloaded['validation_state']
        assert sha(ROOT/'stopping_state.json')==before['old_stopping_sha256']
        for filename,h in before['immutable_checkpoints'].items():assert sha(ROOT/'checkpoints'/filename)==h
        audit=dict(status='PASS',time=time.time(),failure_stage='development_validation_complete',fixed_call='event(VALIDATION_COMPLETE, validation_check=record)',
            old_source_manifest_sha256=before['old_source_manifest_sha256'],new_source_manifest_sha256=sha(current_manifest),
            old_latest_sha256=before['old_latest_sha256'],new_latest_sha256=sha(ROOT/'checkpoints/latest.pt'),
            old_runner_sha256=before['old_runner_sha256'],new_runner_sha256=sha(ROOT/'runner.py'),
            existing_source_changes=changes,changed_checkpoint_metadata_fields=changed_fields,
            committed_router_updates=250,source_cursor=1750,all_four_rank_RNG_exact=True,optimizer_exact=True,Router_state_exact=True,
            immutable_checkpoints_unchanged=True,validation_count=1,bad_check_count=1,validation_state_not_recounted=True,
            actual_logger_regression_probe='PASS',protocol_and_plan_unchanged=True,model_training_loss_and_evaluation_sources_unchanged=True,
            new_optimizer_updates_during_repair=0)
        dump(ROOT/'audits/event_time_repair.json',audit);verify_sources();print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
