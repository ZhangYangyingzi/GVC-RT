"""User-authorized V6.18-only cleanup and factual initialization exposure audit."""
import re,shutil
from io18 import *
UUID=re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',re.I)
def archive_previous():
    marker=ROOT/'audits/cleanup_authorization.json'
    if marker.exists():return
    assert not list((ROOT/'branches').glob('*/checkpoints/*.pt')),'Training already started: do not regenerate inputs'
    targets=[ROOT/n for n in ('config.json','protocol.json','dataset_split.json','sampling_order.json','preflight_audit.json','final_integrity.json','pipeline_status.json','blocking_error.json','train_manifest_ulong.json','train_manifest_ulong_expanded.json','train_manifest_uvg.json','train_manifest_uvg_expanded.json')]
    targets+=[*(ROOT/'plans').glob('*.json'),*(ROOT/'evaluation').glob('*.json'),*(ROOT/'branches').glob('*/initialization_audit.json'),*(ROOT/'branches').glob('*/checkpoint_index.json'),ROOT/'audits/protocol_hashes.json',ROOT/'audits/implementation_check.json',ROOT/'audits/step0_codec_equivalence.json']
    directory=ROOT/'archived_before_cleanup';records=[]
    for p in targets:
        if not p.exists():continue
        dest=directory/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();digest=sha(p);p.replace(dest);records.append(dict(original_path=str(p),archived_path=str(dest),sha256=digest))
    dump(marker,dict(status='PASS',authorization='user explicitly selected exclusion of the four historical test sources; cleaned base pool 1020; no need to replenish to 1024',archived=records,source_checkpoint_unchanged=True))
def clean_original():
    evidence=load(ROOT/'audits/original_pool_historical_test_conflicts.json');ids={r['identity'] for r in evidence['conflicts']};hashes={r['v617_training_record']['sha256'] for r in evidence['conflicts']};old=load(V17/'train_manifest_ulong.json')['videos'];keep=[];removed=[]
    assert len(ids)==4 and len(old)==1024
    for v in old:
        hits=set(UUID.findall(json.dumps(v)))&ids;duplicate=v['sha256'] in hashes
        if hits or duplicate:removed.append(dict(v,matched_identities=sorted(hits),matched_content_hash=duplicate))
        else:keep.append(v)
    assert len(removed)==4 and len(keep)==1020
    dump(ROOT/'audits/cleaned_original_pool.json',dict(status='PASS',before=1024,after=1020,removed_count=4,excluded_identities=sorted(ids),excluded_source_SHA256=sorted(hashes),removed=removed,source_manifest=str(V17/'train_manifest_ulong.json'),source_manifest_sha256=sha(V17/'train_manifest_ulong.json'),conflict_evidence_sha256=sha(ROOT/'audits/original_pool_historical_test_conflicts.json'),historical_manifest_unchanged=True,cleaned_videos=keep))
    return keep
def exposure():
    evidence=load(ROOT/'audits/original_pool_historical_test_conflicts.json');four={r['identity'] for r in evidence['conflicts']};evalvideos=[dict(v,dataset=d) for d in DATASETS for v in sources(d)];targets=four|{v['name'] for v in evalvideos};hits=[];stages=[]
    v3=ROOT.parent/'gvcrt_neural_wrapper_v3_pb_scale';v41=ROOT.parent/'gvcrt_neural_wrapper_v4_1_high_convergence_metrics'
    stage_specs=[
        (V17,'dists_w05',1000,V17/'checkpoints/adaptation_step_1000.pt',V17/'training_logs/dists_w05.jsonl',V17/'train_manifest_ulong.json','adaptation_step'),
        (V62,'schedule_s1p0',1000,V62/'branches/schedule_s1p0/checkpoints/step_1000.pt',V62/'branches/schedule_s1p0/training_log.jsonl',ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/train_manifest_1024.json','update'),
        (v41,'beta_high_extension',20000,v41/'checkpoints/beta_high/step_20000.pt',v41/'training_logs/training_beta_high_extension.csv',v41/'train_manifest.json','step'),
        (V4,'beta_high',10000,V4/'checkpoints/beta_high/step_10000.pt',V4/'training_logs/training_beta_high.csv',V4/'train_manifest.json','step'),
        (v3,'stage_b',20000,v3/'checkpoints/stage_b/step_20000.pt',v3/'training_log.csv',v3/'train_manifest.json','step'),
        (v3,'stage_a',5000,v3/'checkpoints/stage_a/step_5000.pt',v3/'training_log.csv',v3/'train_manifest.json','step')]
    for folder,stage,limit,cp,log,manifest,stepkey in stage_specs:
        values=([json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.suffix=='.jsonl' else read(log)) if log.exists() else []
        rows=[r for r in values if int(r.get(stepkey,r.get('update',0)))<=limit and (not r.get('stage') or r['stage']==stage)]
        pool=json.dumps(load(manifest)) if manifest.exists() else '';matched_pool={t for t in targets if t in pool};stepvalues=[int(r.get(stepkey,r.get('update',0))) for r in rows]
        expected=list(range(10001,20001)) if stage=='beta_high_extension' else list(range(1,limit+1));complete=stepvalues==expected
        stages.append(dict(stage=stage,checkpoint=str(cp),checkpoint_sha256=sha(cp) if cp.exists() else None,log_path=str(log),log_sha256=sha(log) if log.exists() else None,training_pool_manifest=str(manifest),training_pool_sha256=sha(manifest) if manifest.exists() else None,limit_step=limit,rows_available=len(rows),log_contiguous_complete=complete,identities_present_in_pool=sorted(matched_pool)))
        for r in rows:
            video=str(r.get('video',r.get('filename','')))
            for identity in targets:
                if identity not in video:continue
                indices=r.get('source_frame_indices')
                basis='logged source_frame_indices'
                if indices is None and 'start' in r:
                    indices=list(range(int(r['start']),int(r['start'])+4));basis='derived from logged native start and historical 4-frame sequential loader; not a separately logged frame hash'
                hits.append(dict(identity=identity,stage=stage,checkpoint=str(cp),update=int(r.get(stepkey,r.get('update',0))),source_frame_indices=indices,frame_index_evidence=basis if indices is not None else 'not available',video=video,log_path=str(log),log_sha256=sha(log),external_qp=r.get('external_qp',r.get('qp')),crop_x=r.get('crop_x'),crop_y=r.get('crop_y')))
    links=[]
    for child,config,key,parent in [
        (stage_specs[0][3],V17/'config.json','source_checkpoint',stage_specs[1][3]),
        (stage_specs[1][3],V62/'fullqp_config.json','source_checkpoint',stage_specs[2][3]),
        (stage_specs[2][3],v41/'config.json','source_checkpoint',stage_specs[3][3]),
        (stage_specs[3][3],V4/'config.json','initial_checkpoint',stage_specs[4][3])]:
        cfg=load(config);item=cfg[key];source=item['path'] if isinstance(item,dict) else item
        assert Path(source)==parent,(child,source,parent)
        actual=sha(parent);expected=item.get('sha256') if isinstance(item,dict) else cfg.get(key+'_sha256')
        if expected:assert actual==expected,('source chain hash mismatch',parent)
        links.append(dict(checkpoint=str(child),source_checkpoint=str(parent),source_checkpoint_sha256=actual,expected_source_sha256=expected,evidence_config=str(config),evidence_config_sha256=sha(config),evidence_key=key,verified=True))
    source_code=(v3/'train.py').read_text();assert 'checkpoints/stage_a/step_5000.pt' in source_code and 'elif args.stage == "stage_b"' in source_code
    links.append(dict(checkpoint=str(stage_specs[4][3]),source_checkpoint=str(stage_specs[5][3]),source_checkpoint_sha256=sha(stage_specs[5][3]),evidence_code=str(v3/'train.py'),evidence_code_sha256=sha(v3/'train.py'),verified=True))
    records=[]
    for identity in sorted(targets):
        actual=[r for r in hits if r['identity']==identity];in_pool=[s['stage'] for s in stages if identity in s['identities_present_in_pool']];is_eval=[dict(dataset=v['dataset'],name=v['name'],video_index=v['video_index'],source_path=v['source_path'],source_sha256=v['source_sha256']) for v in evalvideos if v['name']==identity or identity in v['source_path']]
        state='log_confirmed_sampled' if actual else 'pool_present_actual_sampling_unknown' if in_pool else 'not_sampled_in_available_complete_adaptation_logs' if all(s['log_contiguous_complete'] for s in stages) else 'insufficient_evidence'
        records.append(dict(identity=identity,is_one_of_four_removed=identity in four,is_in_current_16_evaluation_videos=bool(is_eval),evaluation_matches=is_eval,in_training_pool_stages=in_pool,adaptation_exposure_status=state,confirmed_sampling=actual,source_native_pretraining_exposure='unknown; native codec/generator pretraining source logs unavailable',overall_source_exposure_status='confirmed' if actual else 'unknown',independent_unseen_generalization_claim=False))
    dump(ROOT/'audits/historical_exposure.json',dict(status='PASS',scope='factual available logs on actual source chain; absence of evidence is not unexposed',source_checkpoint=str(V17/'checkpoints/adaptation_step_1000.pt'),source_sha256=EXPECTED,source_chain_stages=stages,verified_source_links=links,records=records,removed_four=sorted(four),evaluation_video_count=16,checkpoint_not_retrained=True))
    write(ROOT/'audits/historical_exposure_records.csv',records)
    if hits:write(ROOT/'audits/historical_exposure_sampling_hits.csv',hits)
    return {v['name']:next(r for r in records if r['identity']==v['name']) for v in evalvideos}
if __name__=='__main__':archive_previous();clean_original()
