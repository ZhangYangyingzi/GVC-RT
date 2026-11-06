"""Copy only the actual audited runtime sources and immutable data descriptions."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time
E=Path(__file__).resolve().parent
PROJECT=Path('/Huang_group/zyyz/Projects/GVC-RT')
OLD=PROJECT/'GVC-RT_worktrees/tcg_stage2_router_only_20261105/experiments/gvcrt_tcg_stage2_router_only_20261105'
BRANCH='codex/tcg-stage2-history-gate-joint-20261106'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    E.mkdir(parents=True,exist_ok=True)
    identity=E/'task_identity.json'
    wanted=dict(task='tcg_stage2_history_gate_joint_20261106',seed=20261106,baseline_cumulative_tcg_step=6500,
        baseline_sha256='e51f8a7353be85ffd47e3a6d76ce2e478f30a66ebc57778638ad5efe1c876260',branch=BRANCH)
    if identity.exists():
        assert json.loads(identity.read_text())==wanted,'OTHER_TASK_AT_REQUESTED_PATH'
        if (E/'audits/source_copy.json').exists():print('SAME_TASK_ALREADY_BOOTSTRAPPED');return
    else:identity.write_text(json.dumps(wanted,indent=2)+'\n')
    for n in ['audits','logs','manifests','plans','vendor/sealed','cache','tmp']:(E/n).mkdir(parents=True,exist_ok=True)
    records=[]
    def copy(source,dest):
        dest=E/dest;dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists(),('REFUSE_OVERWRITE',str(dest))
        shutil.copyfile(source,dest)
        h=sha(source);assert sha(dest)==h
        records.append(dict(source_path=str(source),destination=str(dest.relative_to(E)),sha256=h,size_bytes=source.stat().st_size))
    for tree in ['runtime_tree','reference_tree']:
        for p in sorted((OLD/tree).rglob('*.py')):copy(p,p.relative_to(OLD))
        d=E/tree/'checkpoints';d.mkdir(exist_ok=True)
        for n in ['GVC-RT_I.pt','GVC-RT_P.pt']:(d/n).symlink_to(PROJECT/'checkpoints'/n)
    for name in ['common_s2.py','model_context.py','loss.py','data_reader.py','pipeline.py','eval_data.py','evaluate.py','train.py','checkpoint_state.py','runner.py','finalize.py','integrity.py','publish.py','reuse.py']:
        copy(OLD/name,Path(name))
    for p in sorted((OLD/'vendor/sealed').rglob('*')):
        if p.is_file() and p.suffix in ('.py','.json','.gz'):copy(p,p.relative_to(OLD))
    for p in sorted((OLD/'plans').glob('*')):
        if p.is_file() and p.suffix in ('.json','.gz'):copy(p,p.relative_to(OLD))
    copy(OLD/'manifests/final_videos.json','manifests/final_videos.json')
    for name in ['protocol.json','source_manifest.json','final_integrity.json','baseline_provenance.json','audits/data_plan.json','audits/load_probe.json','audits/reuse_summary.json','audits/runtime_dependency_versions.json']:
        copy(OLD/name,Path('vendor/previous_stage2')/name)
    (E/'audits/source_copy.json').write_text(json.dumps(dict(status='PASS',copied_files=records,
        previous_actual_runtime=str(OLD),previous_worktree_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=OLD.parents[1],text=True).strip(),
        previous_source_manifest_sha256=sha(OLD/'source_manifest.json'),no_checkpoint_or_cache_copied=True,time=time.time()),indent=2)+'\n')
    (E/'run_status.json').write_text(json.dumps(dict(status='RUNNING',current_stage='implementation',committed_joint_updates=0,time=time.time()),indent=2)+'\n')
    print('BOOTSTRAP_PASS',len(records))
if __name__=='__main__':main()
