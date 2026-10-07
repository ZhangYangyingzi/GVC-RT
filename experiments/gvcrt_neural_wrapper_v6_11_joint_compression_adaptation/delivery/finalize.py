"""Package small raw evidence after PASS; keep large artifacts server-local."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v68_io import *

def main():
    final=load(ROOT/'final_integrity.json');assert final['status']=='PASS'
    assert load(ROOT/'pipeline_status.json')['status']=='PASS'
    frozen(True)
    for name,h in final['output_hashes'].items():assert sha(ROOT/name)==h
    copied=[]
    points=[point(d,m,v['video_index'],q) for d in DATASETS for m in METHODS for v in sources(d) for q in range(10)]
    held=load(ROOT/'audits/vimeo_heldout_manifest.json')
    points.extend(ROOT/'parts/heldout'/m/f'video_{v["index"]:03d}_qp{q}.json' for m in METHODS for v in held['clips'] for q in held['external_qps'])
    assert len(points)==2436
    for p in points:
        r=load(p)
        for key,suffix in [('frame_metrics','.frames.csv'),('transitions','.transitions.csv')]:
            source=Path(r[key+'_path']);assert sha(source)==r[key+'_sha256']
            target=p.with_suffix(suffix)
            if target.exists():assert sha(target)==r[key+'_sha256']
            else:target.write_bytes(source.read_bytes())
            assert sha(target)==r[key+'_sha256']
            copied.append(dict(path=str(target.relative_to(ROOT)),source=str(source),sha256=r[key+'_sha256']))
    dump(ROOT/'audits/raw_table_delivery.json',dict(status='PASS',point_records=len(points),per_frame_and_transition_tables=len(copied),tables=copied))
    # Cache-adjacent transition tables have canonical uploadable copies above.
    local_cache_dirs={'features','heldout_features','visualizations','bitstreams','heldout_bitstreams'}
    paths=[p for p in ROOT.rglob('*') if p.is_file() and not (set(p.relative_to(ROOT).parts)&local_cache_dirs) and (p.suffix in ('.py','.csv','.json','.jsonl','.log') or p.name=='.gitignore')]
    result=subprocess.run(['git','check-ignore','--stdin'],cwd=REPO,input='\n'.join(str(p.relative_to(REPO)) for p in paths)+'\n',capture_output=True,text=True)
    assert result.returncode in (0,1)
    ignored=result.stdout.splitlines();assert not ignored,ignored
    dump(ROOT/'audits/git_upload_audit.json',dict(status='PASS',checked=len(paths),ignored=ignored,weights_bitstreams_and_large_features_local=True))
    revision=load(ROOT/'audits/evaluation_adapter_revision.json');assert revision['status']=='PASS_NATIVE_RECHECK'
    assert load(ROOT/'audits/entropy_mode_native_recheck.json')['status']=='PASS'
    final.update(delivery_status='PASS',missing_files=[],unresolved_failures=[],
        recovered_failure_records=[str(p.relative_to(ROOT)) for p in (ROOT/'logs/failures').glob('*.json')],
        delivery_script_sha256=sha(Path(__file__)),git_upload_audit_sha256=sha(ROOT/'audits/git_upload_audit.json'),
        raw_table_delivery_sha256=sha(ROOT/'audits/raw_table_delivery.json'),training_updates=dict(F_frozen_core_reused=1000,J_joint_core_new=1000),delivery_finished_unix=time.time())
    dump(ROOT/'final_integrity.json',final)
    dump(ROOT/'generated_files.json',dict(files=[str(p.relative_to(ROOT)) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]))
    print(json.dumps(dict(status='PASS',counts=final['counts'],missing_files=[],unresolved_failures=[],result_directory=str(ROOT/'results'))),flush=True)
if __name__=='__main__':main()
