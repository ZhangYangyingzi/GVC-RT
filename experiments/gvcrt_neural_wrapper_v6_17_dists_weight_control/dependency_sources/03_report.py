"""Reuse frozen V6.15 numerical tables, restricted to four fixed models."""
import ast,math,statistics,traceback
from io16 import *
PAIRS=tuple((m,b) for i,m in enumerate(METHODS) for b in METHODS[:i])
def numerical_code():
    p=V15/'report.py';assert sha(p)==load(ROOT/'audits/dependencies.json')[str(p)]
    src=p.read_text();tree=ast.parse(src)
    compare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='comparisons')
    text=ast.get_source_segment(src,compare).replace('expected_models=6','expected_models=4').replace('available_models=6','available_models=4').replace('six-model','four-model')
    # Keep raw tables, common-rate interpolation and pooled FID arithmetic intact.
    body=ast.get_source_segment(src,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
    body=body.split('    validation_counts={}')[0].replace('len(vs)*60','len(vs)*40').replace('expected_pooled_FID=60','expected_pooled_FID=40').replace('six-model','four-model')
    ns=dict(globals());exec(compile(text+'\n'+body,str(p),'exec'),ns)
    return ns['main']
def integrity():
    import numpy as np
    frozen();assert load(ROOT/'gradient_audit.json')['status']=='PASS' and load(ROOT/'gradient_audit.json')['gate']=='FIX_REQUIRED'
    assert load(ROOT/'training_integrity.json')['status']=='PASS'
    rows=[json.loads(x) for x in (ROOT/'training_logs/dists_fixed.jsonl').read_text().splitlines()];assert len(rows)==1000
    keys=('domain','video','source_frame_indices','canonical_indices','crop_x','crop_y','crop_size','cropped_RGB_sha256','external_qp','actual_qps')
    assert all(all(a[k]==b[k] for k in keys) for a,b in zip(replay_rows(),rows))
    assert all(r['input_crop_hashes_match'] and r['actual_qps_match'] and r['all_finite'] for r in rows)
    index=load(ROOT/'checkpoint_index.json');assert sorted(map(int,index))==[0,250,500,1000]
    for cp in index.values():assert sha(cp['path'])==cp['sha256'] and cp['compression_hash']==evalcfg()['compression_hash']
    from adapter import validate
    expected=completed=reused=fresh=0;counts={}
    for d in DATASETS:
        vs=sources(d);n=len(vs)*40;expected+=n
        previous_gt=V15/'features'/f'GT_{d}.npy'
        if previous_gt.exists():
            assert np.array_equal(np.load(previous_gt),np.load(ROOT/'features'/f'GT_{d}.npy')),('GT pool changed',d)
        for v in vs:
            for m in METHODS:
                for q in range(10):
                    r=load(point(d,m,v['video_index'],q));validate(r,v,q,m);completed+=1;reused+=bool(r.get('reused'));fresh+=not bool(r.get('reused'))
        assert len(read(ROOT/'results'/d/'per_sequence_all_qp.csv'))==n
        assert len(read(ROOT/'results'/d/'fid_pooled.csv'))==40
        counts[d]=dict(expected_RD=n,completed_RD=n,pooled_FID=40,frames_per_point=64,transitions_per_point=63)
    assert expected==completed==640
    dump(ROOT/'audits/fixed_GT_pool_reuse.json',dict(status='PASS',datasets={d:dict(path=str(V15/'features'/f'GT_{d}.npy'),sha256=sha(V15/'features'/f'GT_{d}.npy'),array_identical=True) for d in DATASETS if (V15/'features'/f'GT_{d}.npy').exists()},basis='same frozen manifests and original feature arrays; new GT arrays checked identical to existing caches when present'))
    plots=list((ROOT/'results').glob('*/Rate_*.png'));assert len(plots)==16
    old=load(ROOT/'audits/historical_inventory.json');now=historical_inventory();changed=[p for p,s in old.items() if now.get(p)!=s]
    assert not changed,('historical files changed',changed)
    outputs={str(p.relative_to(ROOT)):sha(p) for folder in ('results','probes','training_logs','dependency_sources') for p in (ROOT/folder).rglob('*') if p.is_file()}
    dump(ROOT/'final_integrity.json',dict(status='PASS',gradient_verification='PASS',repair_triggered=True,training_updates=1000,source_step=1000,expected_RD=expected,completed_RD=completed,reused_baseline_points=reused,fresh_real_RANS_points=fresh,counts=counts,expected_RD_PNG=16,completed_RD_PNG=16,missing=[],failed=[],checkpoint_steps=[0,250,500,1000],checks=dict(real_RANS=True,independent_decode=True,source_frames_verified=True,exact_1000_sample_replay=True,actual_QP_replay=True,quality_frozen=True,frozen_compression_unchanged=True,historical_files_unchanged=True,no_checkpoint_selection=True),source_hashes=load(ROOT/'audits/dependencies.json'),output_hashes=outputs,finished_unix=time.time()))
    print('FINAL PASS 640/640',flush=True)
def main():numerical_code()();integrity()
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'final_integrity.json',dict(status='FAIL',error=traceback.format_exc()));raise
