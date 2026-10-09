"""Historical pooled FID and common-rate arithmetic; three fixed methods."""
import ast,math,statistics,traceback
from io19 import *
PAIRS=(('B_i_bypass','B_standard'),('B_standard','original'),('B_i_bypass','original'))
def comparisons(*args,**kw):
    src=(V15/'report.py').read_text();tree=ast.parse(src);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='comparisons');code=ast.get_source_segment(src,node).replace('expected_models=6','expected_models=3').replace('available_models=6','available_models=3').replace('six-model','three-model');ns=dict(globals());exec(code,ns);return ns['comparisons'](*args,**kw)
def numerical():
    src=(V15/'report.py').read_text();tree=ast.parse(src);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');code=ast.get_source_segment(src,node).split('    validation_counts={}')[0].replace("audits/step0_codec_equivalence.json","audits/implementation_check.json").replace('len(vs)*60','len(vs)*30').replace('expected_pooled_FID=60','expected_pooled_FID=30').replace('six-model','three-model')
    # Preserve source-model exposure facts from v6.18; no unseen-content claim.
    code=code.replace('dict(r,actual_bpp=', 'dict(r,source_overall_exposure_status="unknown; see v6.18 historical exposure audit",independent_unseen_generalization_claim=False,actual_bpp=')
    ns=dict(globals());exec(compile(code,str(ROOT/'report.py'),'exec'),ns);ns['main']()
def integrity():
    import numpy as np
    from adapter import validate
    frozen();assert load(ROOT/'audits/implementation_check.json')['status']=='PASS';count=reused=fresh=0;by={}
    for d in DATASETS:
        n=0
        for v in sources(d):
            for m in METHODS:
                for q in range(10):
                    r=load(point(d,m,v['video_index'],q));validate(r,v,q,m);count+=1;n+=1;reused+=bool(r.get('reused'));fresh+=not bool(r.get('reused'))
                    if m=='B_i_bypass':assert not r.get('reused')
        assert len(read(ROOT/'results'/d/'per_sequence_all_qp.csv'))==n;assert np.array_equal(np.load(ROOT/'features'/f'GT_{d}.npy'),np.load(V18/'features'/f'GT_{d}.npy'));by[d]=dict(expected=len(sources(d))*30,completed=n,FID_pool_frames=len(sources(d))*64)
    assert count==480 and fresh>=176 and fresh+reused==480
    old=load(ROOT/'audits/historical_inventory.json');now=historical_inventory();assert all(now.get(p)==stamp for p,stamp in old.items()),'historical files changed'
    assert len(list((ROOT/'results').glob('*/Rate_*.png')))==16
    assert len(list((ROOT/'plots/first_frames').glob('*.png')))==12
    paths={str(p.relative_to(ROOT)):sha(p) for folder in ('results','evaluation','plots','audits') for p in (ROOT/folder).rglob('*') if p.is_file() and 'checkpoints' not in p.parts}
    dump(ROOT/'final_integrity.json',dict(status='PASS',no_training=True,expected_points=480,completed_points=480,reused_points=reused,new_run_points=fresh,fresh_bypass_points=160,failed_points=0,missing=[],datasets=by,checks=dict(source_frames_consistent=True,source_checkpoint_verified=True,models_frozen=True,independent_decode=True,state_synchronization=True,full_64_frame_bit_accounting=True,cache_contains_I_mode=True,historical_files_unchanged=True,FID_fixed_dataset_pools=True,no_extrapolation=True),RD_plots=16,first_frame_panels=12,output_hashes=paths,finished_unix=time.time()));print('FINAL PASS 480/480',flush=True)
def main():
    numerical();from diagnostics import main as diagnostic;diagnostic();integrity()
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'final_integrity.json',dict(status='FAIL',error=traceback.format_exc()));raise
