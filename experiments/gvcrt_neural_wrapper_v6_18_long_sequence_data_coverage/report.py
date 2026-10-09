"""Five fixed models; numerical comparisons of weight 0.5 to four baselines."""
import ast,math,statistics,traceback
from io18 import *
PAIRS=(('B_1000','A_1000'),('C_1000','B_1000'),*((m,b) for m in METHODS[2:] for b in METHODS[:2]))
def numerical_code():
    p=V15/'report.py';assert sha(p)==load(ROOT/'audits/dependencies.json')[str(p)]
    src=p.read_text();tree=ast.parse(src)
    compare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='comparisons')
    text=ast.get_source_segment(src,compare).replace('expected_models=6','expected_models=5').replace('available_models=6','available_models=5').replace('six-model','five-model')
    # Keep raw tables, common-rate interpolation and pooled FID arithmetic intact.
    body=ast.get_source_segment(src,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
    body=body.split('    validation_counts={}')[0].replace('len(vs)*60','len(vs)*50').replace('expected_pooled_FID=60','expected_pooled_FID=50').replace('six-model','five-model')
    body=body.replace("dict(r,actual_bpp=", "dict(r,**exposure_fields(v),actual_bpp=")
    body=body.replace("read(r['frame_metrics_path'])", "[dict(x,**exposure_fields(v)) for x in read(r['frame_metrics_path'])]")
    body=body.replace("read(r['transitions_path'])", "[dict(x,**exposure_fields(v)) for x in read(r['transitions_path'])]")
    body=body.replace("aggregation='pooled all dataset frame features; not mean per-video FID'", "aggregation='pooled all dataset frame features; not mean per-video FID',source_exposure_status='see per-video historical_exposure; unknown or confirmed',independent_unseen_generalization_claim=False")
    body=body.replace("aggregation='equal video mean metrics; pooled FID'", "aggregation='equal video mean metrics; pooled FID',source_exposure_status='see per-video historical_exposure; unknown or confirmed',independent_unseen_generalization_claim=False")
    text=text.replace("expected_models=5)", "expected_models=5,source_exposure_status='see historical_exposure; unknown or confirmed',independent_unseen_generalization_claim=False)")
    body=body.replace("delta_definition='method-reference; negative better'))", "delta_definition='method-reference; negative better',source_exposure_status='see historical_exposure; unknown or confirmed',independent_unseen_generalization_claim=False))")
    ns=dict(globals());exec(compile(text+'\n'+body,str(p),'exec'),ns)
    return ns['main']
def exposure_fields(v):
    r=next(r for r in load(ROOT/'audits/historical_exposure.json')['records'] if r['identity']==v['name'])
    return dict(source_adaptation_exposure_status=r['adaptation_exposure_status'],source_overall_exposure_status=r['overall_source_exposure_status'],source_pretraining_exposure=r['source_native_pretraining_exposure'],new_training_exposure='held_out_from_all_v618_training',independent_unseen_generalization_claim=False,historical_exposure_audit_sha256=sha(ROOT/'audits/historical_exposure.json'))
def integrity():
    import numpy as np
    from adapter import validate
    from window_training import replay_rows
    frozen();assert load(ROOT/'audits/implementation_check.json')['status']=='PASS';assert load(ROOT/'audits/step0_codec_equivalence.json')['status']=='PASS'
    branch_counts={};indices={}
    keys=('domain','video','source_frame_indices','canonical_indices','crop_x','crop_y','cropped_RGB_sha256','external_qp','actual_qps')
    for b in ('A','B','C'):
        folder=ROOT/'branches'/b;assert load(folder/'training_integrity.json')['status']=='PASS';rows=[json.loads(x) for x in (folder/'training_logs/train.jsonl').read_text().splitlines()];plan=replay_rows(b)
        assert len(rows)==1000 and [r['adaptation_step'] for r in rows]==list(range(1,1001))
        assert all(all(r[k]==p[k] for k in keys) for r,p in zip(rows,plan))
        assert sum(r['supervision_frames'] for r in rows)==15000 and all(r['normalization_denominator']==15 and len(r['frame_trace'])==15 for r in rows)
        assert all(r['all_finite'] and r['input_crop_hashes_match'] and r['actual_qps_match'] for r in rows)
        assert all(math.isclose(r['weighted_DISTS'],.5*r['DISTS'],rel_tol=1e-12) and 0<r['global_clip_coefficient']<=1 for r in rows)
        index=load(folder/'checkpoint_index.json');assert sorted(map(int,index))==[0,250,500,1000]
        for cp in index.values():assert sha(cp['path'])==cp['sha256'] and sha(cp['inference']['path'])==cp['inference']['sha256'] and cp['compression_hash']==evalcfg()['compression_hash']
        assert index['0']['module_hashes']==load(ROOT/'config.json')['source_checkpoint']['module_hashes'];indices[b]=index
        branch_counts[b]=dict(updates=1000,supervised_P_frames=15000,domain_updates={d:sum(r['domain']==d for r in rows) for d in ('ulong','uvg')},actual_QP_exact=True)
    assert replay_rows('A')==replay_rows('B');assert all((a['domain'],a['external_qp'])==(c['domain'],c['external_qp']) for a,c in zip(replay_rows('A'),replay_rows('C')))
    expected=completed=reused=fresh=0;counts={}
    for d in DATASETS:
        vs=sources(d);n=len(vs)*50;expected+=n
        assert np.array_equal(np.load(V17/'features'/f'GT_{d}.npy'),np.load(ROOT/'features'/f'GT_{d}.npy'))
        for v in vs:
            for m in METHODS:
                for q in range(10):
                    r=load(point(d,m,v['video_index'],q));validate(r,v,q,m);completed+=1;reused+=bool(r.get('reused'));fresh+=not bool(r.get('reused'))
        raw=read(ROOT/'results'/d/'per_sequence_all_qp.csv');assert len(raw)==n and all(r['independent_unseen_generalization_claim']=='False' for r in raw)
        assert len(read(ROOT/'results'/d/'fid_pooled.csv'))==50
        counts[d]=dict(expected_RD=n,completed_RD=n,pooled_FID=50,frames_per_point=64,transitions_per_point=63,GT_frames=len(vs)*64)
        for table in ('equal_rate_per_sequence','equal_rate_summary','equal_rate_fid_pooled'):
            entries=read(ROOT/'results'/d/(table+'.csv'));assert {(r['method'],r['reference']) for r in entries}==set(PAIRS)
    assert expected==completed==800 and fresh>=480 and reused+fresh==800
    assert len(list((ROOT/'results').glob('*/Rate_*.png')))==16
    previous=load(ROOT/'audits/historical_inventory.json');now=historical_inventory();changed=[p for p,stamp in previous.items() if now.get(p)!=stamp];assert not changed,('historical files changed',changed)
    dump(ROOT/'checkpoint_index.json',indices);dump(ROOT/'frozen_checkpoint_index.json',dict(status='PASS',branches=indices,fixed_evaluation_step=1000,no_selection=True))
    outputs={str(p.relative_to(ROOT)):sha(p) for folder in ('results','audits','dependency_sources') for p in (ROOT/folder).rglob('*') if p.is_file()}
    dump(ROOT/'final_integrity.json',dict(status='PASS',branches=branch_counts,expected_RD=800,completed_RD=800,reused_baseline_points=reused,fresh_real_RANS_points=fresh,counts=counts,completed_RD_PNG=16,missing=[],failed=[],source_checkpoint_sha256=EXPECTED,cleaned_original_ULong_videos=1020,cleaning_audit=load(ROOT/'audits/cleaned_original_pool.json'),historical_exposure_audit_sha256=sha(ROOT/'audits/historical_exposure.json'),checks=dict(real_RANS=True,independent_decode=True,source_frame_hashes=True,frozen_compression_unchanged=True,quality_frozen_eval=True,supervision_budget_matched=True,AB_exact_samples=True,global_window_QP=True,all_historical_files_read_only=True,new_training_source_disjoint=True,no_checkpoint_selection=True),output_hashes=outputs,finished_unix=time.time()))
    print('FINAL PASS 800/800',flush=True)
def main():numerical_code()();integrity()
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'final_integrity.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
