"""One train-only measurement of four wrapper gradients and their directions."""
import ast,math,random,statistics
from v67_io import *
from objective import bind
def run_calibration(v4,im,pm,wrapper,quality,cfg,models,initial,compression,optimizer,device):
    out=ROOT/'audits/guard_gradient_calibration.json'
    if out.exists():assert load(out)['status']=='PASS';return
    rows=[];cosines=[];rng=random.Random(cfg['sample_seed']);qps=load(ROOT/'qp_sequence_audit.json')['sequence'];refs=read(V66/'audits/structure_guard_calibration_samples.csv')
    fn=bind(v4,im,pm,wrapper,quality,cfg,calibration=True)
    for i in range(32):
        frames,plan=sample_clip(rng,device,v4);ref=refs[i]
        assert plan['sample_id']==ref['sample_id'] and plan['frame_sha256']==ast.literal_eval(ref['frame_sha256'])
        assert all(plan[k]==int(ref[k]) for k in ('start','crop_x','crop_y')) and qps[i]==int(ref['external_qp'])
        sums,actual=fn(frames,qps[i]);assert all(math.isfinite(z) for z in sums.values())
        assert math.isclose(sums['g_main'],float(ref['g_main']),rel_tol=1e-4,abs_tol=1e-7)
        assert math.isclose(sums['g_B'],float(ref['g_struct_unit']),rel_tol=1e-4,abs_tol=1e-7)
        assert math.isclose(sums['g_original_l1'],float(ref['g_l1']),rel_tol=1e-4,abs_tol=1e-7),('original L1 reference',sums['g_original_l1'],float(ref['g_l1']))
        sums['original_l1_vs_scaled_unit_norm_difference']=sums['g_original_l1']-.01*sums['g_D']
        rows.append(dict(calibration_index=i,external_qp=qps[i],actual_qps=actual,**plan,**sums))
        cosines.append(dict(calibration_index=i,sample_id=plan['sample_id'],start=plan['start'],crop_x=plan['crop_x'],crop_y=plan['crop_y'],external_qp=qps[i],**{k:z for k,z in sums.items() if k.startswith('cos_')}))
        print('CALIBRATION',i+1,'/32',flush=True)
    med={k:statistics.median(r['g_'+k] for r in rows) for k in ('main','B','D','E')};assert all(math.isfinite(z) and z>0 for z in med.values())
    old=load(V66/'audits/structure_guard_calibration.json');weights={'B':old['lambda_struct'],'D':.1*med['main']/med['D'],'E':.1*med['main']/med['E']}
    controls={}
    for k,w in weights.items():
        ratio=w*med[k]/med['main'];assert math.isfinite(w) and w>0 and .095<=ratio<=.105
        controls[k]={'lambda':w,'unit_grad':med[k],'weighted_grad':w*med[k],'weighted_over_main':ratio}
    # Verify the fixed weights by actual backward passes: half-precision STE can
    # round grad(lambda*L) differently from multiplying grad(L) afterwards.
    rng=random.Random(cfg['sample_seed']);verify=bind(v4,im,pm,wrapper,quality,cfg,calibration=True,gradient_weights=weights);verified=[]
    for i in range(32):
        frames,plan=sample_clip(rng,device,v4);sums,_=verify(frames,qps[i]);verified.append(dict(calibration_index=i,**{k:v for k,v in sums.items() if k.startswith('g_')}))
        print('WEIGHTED GRADIENT CHECK',i+1,'/32',flush=True)
    for k in ('B','D','E'):
        actual=statistics.median(r['g_'+k+'_weighted'] for r in verified);ratio=actual/med['main'];assert .095<=ratio<=.105,(k,ratio)
        controls[k].update(actual_backward_weighted_grad=actual,actual_backward_weighted_over_main=ratio)
    write(ROOT/'audits/weighted_guard_gradient_verification.csv',verified)
    assert initial=={k:v4.module_hash(m) for k,m in models.items()} and v4.compression_hash(im,pm)==compression and not optimizer.state
    write(ROOT/'audits/guard_gradient_calibration_samples.csv',rows);write(ROOT/'audits/guard_gradient_cosine.csv',cosines)
    summary={k:dict(mean=statistics.mean(r[k] for r in cosines),median=statistics.median(r[k] for r in cosines),std=statistics.pstdev(r[k] for r in cosines),n=32) for k in cosines[0] if k.startswith('cos_')}
    dump(ROOT/'audits/guard_gradient_cosine_summary.json',dict(status='PASS',statistics=summary,guard_gradients='unit guards; positive weights do not change cosine'))
    dump(out,dict(status='PASS',median_g_main=med['main'],**controls,train_only=True,clips=32,same_V66_calibration_plans=True,no_parameter_updates=True,source_checkpoint_sha256=cfg['source_checkpoint_sha256'],original_lambda_proxy=.01,original_L1_excluded_from_guard_budget=True,median_policy='Norm of summed per-P-frame gradients of the original three-frame-averaged objective; median over 32 clips',lambda_frozen=True))
    print('CALIBRATION PASS',flush=True)
