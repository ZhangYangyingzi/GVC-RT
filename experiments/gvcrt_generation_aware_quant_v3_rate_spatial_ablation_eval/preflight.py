"""Fail-closed checks required before the v3 task queue starts."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

P=Path(__file__).resolve().parent
V2=P.parent/'gvcrt_generation_aware_quant_v2_skip_domain_trend_300'
V1=P.parent/'gvcrt_generation_aware_quant_visual_condition_v1_trend_300'
sys.path.insert(0,str(V2))
import task


def check():
    import train
    train.setup_path()
    import torch
    import ablation
    tolerances={'native_stream_sha':0,'identity_native_stream_sha':0,
                'reused_frame_LPIPS':1e-7,'reused_frame_DISTS':1e-7,
                'reused_frame_PSNR':1e-6,'constant_multiplier':0,
                'mean_definition':1e-7,'rho_post_clamp':1e-7}
    checks={}
    for mode in ('A','MAP'):
        row=json.loads((P/'server_artifacts/evaluation'/f'{task.key(mode,"UVal0",2,2)}.json').read_text())
        old=task.old_record(mode,'UVal0',2)
        assert row['status']=='PASS' and old['status']=='PASS'
        assert row['native_stream_sha256']==old['stream_sha256']
        assert row['checkpoint_sha256']==old['checkpoint_sha256']
        assert row['all_64_reference_states_synchronized']
        old_branch='A' if mode=='A' else 'H11'
        old_csv=(V2/'server_artifacts/evaluation_frames'/f'{old_branch}_s300_UVal0_q2.csv')
        if not old_csv.exists() and mode=='A':
            old_csv=V1/'server_artifacts/evaluation_frames/A_s300_UVal0_q2.csv'
        fresh_csv=P/'server_artifacts/frames'/f'{task.key(mode,"UVal0",2,2)}.csv'
        old_rows=list(csv.DictReader(old_csv.open()));fresh_rows=list(csv.DictReader(fresh_csv.open()))
        assert len(old_rows)==len(fresh_rows)==64
        errors={k:max(abs(float(a[k])-float(b[k])) for a,b in zip(old_rows,fresh_rows))
                for k in ('LPIPS','DISTS','pixel_SSE')}
        assert errors['LPIPS']<=tolerances['reused_frame_LPIPS']
        assert errors['DISTS']<=tolerances['reused_frame_DISTS']
        assert errors['pixel_SSE']<=tolerances['reused_frame_PSNR']
        checks[mode+'_v2_parity']={'native_sha_equal':True,'frame_max_abs_error':errors}
    a=json.loads((P/'server_artifacts/evaluation'/f'{task.key("A","UVal0",2,2)}.json').read_text())
    identity=json.loads((P/'server_artifacts/evaluation'/f'{task.key("IDENTITY","UVal0",2,2)}.json').read_text())
    assert identity['native_stream_sha256']==a['native_stream_sha256']
    assert identity['checkpoint_sha256']!=a['checkpoint_sha256']
    assert identity['all_64_reference_states_synchronized'] and identity['all_P_latents_qmaps_exact']
    checks['identity_A_native_stream']={'sha_equal':True,'checkpoint_sha_distinct':True,
        'encoder_decoder_64_references':True,'all_P_latents_qmaps_exact':True}
    identity_state=json.loads((P/'reference_identity_audit.json').read_text())
    assert identity_state['status']=='PASS' and identity_state['frames']==64
    assert identity_state['reference_sha_equal_all'] and identity_state['P_latent_sha_equal_all'] and identity_state['P_qmap_sha_equal_all']
    checks['identity_A_decoded_state']=identity_state
    raw=torch.full((1,1,12,16),1.2)
    transformed=[ablation.transform_multiplier(raw,mode,2)[1] for mode in ('MAP','MEAN','ROLL')]
    assert all(torch.equal(transformed[0],x) for x in transformed[1:])
    checks['constant_multiplier_modes']={'max_error':max(float((transformed[0]-x).abs().max()) for x in transformed[1:])}
    raw=torch.arange(12*16,dtype=torch.float32).reshape(1,1,12,16)/200+.5
    rolled=ablation.transform_multiplier(raw,'ROLL',2)[0]
    assert torch.equal(torch.sort(raw.flatten()).values,torch.sort(rolled.flatten()).values)
    mean=ablation.transform_multiplier(raw,'MEAN',2)[0]
    mean_error=float((mean-raw.mean(dim=(-2,-1),keepdim=True)).abs().max())
    assert mean_error<=tolerances['mean_definition']
    checks['roll_and_mean']={'roll_multiset_exact':True,'mean_error':mean_error,'roll_blocks':[1,2]}
    # Post-clamp and dtype are part of the actual coordinate transform.
    base=torch.tensor([.5,.7,1.],dtype=torch.float16)
    m=torch.tensor([.5,1.25,2.],dtype=torch.float16)
    q=(base.float()*m).clamp_min(.5).to(base.dtype)
    rho=q/base
    assert q.tolist()==[.5,.875,2.0]
    assert rho.tolist()==[1.0,1.25,2.0]
    checks['actual_rho_after_clamp_dtype']={'q':q.tolist(),'rho':rho.tolist(),
        'first_differs_from_raw_multiplier':bool(rho[0]!=m[0])}
    for mode in ('MEAN','ROLL'):
        path=P/'server_artifacts/rate_only'/f'{task.key(mode,"UVal1",2,0)}.json'
        row=json.loads(path.read_text())
        assert row['status']=='PASS' and row['alpha_index']==0
        assert row['all_64_reference_states_synchronized'] and row['all_P_latents_qmaps_exact']
        assert row['imported_runtime_path']==str(V2/'runtime.py')
        checks[mode+'_nonunit_alpha']={'alpha':.8,'all_64_references':True,'all_P_latents_qmaps_exact':True,
                                       'stream_sha256':row['stream_sha256']}
    for f in ('task.py','ablation.py'):
        checks[f+'_sha256']=task.sha(P/f)
    checks['v2_runtime_sha256']=task.sha(V2/'runtime.py')
    checks['v2_sources_sha256']=task.sha(V2/'evaluation_sources.json')
    checks['weights_sha256']={'A':task.sha(V1/'server_artifacts/A/step_300.pt'),
                             'H11':task.sha(V2/'server_artifacts/H11/step_300.pt')}
    result={'status':'PASS','tolerances':tolerances,'checks':checks}
    (P/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PREFLIGHT_PASS',flush=True)

if __name__=='__main__':check()
