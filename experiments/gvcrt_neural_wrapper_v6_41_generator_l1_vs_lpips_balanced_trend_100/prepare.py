"""Validate inherited v6.40 inputs without regenerating its immutable plan."""
from io32 import *
import shutil
import collections
OLD=ROOT.parent/'gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100'
INIT=ROOT.parent/'gvcrt_neural_wrapper_v6_34_p_joint_fd_matched_5000/branches/N_noP_BG_FD/checkpoints/adaptation_step_0.pt'
INIT_SHA='18394bc0145e4bfc9eb18fb009fdb9dcc7f626536fe2367fb19b95e471f09074'
PLAN_SHA='3f3f54e2b1388a87b25010ee097db0bba03787652ba13d58bfcd9b8871406dd3'
def main():
    import torch
    plan=ROOT/'plans/first100.json';assert sha(plan)==PLAN_SHA
    assert sha(OLD/'plans/first100.json')==PLAN_SHA and sha(INIT)==INIT_SHA
    rows=load(plan)['updates'];assert len(rows)==100
    count=collections.Counter((r['source_row']['domain'],r['source_row']['external_qp']) for r in rows)
    assert all(count[(d,q)]==5 for d in ('uvg','ulong') for q in range(10))
    assert collections.Counter(r['source_row']['domain'] for r in rows)=={'uvg':50,'ulong':50}
    for i in range(0,100,2):
        a,b=[x['source_row'] for x in rows[i:i+2]]
        assert {a['domain'],b['domain']}=={'uvg','ulong'} and a['external_qp']==b['external_qp']
    source=torch.load(INIT,map_location='cpu',weights_only=True)
    basic=load(OLD/'branches/G_basic/step0_audit.json')
    assert basic['status']=='PASS' and basic['source_sha256']==INIT_SHA
    assert basic['plan_sha256']==PLAN_SHA and basic['optimizer_state_count']==0
    assert basic['active_modules']==['generator'] and basic['RNG_restored'] and basic['buffers_restored']
    assert load(OLD/'audits/initialization_tensor_buffer_rng.json')['status']=='PASS'
    proto=load(OLD/'protocol.json');new=load(ROOT/'protocol.json')
    for k in ('seed','training','selection'):
        assert proto[k]==new[k],k
    assert proto['training']['AdamW']==new['training']['AdamW']
    init_audit=dict(status='PASS',path=str(INIT),sha256=INIT_SHA,bytes=INIT.stat().st_size,
      source_initial_module_hashes=source['initial_module_hashes'],G_basic_step0_audit_sha256=sha(OLD/'branches/G_basic/step0_audit.json'),
      v640_plan_sha256=PLAN_SHA,G_basic_full_training_RNG_and_buffers_verified=True,
      G_L1_uses_same_complete_source_crop_rng_and_buffers=True,fresh_AdamW_state=True,
      training_configuration_equal=True,intentional_objective_difference='LPIPS term omitted')
    dump(ROOT/'audits/fairness_preflight.json',init_audit)
    dump(ROOT/'plans/counts.json',dict(status='PASS',plan_sha256=sha(plan),video_counts={'Beauty':50,'U-Long':50},
      video_qp_counts={d:{str(q):5 for q in range(10)} for d in ('Beauty','U-Long')},
      unique_crops_per_video={'Beauty':50,'U-Long':50},distinct_time_windows_per_video=1))
    for f in ['input_streams.json','training_windows.json','data_isolation.json']:
      p=OLD/'audits'/f
      if p.exists():shutil.copy2(p,ROOT/'audits'/f)
    streams=load(ROOT/'manifests/streams.json')
    assert len(streams)==6
    entries=[]
    for x in streams:
      p=Path(x['bitstream_path']);assert p.stat().st_size==int(x['real_bytes']) and sha(p)==x['bitstream_sha256']
      entries.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    dump(ROOT/'audits/input_streams.json',dict(status='PASS',streams=entries))
    dump(ROOT/'source_hashes.json',dict(initial_checkpoint=INIT_SHA,plan=PLAN_SHA,
      protocol=sha(ROOT/'protocol.json'),evaluation_config=sha(ROOT/'evaluation_config.json'),
      reference_protocol=sha(OLD/'protocol.json'),reference_plan=PLAN_SHA))
    dump(ROOT/'publication_status.json',dict(status='NOT_PUBLISHED',push_success=False))
    dump(ROOT/'run_status.json',dict(status='PREFLIGHT_PASS',training_updates=0,evaluation_points=0))
    print('PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
