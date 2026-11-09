"""Freeze a source-isolated plan using inherited metadata; no dataset rescan."""
from common_s2 import *
import collections

def main():
    env()
    if (ROOT/'plans/index.json').exists():return
    assert sha(B300)==B300_SHA
    pool=load(ROOT/'manifests/effective_ULong_pool.json')
    isolation=load(PARENT_EXPERIMENT/'audits/source_isolation.json');assert isolation['status']=='PASS'
    old=[]
    for sh in load(PARENT_EXPERIMENT/'plans/index.json')['shards']:old+=load(PARENT_EXPERIMENT/sh['file'])
    used={r['source_group'] for r in old}
    groups=collections.defaultdict(list)
    for r in pool:groups[r['source_group']].append(r)
    rng=random.Random(2026110905)
    unused=sorted(set(groups)-used);previous=sorted(set(groups)&used);rng.shuffle(unused);rng.shuffle(previous)
    order=(unused+previous)[:1200];assert len(order)==len(set(order))==1200
    qpmap={r['external_qp']:r['actual_frame_qps'] for r in old};qps=[]
    for _ in range(120):b=list(range(10));rng.shuffle(b);qps+=b
    plan=[]
    for slot,g in enumerate(order):
        r=rng.choice(sorted(groups[g],key=lambda x:x['identity']));n=r['frames']-63;start=rng.randrange(n)
        row=dict(domain='ulong',logical_sample=True,plan_seed=PLAN_SEED,plan_slot=slot,global_update=slot//4+1,logical_rank=slot%4,
            video=r['video'],identity=r['identity'],source_group=g,source_path=r['source_path'],source_sha256=r['source_sha256'],source_size_bytes=r['source_size_bytes'],source_total_frames=r['frames'],source_fps=r['source_fps'],width=r['width'],height=r['height'],source_frame_indices=list(range(start,start+64)),canonical_indices=list(range(start,start+64)),crop_x=rng.randrange(r['width']-255),crop_y=rng.randrange(r['height']-255),crop_width=256,crop_height=256,external_qp=qps[slot],actual_frame_qps=qpmap[qps[slot]],temporal_bin=0,temporal_bin_interval=[0,n])
        row['sample_key']=digest(row);plan.append(row)
    # The retained pool is already group-isolated against ALL held-out/quarantine sources.
    assert {p['row']['source_group'] for p in load(ROOT/'manifests/quick_development.json')}.isdisjoint(order)
    dump(ROOT/'plans/train.json.gz',plan)
    dump(ROOT/'plans/index.json',dict(status='FROZEN',rows=1200,global_updates_cap=300,effective_batch=4,seed=PLAN_SEED,shards=[dict(file='plans/train.json.gz',sha256=sha(ROOT/'plans/train.json.gz'),rows=1200)],QP_counts=dict(collections.Counter(qps)),new_source_groups=len(set(order)-used),parent_used_source_groups=len(used),preferred_unseen_source_groups=True))
    dump(ROOT/'audits/source_isolation.json',dict(isolation,plan_groups=1200,parent_audit_SHA256=sha(PARENT_EXPERIMENT/'audits/source_isolation.json'),pool_SHA256=sha(ROOT/'manifests/effective_ULong_pool.json'),metadata_only_no_rescan=True))
    dump(ROOT/'protocol.json',dict(task='gvcrt_tcg_flicker_ef_b300_20261109',source_commit=SOURCE_COMMIT,branch='codex/tcg-flicker-ef-b300-20261109',B300=dict(path=str(B300),SHA256=B300_SHA),inherited_protocol=dict(path=str(PARENT_EXPERIMENT/'protocol.json'),SHA256=sha(PARENT_EXPERIMENT/'protocol.json')),
        groups=dict(E=dict(GPUs=[0,1],detail=False),F=dict(GPUs=[2,3],detail=True)),training=dict(max_additional_updates=300,effective_batch=4,windows_per_rank=2,seed=PLAN_SEED,frames=64,crop=[256,256],supervised_frames=[49,63],segments=[[49,52],[53,56],[57,60],[61,63]],s_train=.5,s_eval=.5,optimizer='fresh AdamW',learning_rates=dict(Bridge=2e-6,Backbone=2e-6,TCG=1e-5,LoRA=5e-5),betas=[.9,.999],eps=1e-8,weight_decay=1e-4,global_clip_norm=1.,checkpoint_every=50),
        loss=dict(LPIPS=1.,DISTS=.5,RGB_L1=.1383589248954108,RAFT_motion=.05,lambda_temp=LAMBDA_TEMP,Sobel_MCResidual=.5,detail='mean of per-scale mean absolute RGB normalized replicate-Sobel differences; average pools 1,2,4',lambda_detail='0.1*median(full-trainables spatial-gradient-L2/detail-gradient-L2) on first8 immutable training windows, unupdated B300'),
        validation=dict(videos=12,QP=[0,5,9],frames=64,steps=[100,200,300],scope='ULong开发子集12视频',step0_check_profiles=[0,4],step0_points=6,stop='At BOTH100 and200: mean PtoP Sobel error >= B300 AND any QP mean LPIPS or DISTS >1.01*B300; otherwise300',final_test_points=0),codec='independent original fixed trajectory; no candidate reference writes',history_warp=False,router=False,plan_SHA256=sha(ROOT/'plans/index.json')))
    dump(ROOT/'scheduler_config.json',dict(startup_windows=16,calibration_windows=8,prefetch_windows=32,derived_byte_cap=160*2**30,reserve_bytes=100*2**30,gpu_cache_workers=1,cache_GPU=0,overlap_allowed_after_probe=True,delete_only_before_both_committed_cursors=True))
    dump(ROOT/'provenance.json',dict(source_commit=SOURCE_COMMIT,B300_SHA256=sha(B300),B300_path=str(B300),parent_protocol_SHA256=sha(PARENT_EXPERIMENT/'protocol.json'),parent_final_integrity=load(PARENT_EXPERIMENT/'final_integrity.json')['status'],time=time.time()))
    print('PLAN_FROZEN',len(plan),'unused_groups',len(set(order)-used),flush=True)
if __name__=='__main__':main()
