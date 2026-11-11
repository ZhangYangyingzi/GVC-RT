"""Pin the inherited plan and identities without scanning any dataset inventory."""
from common_s2 import *
from collections import Counter


def main():
    env()
    if (ROOT/'protocol.json').exists():
        return
    hp=load(H_SOURCE/'protocol.json')
    assert hp['B300']['path']==str(B300) and hp['B300']['SHA256']==sha(B300)==B300_SHA
    rows=plan_rows();dev=load(ROOT/'manifests/development_videos.json')
    pool={r['source_group'] for r in load(ROOT/'manifests/effective_ULong_pool.json')}
    assert len({r['source_group'] for r in rows})==400
    assert {r['source_group'] for r in rows}<=pool
    assert {r['source_group'] for r in rows}.isdisjoint({v['source_group'] for v in dev})
    calibration=load(ROOT/'manifests/calibration_windows.json')
    assert len(calibration)==10 and all(r in rows for r in calibration)
    monitor=[next(r for r in rows if r['external_qp']==q) for q in (0,5,9)]
    dump(ROOT/'manifests/monitor_windows.json',monitor)
    # These are inherited GT-only selections, fixed before any new optimization.
    profiles=load(ROOT/'manifests/long_videos.json');rois=load(ROOT/'visual_ROIs.json')
    cases=[]
    for seq,category in [('24747b35-04ca-416c-b1d0-0bc670fbac99','static fine texture'),
                         ('0aa91dd0-0594-4082-9154-d8d2bc94692e','small motion relative to camera-motion case'),
                         ('43be05fb-f1df-4b75-b646-da0920f35361','large camera motion')]:
        v=next(v for v in dev if v['sequence']==seq);p=next(p for p in profiles if p['sequence']==seq)
        roi=dict(rois[seq]);roi.update(first=0,last=63)
        cases.append(dict(sequence=seq,category=category,video=v,QP=5,ROI=roi,
            inherited_GT_motion=p['motion'],inherited_GT_texture=p['texture'],
            selection_rule='fixed inherited GT profiles and ROI; first64 development frames; no new-model output selection'))
    dump(ROOT/'manifests/media_cases.json',dict(frozen_before_training=True,cases=cases,time=time.time()))
    p=dict(task=ROOT.name,branch=BRANCH,base_commit=SOURCE_COMMIT,status='FIXED',
        B300=hp['B300'],recovery_chain=['Original I/P','pilot full receiver/buffers','B1500 TCG','A2000 LoRA','B300 LAST'],
        original=dict(commit=ORIGINAL_COMMIT,worktree=str(ORIGINAL),weights=load(ORIGINAL_EVAL/'protocol.json')['checkpoint_sha256']),
        frozen='all inherited I/P codec, entropy, full Bridge, 3 TCG, generator, 5 existing LoRA; s=.5 train/eval',
        adapter=dict(position='after ada_final before existing head',C=320,r=32,
            A=dict(kernel=1,bias=False),depthwise3D=dict(kernel=[3,3,3],padding=[0,1,1],groups=32,bias=True),
            B=dict(kernel=1,bias=True,initialization='zero weight and bias'),parameters=21696,
            precision='internal FP32 and FP32 residual sum; cast once to existing head dtype',
            history='last2 detached pre-adapter h; left replicate first available; clear at video/GOP reset only'),
        groups={GROUPS[0]:dict(physical_GPU=0,loss='spatial'),GROUPS[1]:dict(physical_GPU=1,loss='spatial+lambda_temp*temporal')},
        spatial=dict(LPIPS_Alex=1.,DISTS=.5,RGB_L1=.1383589248954108),
        temporal=dict(RGB_MCResidual=1.,Sobel_MCResidual=.5,definition='reconstruction MC change minus GT MC change',
            flow='frozen RAFT estimated from GT RGB, not annotated true flow; supervision/metrics only',valid='FB consistency plus in-bounds; normalized valid pixels/channels then valid pairs/window then nonempty windows/batch'),
        calibration=dict(windows=10,QP_counts=dict(Counter(r['external_qp'] for r in calibration)),
            rule='0.1*median(||grad_B spatial||/||grad_B temporal||); finite nonzero denominators only',updates=0,coefficient='gradient_calibration.json; fixed after one training-only calibration'),
        training=dict(updates=100,effective_batch=4,windows=400,seed=TRAIN_SEED,plan_seed=rows[0]['plan_seed'],
            optimizer='AdamW',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=1e-4,clip=1.,
            checkpoints=[0,25,50,75,100],supervised=[49,63],segments=[[49,52],[53,56],[57,60],[61,63]],
            cached_h=[46,63],precision='FP32 frozen reconstruction prefix and module head; codec features inherited FP16',
            gradient='both outputs within each segment; previous frame recomputed using current adapter weights at segment start; detach previous output at boundaries; no backbone BPTT',
            budget_statement='first-stage budget; no convergence or quality launch threshold'),
        development=dict(videos=32,QP=[0,5,9],points_per_state=96,states=['Original','B300','A50','A100','B50','B100'],
            precision=DEPLOYMENT,final_test=False,no_new_dataset_training=True),
        exclusions=['differentiable RAFT loss','latent loss','equivariance loss','RMS diagnosis','GAN','Teacher','new LoRA','LPIPS-VGG training'],
        only_physical_GPUs=[0,1,2,3],sources='dependency_imports.json',time=time.time())
    dump(ROOT/'protocol.json',p)
    protected={str(f):sha(f) for f in [B300,INIT,PILOT,BASELINE,RAFT,H_SOURCE/'protocol.json',H_SOURCE/'plans/index.json',
        PROJECT/'checkpoints/GVC-RT_I.pt',PROJECT/'checkpoints/GVC-RT_P.pt']}
    dump(ROOT/'source_manifest.json',dict(status='PINNED',base_commit=SOURCE_COMMIT,protected_sources=protected,
        protocol_SHA256=sha(ROOT/'protocol.json'),plan_index_SHA256=sha(ROOT/'plans/index.json'),
        inherited_plan_SHA256=sha(H_SOURCE/'plans/train.json.gz'),dependency_imports_SHA256=sha(ROOT/'dependency_imports.json'),
        source_isolation='inherited audited 400 ULong plan and pool; no new full source scan',
        code={str(f.relative_to(ROOT)):sha(f) for f in ROOT.rglob('*.py') if '__pycache__' not in f.parts},time=time.time()))
    dump(ROOT/'status.json',dict(status='IMPLEMENTING',stage='cache_and_mechanical_checks',committed_updates={g:0 for g in GROUPS},time=time.time()))
    dump(ROOT/'task_started.json',dict(time=time.time()))
    dump(ROOT/'audits/data_isolation.json',dict(status='PASS',windows=400,unique_source_groups=400,
        QP_counts=dict(Counter(r['external_qp'] for r in rows)),development_videos=32,plan_exactly_H=True,
        plan_SHA256=sha(ROOT/'plans/train.json.gz'),H_plan_SHA256=sha(H_SOURCE/'plans/train.json.gz'),no_dataset_rescan=True))
    print('PREPARED',flush=True)

if __name__=='__main__':main()
