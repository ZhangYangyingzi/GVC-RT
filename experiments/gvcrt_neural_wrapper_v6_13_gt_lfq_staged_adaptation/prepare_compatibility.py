"""Freeze non-test teacher checks before observing candidate outputs."""
import random, hashlib, subprocess
from io_utils import *
def main():
    import cv2,numpy as np
    from PIL import Image
    command([sys.executable,'-B',str(Path(__file__).resolve())])
    seed=20261007;rng=random.Random(seed)
    final=load(V612/'source_manifest.json');test_sha={v.get('source_sha256') for v in final['videos']};test_names={v['name'] for v in final['videos']}
    pool=ROOT.parent/'gvcrt_neural_wrapper_v6_3_dataset_ablation/manifests/ulong_train1024.json'
    val=ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/validation_manifest.json'
    train=load(pool)['videos'];valrows=load(val)['videos']
    training_sha={r['sha256'] for r in train}
    eligible=[r for r in valrows if r['sha256'] not in test_sha|training_sha and Path(r['filename']).stem not in test_names]
    assert len(eligible)>=16
    chosen=rng.sample(eligible,16);samples=[]
    for i,r in enumerate(chosen):
        assert sha(r['path'])==r['sha256']
        cap=cv2.VideoCapture(r['path']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start=rng.randrange(n-3);x=rng.randrange(w-255);y=rng.randrange(h-255);cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        arrays=[]
        for _ in range(4):
            ok,bgr=cap.read();assert ok;arrays.append(cv2.cvtColor(bgr[y:y+256,x:x+256],cv2.COLOR_BGR2RGB).copy())
        cap.release()
        samples.append(dict(domain='ulong',index=i,path=r['path'],source_sha256=r['sha256'],video_id=Path(r['filename']).stem,frame_indices=list(range(start,start+4)),crop=[x,y,256,256],frame_rgb_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]))
    vpath=ROOT.parent/'gvcrt_neural_wrapper_v6_4_vimeo_pretrain/manifests/vimeo_official_train.json';vm=load(vpath)
    held=load(V612/'audits/vimeo_heldout_manifest.json')
    held_ids={c['sequence'] for c in held['clips']}
    # Reserve entire first-level Vimeo source folders, not individual septuplets.
    held_groups={str(x).split('/')[0] for x in held_ids};groups={s.split('/')[0] for s in vm['sequences']}-held_groups
    chosen_groups=rng.sample(sorted(groups),16)
    selected_sids=[rng.choice([s for s in vm['sequences'] if s.split('/')[0]==g]) for g in chosen_groups]
    for i,sid in enumerate(selected_sids):
        assert sid not in held_ids
        start=rng.randrange(4)+1;x=rng.randrange(193);paths=[Path(vm['sequence_root'])/sid/f'im{k}.png' for k in range(start,start+4)]
        arrays=[]
        for path in paths:
            with Image.open(path) as im:assert im.mode=='RGB' and im.size==(448,256);arrays.append(np.asarray(im,dtype=np.uint8)[:,x:x+256].copy())
        samples.append(dict(domain='vimeo',index=i,video_id=sid,source_group=sid.split('/')[0],paths=list(map(str,paths)),source_sha256=[sha(path) for path in paths],frame_indices=list(range(start,start+4)),crop=[x,0,256,256],frame_rgb_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]))
    cfg=dict(schema='v6_13_gt_lfq_staged_adaptation',seed=seed,branches=['A_image_rate','B_gt_lfq_cosine'],updates=5000,checkpoint_steps=[0,100,500,1000,2000,3000,5000],evaluation_step=5000,allowed_gpus=[4,5,6,7],clip_length=4,crop=256,batch=1,external_qps=list(range(10)),optimizer=dict(name='AdamW',wrapper_lr=5e-5,bridge_lr=3e-6,compression_core_lr=1e-6,weight_decay=1e-4,wrapper_bridge_clip=1.,compression_core_clip=1.),trainable=['wrapper','bridge','P_compression_core'],frozen=['I_model','generator','quality_models','teacher'],domain_probability=dict(ulong=.5,vimeo=.5),lambda_cosine=dict(A_image_rate=0.,B_gt_lfq_cosine=.2),cosine_eps=1e-8,cosine_dim=1,lambda_proxy=.01,lambda_struct=.045785124942642835,rate_gradient_mode='scale_ste',force_zero_thres=.12,teacher_gate_required=True,teacher_gate_passed=False,final_datasets=['ulong','uvg'],final_methods=['original','v62','A5000','B5000'],expected_final_RD_points=600,expected_pooled_FID_points=80,expected_bootstrap_rows=1600,compatibility_QPs=[0,4,9],init_checkpoints={k:dict(path=str(REPO/'checkpoints'/name),sha256=sha(REPO/'checkpoints'/name)) for k,name in [('I','GVC-RT_I.pt'),('P','GVC-RT_P.pt')]})
    dump(ROOT/'config.json',cfg)
    dump(ROOT/'manifests/teacher_compatibility_samples.json',dict(status='FROZEN',seed=seed,samples=samples,source_manifests={str(p):sha(p) for p in (pool,val,vpath,V612/'source_manifest.json',V612/'audits/vimeo_heldout_manifest.json')},reserved_vimeo_groups=chosen_groups,not_final_test=True,teacher_selection_on_final_test=False))
    dump(ROOT/'source_manifest.json',final)
    dump(ROOT/'audits/initial_native_provenance.json',dict(checkpoints=cfg['init_checkpoints'],adapted_checkpoints_loaded=False,source_files={str(p):sha(p) for p in (REPO/'src/models/improved_model_gvcrt.py',REPO/'src/models/video_model_gvcrt.py',REPO/'src/models/image_model_gvcrt.py')}))
    dump(ROOT/'pipeline_status.json',dict(status='RUNNING',phase='teacher_compatibility',training_updates={b:0 for b in cfg['branches']},final_evaluation_points=0))
    dump(ROOT/'final_integrity.json',dict(status='PENDING',phase='teacher_compatibility',training_updates={b:0 for b in cfg['branches']},final_evaluation_points=0))
    print('FROZEN NON-TEST SAMPLES',len(samples),flush=True)
if __name__=='__main__':main()
