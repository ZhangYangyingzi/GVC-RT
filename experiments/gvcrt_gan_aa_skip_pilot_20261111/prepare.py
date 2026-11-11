"""Register the fixed protocol and source-group-first sampling, without rescanning data."""
from common_s2 import *
from collections import Counter, defaultdict


def main():
    env()
    if (ROOT/'protocol.json').exists():return
    review=load(REVIEW/'audit.json')
    vroot=Path(review['Vimeo']['root'])
    train=vroot/'sep_trainlist.txt';test=vroot/'sep_testlist.txt'
    assert sha(train)==review['Vimeo']['parts']['train']['list_sha256']
    assert sha(test)==review['Vimeo']['parts']['test']['list_sha256']
    train_ids=set(train.read_text().split());test_ids=set(test.read_text().split())
    assert train_ids.isdisjoint(test_ids)
    v_manifest=PROJECT/'experiments/gvcrt_neural_wrapper_v6_13_gt_lfq_staged_adaptation/manifests/vimeo_training.json'
    v=load(v_manifest)
    # This existing source-group-isolated list already excludes historical heldouts.
    vg={g:sorted(set(clips)&train_ids) for g,clips in v['groups'].items()}
    vg={g:clips for g,clips in vg.items() if clips}
    for src,name in [(v_manifest,'Vimeo_isolated_training.json'),(train,'sep_trainlist.txt'),(test,'sep_testlist.txt')]:
        shutil.copy2(src,ROOT/'inputs'/name)
    pool=load(ROOT/'manifests/effective_ULong_pool.json')
    dev=load(ROOT/'manifests/development_videos.json');held=load(ROOT/'manifests/test_videos.json')+dev
    held_groups={x.get('source_group') for x in held};held_sha={x['source_sha256'] for x in held}
    ug=defaultdict(list)
    for p in pool:
        assert p['source_group'] not in held_groups and p['source_sha256'] not in held_sha
        ug[p['source_group']].append(p)
    rng=random.Random(PLAN_SEED)
    def sources(groups):
        while True:
            ids=sorted(groups);rng.shuffle(ids)
            for g in ids:yield g,rng.choice(groups[g])
    us=sources(ug);vs=sources(vg)
    qs=[]
    for i in range(200):
        block=list(range(10));rng.shuffle(block);qs+=block
    def row(domain,slot,q):
        if domain=='ulong':
            g,p=next(us);n=p['frames'];start=rng.randrange(n-63)
            r=dict(domain=domain,logical_sample=True,video=p['video'],identity=p['identity'],source_group=g,source_path=p['source_path'],source_sha256=p['source_sha256'],source_size_bytes=p['source_size_bytes'],source_total_frames=n,source_fps=p.get('source_fps'),width=p['width'],height=p['height'],source_frame_indices=list(range(start,start+64)),crop_x=rng.randrange(p['width']-255),crop_y=rng.randrange(p['height']-255))
        else:
            g,clip=next(vs);path=vroot/'sequences'/clip
            r=dict(domain=domain,logical_sample=True,video=clip,identity=clip,source_group='Vimeo:'+g,source_path=str(path),source_sha256=None,source_total_frames=7,source_fps=None,width=448,height=256,source_frame_indices=list(range(7)),crop_x=rng.randrange(193),crop_y=0,official_split='train')
        r.update(plan_seed=PLAN_SEED,plan_slot=slot,global_update=slot//4+1,logical_rank=slot%4,external_qp=q,crop_width=256,crop_height=256)
        r['canonical_indices']=r['source_frame_indices']
        offsets=[0,2,1];order=[0,1,0,2,0,2,0,2]
        r['actual_frame_qps']=[q if i==0 else q+offsets[order[i%8]] for i in range(r['source_total_frames'] if domain=='vimeo' else 64)]
        r['sample_key']=digest(r)
        return r
    rows=[row('ulong' if i%2==0 else 'vimeo',i,qs[i]) for i in range(2000)]
    # Independent calibration windows, not a prefix of the optimizer sampling plan.
    cal=[row('ulong' if i%2==0 else 'vimeo',-i-1,[0,5,9][i//8]) for i in range(24)]
    assert len({r['sample_key'] for r in rows+cal})==2024
    dump(ROOT/'plans/train.json.gz',rows);dump(ROOT/'plans/calibration.json.gz',cal)
    dump(ROOT/'plans/index.json',dict(windows=2000,shards=[dict(file='plans/train.json.gz',sha256=sha(ROOT/'plans/train.json.gz'))]))
    datasets=dict(Counter(x['dataset'] for x in dev));assert len(dev)==32
    protocol=dict(task='gvcrt_gan_aa_skip_pilot_20261111',status='FROZEN',branch=BRANCH,seed=TRAIN_SEED,only_physical_GPUs=[0,1,2,3],B300=dict(path=str(B300),SHA256=B300_SHA,overlay_last=True,incremental_keys=52),LoRA_gain=.5,groups={g:dict(GPU=i,anti_alias=i>=2,GAN=i%2==1) for i,g in enumerate(GROUPS)},training=dict(max_generator_updates=500,effective_batch=4,microbatch_windows=1,source_mix_ULong_Vimeo=[1,1],mix_reason='no applicable audited mixed ratio; registered budget choice',crop=[256,256],seed=TRAIN_SEED,ULong=dict(real_frames=64,I=0,warmup_P=[1,48],supervised_P=[49,63],chunks=[[49,52],[53,56],[57,60],[61,63]]),Vimeo=dict(real_frames=7,I=0,supervised_P=[1,6],chunks=[[1,4],[5,6]]),normalization='pixels/channels, then supervised frames/transitions per window, then four windows',QP=list(range(10)),save_every=25,commit='atomic resumable latest after each optimizer pair; archives every25',optimizer=dict(type='AdamW',Skip_lr=5e-5,Native_lr=2e-6,betas=[.9,.999],eps=1e-8,weight_decay=1e-4,clip=1)),trainable_boundary=['skip_adapter','decoder.upsample.conv1','decoder.stage3 native Conv2d weight/bias'],frozen=['codec','Bridge','TCG','LoRA','norm','AdaGN','all other decoder parameters'],precision=DEPLOYMENT,loss=dict(LPIPS_Alex=1,DISTS=.5,RGB_L1=.1383589248954108,RAFT_motion=.05,MCResidual=LAMBDA_TEMP,Sobel_factor=.5),GAN=dict(channels=[64,128,256,512,1],kernel=4,strides=[2,2,2,1,1],padding=1,spectral_norm=True,activation='LeakyReLU(.2) first four',D_optimizer=dict(type='Adam',lr=2e-4,betas=[0,.9]),warmup_updates=50,calibration_windows=24,calibration_QP_counts={0:8,5:8,9:8},lambda_rule='.1 * median(norm(g_base)/(norm(g_adv)+1e-12)); 24 windows per architecture, equal architectures',ramp_updates=100,D_per_G=1,G_D_buffers='eval frozen including spectral norm power iteration'),anti_alias=dict(upsample_main_lowpass_cutoff_cycles_per_pixel=.25,passband='rectangular inclusive conjugate cutoff',WSiLU='ideal2x -> x*sigmoid(4x) -> ideal downsample2x',halo='reflection min(16,H-1,W-1)',FP32=True,Nyquist='split on interpolation, conjugate sum on decimation'),cache=dict(startup_windows=16,prefetch_windows=48,atomic=True,shared_all_groups=True,byte_cap=24*2**30,media_cap=8*2**30,min_free_bytes=15*2**30),development=dict(datasets=datasets,videos=32,QP=[0,5,9],points=96,steps=[0,250,500],baselines=['Original','B300'],precision=DEPLOYMENT,no_final_test=True),stop='500 G updates and prescribed evaluation; implementation/numerical/data/codec faults stop affected part',report='execution facts and numerical results only')
    dump(ROOT/'protocol.json',protocol)
    dump(ROOT/'source_inputs.json',dict(readonly=load(ROOT/'inputs/H_source_inputs.json')['readonly'],extra={str(v_manifest):sha(v_manifest),str(train):sha(train),str(test):sha(test)},no_runtime_implicit_imports=True))
    dump(ROOT/'audits/data_isolation.json',dict(status='PASS',ULong_pool_videos=len(pool),ULong_source_groups=len(ug),Vimeo_source_groups=len(vg),Vimeo_eligible_clips=sum(map(len,vg.values())),Vimeo_official_train_test_overlap=0,train_domains=dict(Counter(x['domain'] for x in rows)),training_QP_counts=dict(Counter(x['external_qp'] for x in rows)),unique_windows=len({r['sample_key'] for r in rows}),unique_source_groups=len({r['source_group'] for r in rows}),development_datasets=datasets,ULong_dev_final_source_group_and_SHA_overlap=0,reused_source_isolation='H effective ULong + v6.13 Vimeo grouped training manifest intersect official train; no lifetime pretraining independence claim'))
    dump(ROOT/'status.json',dict(status='PREPARING',stage='implementation_preflight',experiment_directory=str(ROOT),committed_steps={g:0 for g in GROUPS},time=time.time()))


if __name__=='__main__':main()
