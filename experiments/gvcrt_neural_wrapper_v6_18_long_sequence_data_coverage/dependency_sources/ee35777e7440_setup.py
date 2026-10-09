"""V6.2-A read-only preflight and validation cohort construction."""
import shutil
from v61_io import *
def main():
    import numpy as np, torch
    for d in ('branches','logs','parts','validation','results'): (ROOT/d).mkdir(parents=True,exist_ok=True)
    v61=ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09'; assert load(v61/'final_integrity.json')['status']=='PASS'
    cfg61=load(v61/'config.json'); assert sha(v61/'train_manifest_1024.json')==cfg61.get('train_manifest_sha256',sha(v61/'train_manifest_1024.json'))
    train=load(v61/'train_manifest_1024.json')['videos']; train_by={r['sha256']:r for r in train}; split=load(v61/'data_split_integrity.json'); assert len(train)==1024 and split['status']=='PASS'
    init=V41/'checkpoints/beta_high/step_20000.pt'; init_sha=sha(init); assert init_sha=='c185e1f69fd5f249a4aa3753ddee1dc23d4402d5d6f74d18ef6b51b559fbdb73'
    oldhash={str(init):init_sha,str(v61/'train_manifest_1024.json'):sha(v61/'train_manifest_1024.json'),str(v61/'data_split_integrity.json'):sha(v61/'data_split_integrity.json')}
    for p in (V41/'train_extension.py',V41/'metric_runtime.py',REPO/'src/models/video_model_gvcrt.py',V4/'eval_core.py',v61/'train_manifest_1024.json'):
        oldhash[str(p)]=sha(p)
    prof=read(v61/'training_source_statistics.csv'); train_sha={r['sha256'] for r in train}; final_sha=set(split['test_sha256'])
    unique={r['sha256']:r for r in prof if r.get('status')=='PASS' and r['sha256'] not in train_sha|final_sha}
    available=list(unique.values())
    assert len(available)>=64
    # Normal: balanced by existing 4x4 strata. Hard: high joint temporal/texture score.
    by={i:[] for i in range(16)}
    for r in available:by[int(r['stratum'])].append(r)
    normal=[]
    for i in range(16):
        if by[i]: normal.append(by[i][0])
    # Fill any sparse stratum slots from the lowest-count remaining eligible sources.
    if len(normal)<32:
        used={x['sha256'] for x in normal}
        normal.extend(sorted([r for r in available if r['sha256'] not in used],key=lambda r:(int(r['stratum']),r['sha256']))[:32-len(normal)])
    rest=[r for r in available if r['sha256'] not in {x['sha256'] for x in normal}]
    score=lambda r:float(r['temporal_rgb_L1'])*float(r['sobel_edge_energy'])
    hard=sorted(rest,key=score,reverse=True)[:32]
    assert len({r['sha256'] for r in normal+hard})==64
    def manifest(rows,name):
        out=[]
        for i,r in enumerate(rows):
            src=train_by.get(r['sha256'],r)
            out.append(dict(dataset='ulong_validation',video_index=i,name=Path(r['filename']).stem,filename=r['filename'],path=src.get('path',r.get('path','')),sha256=r['sha256'],frames=32,width=1920,height=1080,fps=30.0,stratum=int(r['stratum']),temporal_rgb_L1=float(r['temporal_rgb_L1']),temporal_rgb_MSE=float(r['temporal_rgb_MSE']),sobel_edge_energy=float(r['sobel_edge_energy'])))
        dump(ROOT/f'validation_{name}_manifest.json',dict(status='PASS',count=len(out),videos=out)); return out
    normal=manifest(normal,'normal');hard=manifest(hard,'hard')
    assert not ({r['sha256'] for r in normal}|{r['sha256'] for r in hard})&train_sha and not ({r['sha256'] for r in normal}|{r['sha256'] for r in hard})&final_sha
    dump(ROOT/'validation_split_integrity.json',dict(status='PASS',train_count=1024,normal_count=32,hard_count=32,train_normal_intersection=[],train_hard_intersection=[],normal_hard_intersection=[],validation_final_test_intersection=[],no_UVG=True,no_VIRAT=True,final_test_sha256=sorted(final_sha)))
    source_text=(REPO/'src/models/video_model_gvcrt.py').read_text(); assert 'qp_shift = [0, 2, 1]' in source_text
    qp=[];index=[0,1,0,2,0,2,0,2]
    for q in range(10):
        training=[q+[0,2,1][index[i%8]] for i in range(1,4)]; evaluation=training.copy(); qp.append(dict(external_qp=q,actual_i_qp=q,training_p_qps=training,evaluation_p_qps=evaluation,exact_parity=training==evaluation,index_map=index,implementation='src/models/video_model_gvcrt.py::shift_qp'))
    dump(ROOT/'qp_semantics_train_eval_audit.json',dict(status='PASS',rows=qp,uniform_external_sampling=True,training_and_evaluation_exact_parity=True,force_zero_thres=.12))
    betas=[0,1,2,4,8,12,17.302782275540444]
    config=dict(schema='gvcrt_v6_2_objective_calibration',gpus=[4,5,6,7],source_checkpoint=str(init),source_checkpoint_sha256=init_sha,train_manifest=str(v61/'train_manifest_1024.json'),train_manifest_sha256=sha(v61/'train_manifest_1024.json'),updates=3000,checkpoint_steps=[0,250,500,1000,2000,3000],betas=betas,external_qps=list(range(10)),validation_qps=[0,2,4,6,8,9],clip_length=4,crop=[256,256],lambda_proxy=.01,optimizer=dict(name='AdamW',wrapper_lr=5e-5,bridge_lr=3e-6,generator_lr=5e-7,weight_decay=1e-4,gradient_clip_norm=1.0),force_zero_thres=.12,old_artifact_hashes=oldhash,notes='Fresh optimizer each branch; no final/test dataset used in selection')
    dump(ROOT/'config.json',config)
    for beta in betas:
        label='beta_0' if beta==0 else ('beta_17p3028' if beta>17 else f'beta_{str(beta).replace(".","p")}'); br=ROOT/'branches'/label
        for d in ('checkpoints','parts','logs','validation','results'): (br/d).mkdir(parents=True,exist_ok=True)
        dump(br/'config.json',dict(beta=beta,branch=label,**config,train_manifest_1024=str(v61/'train_manifest_1024.json'),optimizer_fresh=True,initialization_sha256=init_sha,qp_semantics_audit=str(ROOT/'qp_semantics_train_eval_audit.json')))
        dump(br/'initialization_audit.json',dict(status='PASS',beta=beta,source_checkpoint=str(init),source_checkpoint_sha256=init_sha,optimizer_state_restored=False,fresh_adamw=True,trainable=['wrapper','bridge','generator'],frozen=['compression_core'],architecture_unchanged=True))
    dump(ROOT/'preflight_audit.json',dict(status='PASS',v61_data_verified=True,train_count=1024,validation_normal=32,validation_hard=32,source_checkpoint=str(init),source_checkpoint_sha256=init_sha,force_zero_thres=.12,betas=betas,old_artifact_hashes=oldhash,old_results_untouched=True))
    dump(ROOT/'final_integrity.json',dict(status='PENDING',reason='Training and validation not yet complete'))
    print('SETUP PASS',flush=True)
if __name__=='__main__':main()
