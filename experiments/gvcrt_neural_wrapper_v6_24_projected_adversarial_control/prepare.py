from io21 import *
import shutil

def main():
    command([PYTHON,'-B',str(Path(__file__).resolve())]);ROOT.mkdir(exist_ok=True)
    cp=load(V22/'branches/B1_reconstruction/checkpoint_index.json')['1000'];assert cp['sha256']==EXPECTED and sha(cp['path'])==EXPECTED
    cfg=load(V22/'config.json');cfg.update(schema='v624',source_checkpoint={k:cp[k] for k in ('path','sha256','module_hashes','compression_hash')},seed=20261024,domain_seed=20261024,calibration_seed=20261025,branches=BRANCHES,methods=METHODS,expected_points=640,IPs=IPS,source_experiment='v622_B1_1000',lambda_rec=0.1383589248954108,lambda_adv=None,training_plan_reused=False,objective='LPIPS + 0.5*DISTS(require_grad=True) + beta(q)*rate_bpp + 0.01*proxy_L1 + branch_lambda_rec*mean(abs(rec-GT))',i_frame_modes={m:'native_RGB_bypass' for m in EVAL_METHODS})
    cfg['projected']=dict(model='tf_efficientnet_lite1',projection='official CCM/CSM, frozen',scales=4,diffaug=False,D_optimizer=dict(lr=2e-4,betas=[0.,.9]));cfg['lambda_adv']=None
    if not (ROOT/'config.json').exists():dump(ROOT/'config.json',cfg)
    for n in ('train_manifest_ulong.json','train_manifest_uvg.json','dataset_split.json','inherited_exclusions.json','qp_semantics_audit.json'):
        dest=ROOT/n
        if not dest.exists():shutil.copy2(V22/n if (V22/n).exists() else V22/'audits'/n,dest)
    for d in DATASETS:
        p=ROOT/'manifests'/f'{d}.json';p.parent.mkdir(exist_ok=True)
        if not p.exists():shutil.copy2(V22/'manifests'/f'{d}.json',p)
    for n in ('forbidden_content_windows.json','pool_filter.json','historical_exposure.json'):
        p=ROOT/'audits'/n;p.parent.mkdir(exist_ok=True)
        if not p.exists():shutil.copy2(V22/'audits'/n,p)
    deps=load(V22/'audits/dependencies.json')
    for p in list(V22.glob('*.py'))+[V22/'config.json',V22/'protocol.json',V22/'branches/B1_reconstruction/checkpoint_index.json',Path(cp['path'])]:deps[str(p)]=sha(p)
    dump(ROOT/'audits/dependencies.json',deps)
    if not (ROOT/'audits/historical_inventory.json').exists():dump(ROOT/'audits/historical_inventory.json',historical_inventory())
    dump(ROOT/'audits/checkpoint_integrity.json',dict(status='PASS',source=cp,expected_SHA256=EXPECTED,actual_SHA256=sha(cp['path']),historical_checkpoint_index_sha256=sha(V22/'branches/B1_reconstruction/checkpoint_index.json')))
    old=load(V22/'evaluation/config.json');ev=dict(old,schema='v624',methods=METHODS,expected_points=640,IPs=IPS,checkpoints={'B1_initial':old['checkpoints']['B1_reconstruction_1000']},deployment_receiver_hashes={'B1_initial':old['deployment_receiver_hashes']['B1_reconstruction_1000']})
    if not (ROOT/'evaluation/config.json').exists():dump(ROOT/'evaluation/config.json',ev)
    from sampling import make_plan,calibration_plan
    make_plan()
    if not (ROOT/'plans/calibration.json').exists():calibration_plan()
    proto=load(V22/'protocol.json');proto.update(schema='v624',source_checkpoint=cfg['source_checkpoint'],IPs=IPS,methods=METHODS,expected_points=640,I_positions={'-1':[0]},source_protocol_sha256=sha(V22/'protocol.json'),training=dict(shared_plan_sha256=sha(ROOT/'plans/shared.json'),calibration_plan_sha256=sha(ROOT/'plans/calibration.json'),reference='native_I_at_0_then_current_model_no_grad_P1_to48_then_supervised_P49_to63',targets=list(range(49,64)),normalization_denominator=15,reference_detach=True,BPTT=False,optimizer=cfg['optimizer'],updates=1000,branches=BRANCHES,lambda_rec=0.1383589248954108,lambda_adv=None),relative_change='100*(dataset method_mean-dataset reference_mean)/dataset reference_mean; per-video percent mean separate')
    if not (ROOT/'protocol.json').exists():dump(ROOT/'protocol.json',proto)
    paths=['config.json','protocol.json','plans/shared.json','plans/calibration.json','sampling_order.json','dataset_split.json','train_manifest_ulong.json','train_manifest_uvg.json','qp_semantics_audit.json',*[f'manifests/{d}.json' for d in DATASETS]]
    dump(ROOT/'audits/protocol_hashes.json',{n:sha(ROOT/n) for n in paths})
    dump(ROOT/'preflight_audit.json',dict(status='PASS',source_verified=True,pools_inherited_readonly=True,shared_plan=sha(ROOT/'plans/shared.json'),calibration_plan=sha(ROOT/'plans/calibration.json'),test_sources_excluded=True))
    dump(ROOT/'pipeline_status.json',dict(status='READY',phase='calibration',active={},training_steps={b:0 for b in BRANCHES},completed_RD=0,expected_RD=640))
    print('PREPARATION PASS',flush=True)
if __name__=='__main__':main()
