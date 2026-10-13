"""Freeze the pre-registered experiment and byte-exact inherited sampling plan."""
from io21 import *
import random,collections

def main():
    if (ROOT/'audits/protocol_hashes.json').exists():
        frozen();print('SETUP ALREADY FROZEN');return
    for name in ('logs','audits','plans','manifests','evaluation','calibration','results','branches','features'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    old=load(V26/'config.json');assert sha(old['source_checkpoint']['path'])==EXPECTED
    plan=V26/'plans/shared.json';assert sha(plan)=='e83011570849e027883fc9e34e7bf04907e79dabbcfb89382a50571f774902db'
    (ROOT/'plans/shared.json').write_bytes(plan.read_bytes())
    for n in ('train_manifest_ulong.json','train_manifest_uvg.json','qp_semantics_audit.json'):
        (ROOT/n).write_bytes((V26/n).read_bytes())
    for d in DATASETS:
        src=(V27 if d=='mcljcv' else V26)/'manifests'/f'{d}.json'
        (ROOT/'manifests'/f'{d}.json').write_bytes(src.read_bytes())
    cfg=dict(old,schema='v629',branches=list(BRANCHES),methods=list(METHODS),expected_points=1840,
             seed=20261024,calibration_seed=20261029,lambda_stat=None,
             objective='complete B1 + fixed lambda_stat*DISTS centered spatial covariance loss',
             no_discriminator=True,stat_feature_indices=[2,3,4],stat_feature_channels=[128,256,512],
             stat_ddof=0,stat_denominator_epsilon=1e-6,stat_unit_scale=4096.,rho_cal=.10,
             calibration_eps=1e-12,calibration_windows=40,primary_checkpoint=1000)
    for k in ('projected','rho','lambda_ceiling','lambda_adv','labeler_weight_sha256','condition_modes','i_frame_modes'):
        cfg.pop(k,None)
    dump(ROOT/'config.json',cfg)
    proto=dict(schema='v629',frames=64,crop=256,IP=-1,methods=list(METHODS),datasets=list(DATASETS),
        expected_points=1840,source_checkpoint_SHA256=EXPECTED,shared_plan_SHA256=sha(plan),
        native_I_bypass=True,supervised_positions=list(range(49,64)),warmup_positions=list(range(1,49)),
        detach_references=True,BPTT=False,trainable=['bridge','generator'],frozen=['wrapper','compression_core','I_codec','quality'],
        full_B1_loss='LPIPS+.5*DISTS+beta(q)*rate_bpp+.01*proxy_L1+.1383589248954108*reconstruction_L1',
        stat_loss='mean_layers mean((Cov_rec-Cov_GT)^2)/(mean(Cov_GT^2)+1e-6); spatial ddof0, centered, includes off diagonal',
        stat_layers=['stage2:forward_once[2]','stage3:forward_once[3]','stage4:forward_once[4]'],
        averaging='each supervised frame/15 exactly once, FP32 covariance and restored gradients',
        scales=dict(reconstruction=4096.,stat=4096.),fixed_lambda_rule='median40(.10*norm(g0)/(norm(unit_gstat)+1e-12))',
        no_GAN=True,AdamW=cfg['optimizer'],seed=cfg['seed'],test_checkpoint=1000,
        equal_rate=dict(grid=100,space='ln(bpp)',interpolation='PCHIP',extrapolate=False),
        feature_tolerance=dict(relative_L2=1e-4,max_abs='1e-4+1e-4*max(abs(feature))'),
        FID_tolerance='1e-4+1e-4*max(1,abs(new),abs(old))',
        PSD_roundoff_rule='64*float64_epsilon*matrix_dimension*max(1,max_abs_eigenvalue); reject lower eigenvalues',
        GT_self_tolerance='1e-4+1e-6*(trace(Creal)+trace(Crec))',
        FID=dict(features=2048,precision='float64',ddof=1,no_jitter=True,no_final_clamp=True),
        diagnostic_groups=dict(I0=[0,1],P1_16=[1,17],P17_32=[17,33],P33_48=[33,49],P49_63=[49,64]),
        MCL_overlap='videoSRC17_1920x1080_24 duplicates historical HEVC-B Kimono1; cross-test overlap, not training leakage')
    dump(ROOT/'protocol.json',proto)
    rng=random.Random(20261029);rows=load(plan)['updates'];picked=[]
    for d in ('ulong','uvg'):
        for q in range(10):picked+=rng.sample([r for r in rows if r['domain']==d and r['external_qp']==q],2)
    dump(ROOT/'plans/calibration.json',dict(seed=20261029,updates=picked,only_training=True,source_sha256=sha(plan)))
    counts=collections.Counter((r['domain'],r['external_qp']) for r in rows)
    assert len(rows)==1000 and len(counts)==20 and all(v==50 for v in counts.values())
    ec=load(V26/'evaluation/config.json');ec['methods']={m:([None,None] if m=='original' else [m,m]) for m in METHODS}
    ec['checkpoints']={m:ec['checkpoints'][m] for m in ('B1_initial',)}
    ec['deployment_receiver_hashes']={m:ec['deployment_receiver_hashes'][m] for m in ('B1_initial',)}
    ec['methods']['A_continue_1000']=['A_continue_1000','A_continue_1000'];ec['methods']['B_covstats_1000']=['B_covstats_1000','B_covstats_1000']
    dump(ROOT/'evaluation/config.json',ec)
    deps={};missing=[]
    for folder in (V22,V26,V27,V28):
        for n in ('config.json','protocol.json','train.py','window_training.py','io21.py','evaluate.py','REPORT.md','RESULTS_SUMMARY.md','RESULTS_NUMBERS_ONLY.md'):
            p=folder/n
            if p.exists():deps[str(p)]=sha(p)
            else:missing.append(str(p))
    inherited=load(V26/'audits/dependencies.json');deps.update(inherited)
    deps[str(Path(old['source_checkpoint']['path']))]=EXPECTED
    dump(ROOT/'audits/dependencies.json',deps)
    dump(ROOT/'audits/protocol_hashes.json',{n:sha(ROOT/n) for n in ('config.json','protocol.json','plans/shared.json','plans/calibration.json','train_manifest_ulong.json','train_manifest_uvg.json','qp_semantics_audit.json')})
    dump(ROOT/'audits/preflight.json',dict(status='PASS',source=old['source_checkpoint'],domain_QP_counts={str(k):v for k,v in counts.items()},historical_optional_files_missing=missing,manifest_hashes={d:sha(ROOT/'manifests'/f'{d}.json') for d in DATASETS},no_test_calibration=True))
    dump(ROOT/'pipeline_status.json',dict(status='READY',training_steps={},completed_RD=0,expected_RD=1840,active={},blocked={}))
    print('SETUP PASS 46 videos; 1840 logical points',flush=True)
if __name__=='__main__':main()
