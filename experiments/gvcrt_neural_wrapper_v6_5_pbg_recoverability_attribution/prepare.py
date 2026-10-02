"""Freeze exact checkpoints and the unchanged 15-video evaluation cohort."""
import traceback
from v65_io import *
def main():
    for d in ('audits','logs','parts','results','visualizations'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    if (ROOT/'preflight_audit.json').exists():frozen(True);return
    assert load(V64/'final_integrity.json')['status']=='PASS'
    cps={'V62':load(V62B/'config.json')['checkpoints']['v62'],'V64':load(V64/'checkpoint_hashes.json')['vimeo_only']}
    expected={'V62':'fe63a6f00ca6bae59b23c8683146dbe64b6be67c1463a552af851bbfc31e9cba','V64':'1c7c9a3f89cbbe4abd4ba6f6f7c963d2fab22737e9928fe222f3aa4c8a8c75e1'}
    for k,cp in cps.items():assert sha(cp['path'])==cp['sha256']==expected[k]
    manifest=load(V64/'evaluation_manifest.json');vs=[v for v in manifest['videos'] if v['dataset'] in ('uvg','ulong')]
    assert len(vs)==15 and sum(v['dataset']=='uvg' for v in vs)==7 and all(v['frames']==64 for v in vs)
    assert {v['name'] for v in vs if v['dataset']=='uvg'}=={'Beauty','Bosphorus','HoneyBee','Jockey','ReadySteadyGo','ShakeNDry','YachtRide'}
    dump(ROOT/'source_manifest.json',dict(videos=vs,origin=str(V64/'evaluation_manifest.json'),origin_sha256=sha(V64/'evaluation_manifest.json')))
    dump(ROOT/'qp_semantics_audit.json',load(V64/'qp_semantics_audit.json'))
    cfg=dict(experiment='B',methods=METHODS,checkpoints=cps,force_zero_thres=.12,external_qps=list(QPS),no_training=True,expected_factorial_points=315,
        expected_proxy_videos=30,expected_proxy_frames=1920,allowed_gpus=[4,5,6,7],gpu_policy='Only GPUs without existing compute processes and with >=22000MiB free',
        comparison_fps=30,compression_hash=load(V64/'branches/vimeo_only/initialization_audit.json')['compression_hash'],visual_frames=[0,15,31,63])
    dump(ROOT/'config.json',cfg)
    deps={}
    for p,h in load(V64/'audits/frozen_dependencies.json').items():assert sha(p)==h;deps[p]=h
    for folder in (V64,V62B,ENGINE,ROOT.parent/'gvcrt_neural_wrapper_v6_3_dataset_ablation'):
        for p in folder.glob('*.py'):deps[str(p)]=sha(p)
    for p in [Path(c['path']) for c in cps.values()]+[V64/'evaluation_manifest.json',AUDIT/'source_stats.py',AUDIT/'motion_method_audit.json',REPO/'src/models/video_model_gvcrt.py']:
        deps[str(p)]=sha(p)
    motion=load(AUDIT/'motion_method_audit.json')
    for p in (motion['source'],motion['weights']):deps[p]=sha(p)
    dump(ROOT/'audits/dependency_hashes.json',deps)
    dump(ROOT/'audits/old_inventory.json',inventory())
    local=[*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'source_manifest.json',ROOT/'qp_semantics_audit.json']
    dump(ROOT/'audits/local_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in local})
    dump(ROOT/'audits/runtime_factorization_audit.json',dict(status='PENDING_RUNTIME_SMOKE',engine=str(ENGINE/'engine.py'),
        original_loading='gvc_hooks.load_models: original GVC-RT_I.pt and GVC-RT_P.pt; no joint payload',
        wrapper_loading='eco.load_joint loads wrapper strictly, float32 eval, requires_grad False',
        BG_loading='eco.apply_joint strictly loads only recon_generation_net.mlp and decoder; eval float16',
        compression_hash_definition='All I-model named parameters plus P-model named parameters excluding recon_generation_net.*',
        sender_state='DMC.compress saves decoded latent feature; apply_feature_adaptor uses feature after I frame; no reconstructed RGB feedback or feature reset in runtime',
        independent_factorization_supported_by_source=True,independent_decode='Runtime creates a fresh decoder and validates reconstructed-frame hashes and latent state synchronization',
        no_training=True,force_zero_thres=.12,full_runtime_hash_and_bitstream_smoke_required=True))
    dump(ROOT/'preflight_audit.json',dict(status='PASS',checkpoint_sha256_pass=True,source_videos=15,formal_execution_requires_smoke=True))
    dump(ROOT/'final_integrity.json',dict(status='PENDING',no_training=True))
    print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs/preflight_failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
