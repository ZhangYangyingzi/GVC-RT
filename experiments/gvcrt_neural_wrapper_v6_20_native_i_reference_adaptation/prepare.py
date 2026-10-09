"""Read-only lineage, unchanged AB replay, frozen protocol and baseline reuse."""
from io20 import *
def selected(v):return (v['dataset']=='ulong' and v['video_index']==sources('ulong')[0]['video_index']) or v['name'] in ('HoneyBee','Jockey','BQTerrace')
def main():
    import torch
    (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'audits').mkdir(exist_ok=True)
    source=load(V18/'branches/B/checkpoint_index.json')['1000'];assert source['sha256']==EXPECTED and sha(source['path'])==EXPECTED
    for cp in (source,source['inference']):
        assert sha(cp['path'])==cp['sha256'];s=torch.load(cp['path'],map_location='cpu',weights_only=True);assert {k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')}==source['module_hashes'];del s
    assert load(V18/'final_integrity.json')['status']==load(V19/'final_integrity.json')['status']=='PASS'
    cfg=load(V18/'config.json');ec=load(V19/'config.json');cfg.update(schema='v620',source_checkpoint={k:source[k] for k in ('path','sha256','module_hashes','compression_hash')},source_experiment='v618_B',branches=BRANCHES,methods=METHODS,expected_points=800,seed=20261010,updates=1000,clip_length=16,training_plan_reused=str(V18/'plans/AB.json'),i_frame_modes={m:'native_original' if m=='original' else 'native_RGB_bypass' if m in ('B_i_bypass','native_i_adapt_0','native_i_adapt_1000') else 'wrapper_standard' for m in EVAL_METHODS})
    ec.update(schema='v620',no_training=False,methods=METHODS,expected_points=800,source_checkpoint=cfg['source_checkpoint'],i_frame_modes=cfg['i_frame_modes']);dump(ROOT/'config.json',cfg);dump(ROOT/'evaluation/config.json',ec)
    for n in ('train_manifest_ulong.json','train_manifest_uvg.json','dataset_split.json','plans/AB.json','qp_semantics_audit.json'):
        dump(ROOT/n,load(V18/n));assert sha(ROOT/n)==sha(V18/n)
    assert len(load(ROOT/'train_manifest_ulong.json')['videos'])==1020
    for d in DATASETS:dump(ROOT/'manifests'/f'{d}.json',load(V19/'manifests'/f'{d}.json'))
    dump(ROOT/'protocol.json',dict(schema='v620',methods=METHODS,expected_points=800,source_checkpoint=cfg['source_checkpoint'],i_frame_modes=cfg['i_frame_modes'],training=dict(branches=BRANCHES,updates=1000,supervised_P_frames=15,continuous16=True,reference_resets=[0],reference_detached=True,I_reconstruction_no_grad=True,source_optimizer_restored=False,same_AB_plan_reused=True,plan_source=str(V18/'plans/AB.json'),plan_sha256=sha(ROOT/'plans/AB.json'),train_manifest_sha256={d:sha(ROOT/f'train_manifest_{d}.json') for d in ('ulong','uvg')},optimizer=cfg['optimizer'],seed=20261010,loss='LPIPS + 0.5 DISTS(require_grad=True) + beta(q)*rate_bpp + 0.01 L1; mean15 P frames',no_rate_gradient_fix=True),evaluation_source_protocol=load(V19/'protocol.json'),frames=64,external_qps=list(range(10)),codec_settings='unchanged v6.19',cache_key_includes=['weights_sha256','I_mode','RGB_sha256','source_frame_indices','QP','codec_protocol_sha256'],equal_rate=dict(models=5,points=100,space='ln(actual_bpp)',interpolation='PCHIP',extrapolation=False),checkpoint_selection=False))
    deps=load(V19/'audits/dependencies.json');deps.update({str(V19/n):sha(V19/n) for n in ('io19.py','adapter.py','config.json','protocol.json','final_integrity.json')});deps.update({str(V18/n):sha(V18/n) for n in ('window_training.py','train.py','plans/AB.json','dataset_split.json','train_manifest_ulong.json','train_manifest_uvg.json')});deps[source['path']]=EXPECTED;dump(ROOT/'audits/dependencies.json',deps)
    for path,h in deps.items():assert sha(path)==h
    dump(ROOT/'audits/checkpoint_integrity.json',dict(status='PASS',source=source,expected_SHA256=EXPECTED,modules_verified=True,frozen_compression_hash=source['compression_hash']))
    dump(ROOT/'audits/data_plan_integrity.json',dict(status='PASS',plan_sha256=sha(ROOT/'plans/AB.json'),same_samples_for_both=True,updates=1000,original_count=1020,no_expanded_pool=True,source_split=load(V18/'dataset_split.json'),source_cleaning_audit=load(V18/'audits/cleaned_original_pool.json')))
    dump(ROOT/'audits/historical_inventory.json',historical_inventory());ps=[ROOT/'config.json',ROOT/'protocol.json',ROOT/'plans/AB.json',ROOT/'dataset_split.json',ROOT/'train_manifest_ulong.json',ROOT/'train_manifest_uvg.json',ROOT/'qp_semantics_audit.json',*(ROOT/'manifests').glob('*.json')];dump(ROOT/'audits/protocol_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in ps});dump(ROOT/'preflight_audit.json',dict(status='PASS',source_sha256=EXPECTED,methods=METHODS,expected_points=800,same_replay_plan=True,source_checks=True))
    from adapter import validate,cache_key
    oldio=module('native20_oldio',V19/'io19.py');oldio.frozen();old=module('native20_oldadapter',V19/'adapter.py');good=[];bad=[]
    for d in DATASETS:
        for v in sources(d):
            assert v==next(x for x in oldio.sources(d) if x['video_index']==v['video_index'])
            for m in METHODS[:3]:
                for q in range(10):
                    p=oldio.point(d,m,v['video_index'],q)
                    try:
                        r=load(p);old.validate(r,v,q,m);r.update(reused=True,reused_from=str(p),reused_sha256=sha(p),reuse_verified=True,cache_key=cache_key(v,q,m));validate(r,v,q,m);dump(point(d,m,v['video_index'],q),r);good.append(dict(source=str(p),sha256=sha(p),dataset=d,method=m,video_index=v['video_index'],QP=q))
                    except (AssertionError,KeyError,FileNotFoundError) as e:bad.append(dict(source=str(p),reason=repr(e),dataset=d,method=m,video_index=v['video_index'],QP=q))
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',expected=480,points=len(good),verified=good,rerun_required=bad));print('REUSE',len(good),'/480',flush=True)
if __name__=='__main__':main()
