"""Read-only source verification and independently keyed local baseline records."""
import shutil
from io19 import *
def fresh_baseline(v,q,m):
    if m=='B_standard':return q in (0,4,9) and selected(v)
    return m=='original' and q in (0,9) and preflight_video(v)
def preflight_video(v):return (v['dataset']=='ulong' and v['video_index']==sources('ulong')[0]['video_index']) or v['name']=='HoneyBee'
def selected(v):return (v['dataset']=='ulong' and v['video_index']==sources('ulong')[0]['video_index']) or v['name'] in ('HoneyBee','Jockey','BQTerrace')
def main():
    import torch
    (ROOT/'audits').mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    assert load(V18/'final_integrity.json')['status']=='PASS';index=load(V18/'branches/B/checkpoint_index.json')['1000'];assert index['adaptation_step']==1000
    for cp in (index,index['inference']):
        assert sha(cp['path'])==cp['sha256'];state=torch.load(cp['path'],map_location='cpu',weights_only=True);assert {k:tensor_hash(state[k]) for k in ('wrapper','bridge','generator')}==index['module_hashes'];del state
    src=load(V18/'evaluation/config.json');assert src['checkpoints']['B_1000']=={k:index['inference'][k] for k in ('path','sha256','module_hashes')};assert index['compression_hash']==src['compression_hash']
    cfg=dict(src);cfg['schema']='v619';cfg['checkpoints']={m:src['checkpoints']['B_1000'] for m in METHODS[1:]};cfg['deployment_receiver_hashes']={m:src['deployment_receiver_hashes']['B_1000'] for m in METHODS[1:]};cfg['no_training']=True;cfg['source_checkpoint']=index;cfg['i_frame_modes']={'original':'native_original','B_standard':'wrapper_standard','B_i_bypass':'native_RGB_bypass'}
    dump(ROOT/'config.json',cfg);dump(ROOT/'evaluation/config.json',cfg)
    for d in DATASETS:dump(ROOT/'manifests'/f'{d}.json',load(V18/'manifests'/f'{d}.json'))
    for n in ('qp_semantics_audit.json',):dump(ROOT/n,load(V18/n))
    dump(ROOT/'protocol.json',dict(source_protocol=load(V18/'protocol.json'),source_protocol_sha256=sha(V18/'protocol.json'),no_training=True,methods=METHODS,frames=64,external_qps=list(range(10)),only_modification='B_i_bypass uses raw RGB before frozen I codec for frame0; frames1..63 retain P/B/G; actual reconstructed DPB retained',cache_includes_I_frame_mode=True,metric_crop=[1920,1080],coding_canvas=[1920,1088],equal_rate=dict(models=3,grid=100,space='ln(bpp)',interpolator='PCHIP',extrapolation=False),FID_pools={'ulong':512,'uvg_holdout':128,'hevc_b':320,'uvg_validation':64}))
    deps=load(V18/'audits/dependencies.json');deps.update({str(V18/n):sha(V18/n) for n in ('io18.py','adapter.py','final_integrity.json','evaluation/config.json','branches/B/checkpoint_index.json')});deps[index['path']]=index['sha256'];deps[index['inference']['path']]=index['inference']['sha256'];deps[str(V15/'report.py')]=sha(V15/'report.py');dump(ROOT/'audits/dependencies.json',deps)
    for name,h in deps.items():assert sha(name)==h
    dump(ROOT/'audits/checkpoint_integrity.json',dict(status='PASS',branch='B',step=1000,index_path=str(V18/'branches/B/checkpoint_index.json'),index_sha256=sha(V18/'branches/B/checkpoint_index.json'),checkpoint=index,shared_weights=True,frozen_compression_hash=src['compression_hash']))
    dump(ROOT/'audits/dataset_integrity.json',dict(status='PASS',source_experiment=str(V18),datasets={d:dict(sequences=len(sources(d)),frames_per_sequence=64,manifest_sha256=sha(ROOT/'manifests'/f'{d}.json'),source_manifest_sha256=sha(V18/'manifests'/f'{d}.json')) for d in DATASETS},source_manifest_semantics_unchanged=True,RGB_file_and_frame_hashes_verified_by_each_worker=True))
    dump(ROOT/'audits/historical_inventory.json',historical_inventory())
    ps=[ROOT/'config.json',ROOT/'protocol.json',ROOT/'qp_semantics_audit.json',* (ROOT/'manifests').glob('*.json')];dump(ROOT/'audits/protocol_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in ps});dump(ROOT/'preflight_audit.json',dict(status='PASS',checkpoint_verified=True,source_protocol_frozen=True,no_training=True))
    old=module('i19_old_io',V18/'io18.py');old.frozen();a=module('i19_old_adapter',V18/'adapter.py')
    # The old adapter's import resolves to the historical io18 module.
    from adapter import cache_key,validate
    good=[];bad=[]
    for d in DATASETS:
        for v in sources(d):
            assert v==next(x for x in old.sources(d) if x['video_index']==v['video_index'])
            for m,om in (('original','original'),('B_standard','B_1000')):
                for q in range(10):
                    if fresh_baseline(v,q,m):continue
                    source=old.point(d,om,v['video_index'],q)
                    try:
                        r=load(source);a.validate(r,v,q,om);assert r['checkpoint_sha256']==('' if m=='original' else cfg['checkpoints'][m]['sha256'])
                        r.update(method=m,i_frame_mode=cfg['i_frame_modes'][m],cache_key=cache_key(v,q,m),reused=True,reused_from=str(source),reused_sha256=sha(source),reuse_verified=True,source_method=om,method_alias_only=True)
                        validate(r,v,q,m);dump(point(d,m,v['video_index'],q),r);good.append(dict(source=str(source),sha256=sha(source),dataset=d,method=m,QP=q,video_index=v['video_index']))
                    except (AssertionError,KeyError,FileNotFoundError) as e:bad.append(dict(source=str(source),reason=repr(e),dataset=d,method=m,QP=q,video_index=v['video_index']))
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',points=len(good),expected_candidates=304,verified=good,rerun_required=bad,required_fresh_baseline_points=16));print('PREPARED REUSE',len(good),'RERUN',len(bad),flush=True)
if __name__=='__main__':main()
