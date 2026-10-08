"""Freeze explicit checkpoint hashes and read-only historical protocols."""
from retention_io import *
def main():
    import torch
    command([PYTHON,'-B',str(Path(__file__).resolve())]);assert not (ROOT/'config.json').exists(),'already prepared'
    assert load(ROOT/'audits/dataset_audit.json')['status']=='PASS'
    acceptance=dict(previous_request_digest='fe63a6f00ca6bae59b23c8683146dbe67c1463a552af851bbfc31e9cba',accepted_sha256=EXPECTED,user_confirmation='按完整哈希继续',no_checkpoint_replacement=True)
    dump(ROOT/'audits/hash_confirmation.json',acceptance)
    cfg=load(HEVC/'config.json');cp=load(BASE/'config.json')['checkpoints']['v62'];assert sha(cp['path'])==cp['sha256']==EXPECTED
    checkpoints={'v62_initial':cp};receiver={};audits={};idx=load(V614/'frozen_checkpoint_index.json');assert idx['checkpoint_index_sha256']==sha(V614/'checkpoint_index.json')
    assert load(V614/'validation_selection.json')['selected_step']==1000
    for step in (1000,5000):
        r=idx['checkpoints'][str(step)];assert r['source_step']==1000 and r['adaptation_step']==step and r['compression_hash']==cfg['compression_hash']
        assert sha(r['path'])==r['sha256'];s=torch.load(r['path'],map_location='cpu',weights_only=True)
        assert s['source_step']==1000 and s['adaptation_step']==step and s['compression_hash']==cfg['compression_hash']
        assert {k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')}==r['module_hashes'];del s
        checkpoints[f'uvg_adapt_{step}']=r['inference'];audits[f'uvg_adapt_{step}']=r
    for m,c in checkpoints.items():
        assert sha(c['path'])==c['sha256'];s=torch.load(c['path'],map_location='cpu',weights_only=True)
        assert {k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')}==c['module_hashes']
        receiver[m]={k:tensor_hash({n:t.half() for n,t in s[k].items()}) for k in ('bridge','generator')};del s
    old_runtime=load(HEVC/'runtime_model_metric_audit.json')['methods']['original']
    cfg.update(schema='v614b_cross_domain_retention',checkpoints=checkpoints,methods=METHODS,datasets=DATASETS,no_training=True,no_model_selection=True,expected_points=520,deployment_receiver_hashes=receiver,original_receiver_hashes=old_runtime['receiver_hashes'],metric_module_hashes=old_runtime['metric_module_hashes'])
    dump(ROOT/'config.json',cfg);dump(ROOT/'checkpoint_audit.json',dict(status='PASS',hash_confirmation=acceptance,adapted=audits,frozen_index_sha256=sha(V614/'frozen_checkpoint_index.json'),deployment_receiver_hashes=receiver))
    q=load(BASE/'qp_semantics_audit.json');assert q==load(HEVC/'qp_semantics_audit.json');dump(ROOT/'qp_semantics_audit.json',q)
    dump(ROOT/'protocol.json',dict(no_training=True,no_selection=True,actual_bpp='complete bitstream bytes*8/(frames*width*height)',kbps='bits/frame * inherited per-video rate_accounting_fps / 1000',frames=64,unchanged_color_conversion=True,padding='replicate to multiples of 64; crop to original 1920x1080',codec_dtype='float16',wrapper_dtype='float32',force_zero_thres=.12,methods=METHODS,external_QP=list(range(10)),comparison='four-method common actual bpp interval; 100 uniform ln(bpp) points; PCHIP without extrapolation',delta='method-reference; negative better',FID='pooled fixed dataset frame features; dataset mean actual bpp; no cross-dataset pooling'))
    deps={}
    for path in (BASE/'frozen_source_hashes.json',HEVC/'dependency_hashes.json'):
        for p,h in load(path).items():
            assert sha(p)==h,('historical dependency mismatch',p);deps[p]=h
    for folder in (BASE,HEVC,V614):
        for name in ('config.json','source_manifest.json','final_integrity.json','frozen_checkpoint_index.json','checkpoint_index.json','validation_selection.json'):
            p=folder/name
            if p.exists():deps[str(p)]=sha(p)
    for c in checkpoints.values():deps[c['path']]=c['sha256']
    for r in audits.values():deps[r['path']]=r['sha256']
    dump(ROOT/'audits/dependencies.json',deps);dump(ROOT/'audits/old_inventory.json',inventory())
    local=[ROOT/'config.json',ROOT/'protocol.json',ROOT/'qp_semantics_audit.json',*list((ROOT/'manifests').glob('*.json'))]
    dump(ROOT/'audits/local_protocol_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in local})
    dump(ROOT/'preflight_audit.json',dict(status='PASS',videos=13,frames=832,no_training=True,checkpoint_hashes=True,parameter_hashes=True))
    (ROOT/'parts').mkdir(exist_ok=True);print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()
