"""Freeze checkpoints, FID protocol, evaluation cohorts and optional held-out set."""
import ast,importlib.metadata
from v67b_io import *
def tensor_hash(state):
    h=hashlib.sha256()
    for k,t in state.items():h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    if (ROOT/'preflight_audit.json').exists():frozen();return
    for d in ('audits','logs','parts','results','plots','evaluation','fid_cache'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    import torch,numpy as np
    from PIL import Image
    torch.set_num_threads(2)
    assert load(V67A/'final_integrity.json')['status']=='PASS'
    metadata=load(V66/'branches'/OLD_B/'checkpoint_hashes.json');cps={A:load(V64/'checkpoint_hashes.json')['vimeo_only']};inventory=[]
    for m,step in zip(B_METHODS,(250,500,1000)):
        cp=metadata[str(step)];p=Path(cp['path'])
        if not p.is_file():
            candidates=[p for p in (V66/'branches'/OLD_B/'checkpoints').glob('*.pt') if sha(p)==cp['sha256']];assert len(candidates)==1;p=candidates[0];cp=dict(cp,path=str(p))
        assert sha(p)==cp['sha256'];s=torch.load(p,map_location='cpu',weights_only=True)
        assert s['step']==step and s['branch']==OLD_B and s['compression_hash']==cp['compression_hash']
        assert {k:tensor_hash(s[k]) for k in ('wrapper','bridge','generator')}==cp['module_hashes'];del s
        cps[m]=cp;inventory.append(dict(method=m,step=step,path=str(p),sha256=cp['sha256'],**{k+'_hash':v for k,v in cp['module_hashes'].items()},compression_hash=cp['compression_hash']))
    assert len({r['compression_hash'] for r in inventory})==1
    assert sha(cps[A]['path'])==cps[A]['sha256'];state=torch.load(cps[A]['path'],map_location='cpu',weights_only=True)
    assert {k:tensor_hash(state[k]) for k in ('wrapper','bridge','generator')}==cps[A]['module_hashes'];del state
    dump(ROOT/'audits/B_checkpoint_inventory.json',dict(status='PASS',checkpoints=inventory,same_compression_core=True))
    cfg=dict(no_training=True,no_checkpoint_selection=True,checkpoints=cps,methods=list(METHODS),primary_metrics=list(PRIMARY),compression_hash=inventory[0]['compression_hash'],force_zero_thres=.12,external_qps=list(range(10)),bootstrap_seed=20261004,bootstrap_repeats=20,allowed_gpus=[4,5,6,7],minimum_free_evaluation_mib=16000,gpu_policy='User permits sharing occupied GPU 4/5/6/7 when memory sufficient; never terminate other processes',expected_main_points=1550,fresh_main_points=620,reused_main_points=930,expected_fid_points=200,expected_bootstrap_points=4000,heldout_limit=32,heldout_qps=[0,4,9])
    dump(ROOT/'config.json',cfg);dump(ROOT/'evaluation/config.json',dict(checkpoints=cps))
    for name in ('source_manifest.json','qp_semantics_audit.json'):dump(ROOT/name,load(V67A/name))
    # Search all repository code/configs rather than inventing a new metric definition.
    scan=subprocess.run(['rg','-l','-i','FID|Frechet|Inception|torchmetrics|pytorch_fid|pyiqa','--glob','*.py','--glob','*.json','--glob','!**/.git/**','.'],cwd=REPO,capture_output=True,text=True,check=False)
    assert scan.returncode in (0,1)
    dump(ROOT/'audits/fid_repository_search.json',dict(pattern='FID|Frechet|Inception|torchmetrics|pytorch_fid|pyiqa',scope=str(REPO),files=scan.stdout.splitlines()))
    counts={d:sum(v['frames'] for v in sources(d)) for d in DATASETS}
    packages={k:importlib.metadata.version(k) for k in ('torch','torchvision','numpy','scipy')}
    assert FID_WEIGHTS.is_file()
    dump(ROOT/'audits/fid_protocol_audit.json',dict(status='PASS',implementation=str(FID_SOURCE),implementation_sha256=sha(FID_SOURCE),package=packages,feature_network='Existing patched pytorch-fid compatible Inception-v3 (1008-class pretrained state)',feature_layer='2048-dimensional final average-pool; classifier replaced by Identity',input_range='float RGB [0,1]',RGB_order='RGB NCHW',resize='bilinear 299x299 align_corners=False',normalization='2*x-1; transform_input=False; no ImageNet mean/std',frame_sampling='all exact frozen canonical frames, no stride/resampling',number_of_GT_frames=counts,number_of_reconstructed_frames=counts,GT_feature_cache_policy='One original QP0 GT array per dataset with source manifest and array/file SHA256; every input feature pool verified against it using prior tolerance rtol=1e-5 atol=1e-6; max differences recorded',per_dataset_or_per_sequence='Formal FID pooled per dataset/method/QP; no averaging per-frame or per-sequence FID',weights_path=str(FID_WEIGHTS),weights_sha256=sha(FID_WEIGHTS),covariance='float64 unbiased covariance, exact existing low_rank_fid SVD formula',rate_axis='sum(real_bits)/sum(frame_count/fps)/1000 kbps',prior_use=[str(V64/'metric_implementation_audit.json'),str(V62B/'report.py'),str(ENGINE/'engine.py')],bootstrap='20 repeats, paired sampling with replacement within each sequence, preserve each sequence frame count; same indices for every method/QP; diagnostic only'))
    # Freeze all bootstrap indices before observing new B250/B500 results.
    for di,d in enumerate(DATASETS):
        rng=np.random.RandomState(cfg['bootstrap_seed']+di);draws=[]
        for _ in range(20):
            indices=[];offset=0
            for v in sources(d):indices.extend((offset+rng.randint(0,v['frames'],size=v['frames'])).tolist());offset+=v['frames']
            draws.append(indices)
        dump(ROOT/'audits'/f'fid_bootstrap_indices_{d}.json',dict(seed=cfg['bootstrap_seed']+di,repeats=20,indices=draws,paired_GT_reconstruction=True,strata=[dict(sequence=v['name'],frames=v['frames']) for v in sources(d)]))
    official=Path('/Huang_group/zyyz/datasets/Vimeo90K/vimeo_septuplet');test=official/'sep_testlist.txt';train=official/'sep_trainlist.txt'
    if test.is_file() and train.is_file():
        test_ids=[s.strip() for s in test.read_text().splitlines() if s.strip()];train_ids=set(s.strip() for s in train.read_text().splitlines() if s.strip());assert len(set(test_ids))==len(test_ids) and not set(test_ids)&train_ids
        selected=test_ids[:32];held=[]
        for i,sid in enumerate(selected):
            paths=[official/'sequences'/sid/f'im{j}.png' for j in range(1,8)];rgb=[];filehash=[]
            for p in paths:
                with Image.open(p) as im:assert im.mode=='RGB' and im.size==(448,256);rgb.append(hashlib.sha256(np.asarray(im,dtype=np.uint8).tobytes()).hexdigest())
                filehash.append(sha(p))
            held.append(dict(index=i,sequence=sid,paths=list(map(str,paths)),file_sha256=filehash,frame_rgb_sha256=rgb,frames=7,width=448,height=256))
        dump(ROOT/'audits/vimeo_heldout_manifest.json',dict(status='AVAILABLE',selection='first 32 sequence IDs in official sep_testlist.txt file order; fixed before evaluation',official_test=str(test),test_sha256=sha(test),official_train=str(train),train_sha256=sha(train),disjoint_train_test=True,clips=held,external_qps=[0,4,9],all_native_frames=True,no_resize_crop_padding_source=True,rate_metric='actual bpp; 30fps engine accounting is nominal only, not an assertion of Vimeo source fps',separate_from_formal_64_frame_RD=True))
    else:dump(ROOT/'audits/vimeo_heldout_manifest.json',dict(status='NOT_AVAILABLE',official_test=str(test)))
    deps=load(V67A/'audits/dependencies.json')
    for folder in (V67A,V66,V64,V62B,ENGINE):
        for p in folder.glob('*.py'):deps[str(p)]=sha(p)
    for p in [FID_SOURCE,FID_WEIGHTS,*[Path(cp['path']) for cp in cps.values()],V66/'branches'/OLD_B/'checkpoint_hashes.json',V66/'training_logs'/f'{OLD_B}.csv',V67A/'source_manifest.json',V67A/'qp_semantics_audit.json',V67A/'final_integrity.json']:
        deps[str(p)]=sha(p)
    if test.is_file():deps[str(test)]=sha(test);deps[str(train)]=sha(train)
    for p,h in deps.items():assert sha(p)==h
    dump(ROOT/'audits/dependencies.json',deps);dump(ROOT/'audits/old_inventory.json',old_inventory())
    fixed=[*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'source_manifest.json',ROOT/'qp_semantics_audit.json',ROOT/'evaluation/config.json',ROOT/'audits/fid_protocol_audit.json',ROOT/'audits/vimeo_heldout_manifest.json',*sorted((ROOT/'audits').glob('fid_bootstrap_indices_*.json'))]
    dump(ROOT/'audits/local_protocol.json',{str(p.relative_to(ROOT)):sha(p) for p in fixed})
    dump(ROOT/'preflight_audit.json',dict(status='PASS',no_training=True,FID_protocol_audited=True));dump(ROOT/'final_integrity.json',dict(status='PENDING'))
    print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()
