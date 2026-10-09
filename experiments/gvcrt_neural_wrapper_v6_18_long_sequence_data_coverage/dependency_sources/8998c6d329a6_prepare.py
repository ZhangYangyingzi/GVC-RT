"""CPU-only source validation and immutable module resolution."""
import argparse,hashlib,sys
from fractions import Fraction
from parallel_utils import *

def main():
    import torch
    torch.set_num_threads(4)
    for root in (ROOT,A,B):
        root.mkdir(parents=True,exist_ok=True)
        for sub in ('logs','parts','bitstreams','features'): (root/sub).mkdir(exist_ok=True)
    assert load(V52/'final_integrity.json')['status']=='PASS'
    v52=load(V52/'config.json')
    # Resolve clip8 from its existing final record, not from a constructed checkpoint name.
    evidence=V5/'parts/final/clip8/video_0_qp0.json';record=load(evidence)
    cps={'clip8':{'path':record['checkpoint'],'sha256':record['checkpoint_sha256']},'v41':v52['checkpoints']['v41_20000']}
    assert cps['clip8']==v52['checkpoints']['clip8']
    base=load(V52/'checkpoint_audit.json')['original_base_checkpoints']
    hashes=dict(v52['additional_source_hashes']);hashes.update(v52['source_hashes'])
    hashes.update(load(V52/'metric_implementation_audit.json')['source_and_weight_sha256'])
    hashes.update(base)
    for key,value in cps.items():
        assert sha(value['path'])==value['sha256'];hashes[value['path']]=value['sha256']
        payload=torch.load(value['path'],map_location='cpu',weights_only=True)
        value['module_hashes']={part:tensor_dict_hash(payload[part]) for part in ('wrapper','bridge','generator')}
    p_path=next(p for p in base if Path(p).name=='GVC-RT_P.pt')
    original=torch.load(p_path,map_location='cpu',weights_only=True)
    state=original.get('student_ema',original.get('student',original.get('state_dict',original)))
    original_hashes={}
    for part,prefix in [('bridge','recon_generation_net.mlp.'),('generator','recon_generation_net.decoder.')]:
        subset={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        assert subset,prefix
        original_hashes[part]=tensor_dict_hash(subset)
    checkpoints=dict(status='PASS',checkpoints=cps,original_base_checkpoints=base,original_module_hashes=original_hashes,
        clip8_evidence=str(evidence),v41_evidence=str(V52/'config.json'),hash_algorithm='SHA256 over state_dict keys and contiguous tensor bytes in state_dict order')
    for p in (evidence,PUBLIC/'visualizations/originals_manifest.json',PUBLIC/'test_config.json',REPO/'test_video_gvcrt_1088.py',
              V52/'canonical_source_audit.json',V52/'canonical_loader.py',V52/'final_integrity.json'):
        hashes[str(p)]=sha(p)
    manifest=load(PUBLIC/'visualizations/originals_manifest.json')
    selected=[r for r in manifest if r['resolution']=='854x480']
    assert len(selected)==8 and sum(r['group']=='random' for r in selected)==5
    assert sum(r['group']=='specified' for r in selected)==3
    videos=[];audit=[]
    for i,v in enumerate(selected):
        path=Path(v['path']).resolve();assert path.parent==PUBLIC/'visualizations'
        assert v['frames']==(96 if v['group']=='random' else 33)
        probe=load_probe(path)
        assert (probe['width'],probe['height'])==(854,480)
        assert int(probe['nb_read_frames'])==v['frames']
        h=hashlib.sha256();frames=[]
        for a in raw_mp4(v):h.update(a.tobytes());frames.append(hashlib.sha256(a.tobytes()).hexdigest())
        assert len(frames)==v['frames']
        digest=sha(path);hashes[str(path)]=digest
        video=dict(v,dataset='virat',name=v['group']+'__'+v['video'],video_index=i,width=854,height=480,
                   rate_fps=20.0,rgb_sha256=h.hexdigest(),source_sha256=digest)
        videos.append(video)
        audit.append(dict(source_path=str(path),sha256=digest,resolution=[854,480],decoded_frame_count=len(frames),
                          group=v['group'],container_fps=float(Fraction(probe['avg_frame_rate'])),manifest_fps=v['fps'],
                          RGB_sha256=h.hexdigest(),frame_RGB_sha256=frames,probe=probe))
    source_a=dict(status='PASS',source_count=8,random_count=5,specified_count=3,source_paths_are_existing_visualization_originals=True,
                  source_resolution=[854,480],no_source_regeneration=True,no_temporal_resampling=True,no_frame_drop=True,records=audit)
    dump(A/'source_integrity.json',source_a)
    sources=load(V52/'canonical_source_audit.json')['records'];bvideos=[]
    loader=module('source_v52',V52/'canonical_loader.py')
    for s in sources:
        directory=Path(s['canonical_dir'])
        assert [sha(directory/f'frame_{i:06d}.png') for i in range(64)]==s['canonical_frame_sha256']
        assert loader.rgb_hash(directory)==s['whole_sequence_rgb_hash']
        bvideos.append(dict(dataset=s['dataset'],name=s['name'],video=s['name'],group=s['dataset'],video_index=s['video_index'],
              canonical_dir=s['canonical_dir'],width=1920,height=1080,frames=64,rate_fps=30.0,rgb_sha256=s['whole_sequence_rgb_hash']))
    assert len(bvideos)==15
    dump(B/'source_integrity.json',dict(status='PASS',canonical_sources_reused=True,source_hashes_match_v5a2=True,
         canonical_audit=str(V52/'canonical_source_audit.json'),canonical_audit_sha256=sha(V52/'canonical_source_audit.json'),records=sources))
    combos={'ORIGINAL':[None,None],'V41_P_ONLY':['v41',None],'V41_RECEIVER_ONLY':[None,'v41'],'V41_FULL':['v41','v41'],
            'CLIP8_P_ONLY':['clip8',None],'CLIP8_RECEIVER_ONLY':[None,'clip8'],'CLIP8_FULL':['clip8','clip8']}
    for which,root,vs,methods,qps,gpus in [('A',A,videos,{'original':[None,None],'clip8':['clip8','clip8']},list(range(10)),[4,5]),
                                         ('B',B,bvideos,combos,[0,3,6,9],[6,7])]:
        config=dict(experiment=which,evaluation_only=True,no_training=True,methods=methods,qps=qps,gpus=gpus,videos=vs,
            checkpoints=cps,original_module_hashes=original_hashes,frozen_hashes=hashes,
            expected_points=len(vs)*len(methods)*len(qps),bpp_denominator='valid source pixels',
            force_zero_thres=None if which=='A' else .12,
            aggregation='spatial/video rate: equal-weight video means; FloLPIPS: transition-weighted; FID: all dataset frames pooled')
        dump(root/'config.json',config);dump(root/'checkpoint_audit.json',checkpoints)
        dump(root/'pipeline_status.json',dict(status='PREPARED',experiment=which,expected_points=config['expected_points']))
        dump(root/'final_integrity.json',dict(status='PENDING',experiment=which))
        dump(root/'codec_semantics_audit.json',dict(status='PASS',reference=str(REPO/'test_video_gvcrt_1088.py') if which=='A' else str(V52/'config.json'),
            source_resolution=[vs[0]['width'],vs[0]['height']],processing_canvas=[896,512] if which=='A' else [1920,1088],
            padding_right=42 if which=='A' else 0,padding_bottom=32 if which=='A' else 8,padding_mode='replicate',
            codec_input='RGB [0,1] to float16, replicate padding to multiple of 64, then 2*x-1',metric_crop='original source dimensions',
            two_entropy_coders=False if which=='A' else True,force_zero_thres=config['force_zero_thres'],
            force_zero_evidence='public evaluator parser default None' if which=='A' else 'unchanged V5-A.2 neural evaluation',
            intra_period=-1,reset_interval=-1,index_map=[0,1,0,2,0,2,0,2],bpp_denominator='source width*height, as requested'))
        verify_frozen(config)
    print('PREPARE PASS A=160 points B=420 points',flush=True)

def load_probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames,pix_fmt','-of','json',str(path)],text=True))['streams'][0]
if __name__=='__main__':main()
