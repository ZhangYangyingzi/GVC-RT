"""Validate user-approved V6.9 pools, stage reuse, then release the live evaluator."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v68_io import *
import numpy as np

def main():
    frozen(True)
    gate=ROOT/'audits/evaluation_protocol_resolution.json'
    if gate.exists():
        assert load(gate)['status']=='PASS'
        print('EVALUATION ALREADY RELEASED',flush=True);return
    for name in ('source_manifest.json','qp_semantics_audit.json'):
        assert sha(ROOT/name)==sha(V69/name)
    names=('fid_protocol_audit.json','GT_feature_cache.json',*[f'fid_bootstrap_indices_{d}.json' for d in DATASETS])
    for name in names:
        assert sha(ROOT/'audits'/name)==sha(V69/'audits'/name)
    protocol=load(ROOT/'audits/fid_protocol_audit.json')
    assert sha(protocol['implementation'])==protocol['implementation_sha256']
    assert sha(protocol['weights_path'])==protocol['weights_sha256']
    reference_deps=load(V69/'audits/dependencies.json')
    for p in (ENGINE/'engine.py',REPO/'checkpoints/GVC-RT_I.pt',REPO/'checkpoints/GVC-RT_P.pt'):
        assert str(p) in reference_deps and sha(p)==reference_deps[str(p)]
    rows=[];pools={};cache=load(ROOT/'audits/GT_feature_cache.json')['datasets']
    source_io=module('v610_activation_source_io',V62B/'v62b_io.py')
    for d in DATASETS:
        videos=sources(d);item=cache[d];assert sha(item['path'])==item['sha256']
        gt=np.load(item['path']);assert array_hash(gt)==item['array_sha256']
        assert gt.shape==(sum(v['frames'] for v in videos),2048) and np.isfinite(gt).all()
        assert len(item['sequences'])==len(videos)
        bootstrap=load(ROOT/'audits'/f'fid_bootstrap_indices_{d}.json')
        assert bootstrap['repeats']==20 and len(bootstrap['indices'])==20
        offset=0
        for v,entry in zip(videos,item['sequences']):
            n=v['frames']
            assert n in ((96,33) if d.startswith('virat') else (64,))
            assert v['rate_accounting_fps']==(20. if d.startswith('virat') else 30.)
            assert len(v['source_frame_indices'])==len(v['frame_rgb_sha256'])==n
            assert entry['sequence']==v['name'] and entry['frames']==n and entry['source_rgb_sha256']==v['rgb_sha256']
            assert sha(entry['feature_path'])==entry['feature_file_sha256']
            with np.load(entry['feature_path']) as f:
                assert array_hash(f['real'])==entry['GT_array_sha256']
                assert np.array_equal(f['real'],gt[offset:offset+n])
            for draw in bootstrap['indices']:
                assert len(draw)==len(gt)
                block=np.asarray(draw[offset:offset+n]);assert np.all((block>=offset)&(block<offset+n))
            h=hashlib.sha256();count=0
            for i,arr in enumerate(source_io.arrays(v)):
                b=arr.tobytes();assert hashlib.sha256(b).hexdigest()==v['frame_rgb_sha256'][i]
                h.update(b);count+=1
            assert count==n and h.hexdigest()==v['rgb_sha256']
            rows.append(dict(dataset=d,sequence=v['name'],video_index=v['video_index'],
                frames=n,source_frame_indices=v['source_frame_indices'],source_frame_rgb_sha256=v['frame_rgb_sha256'],
                source_rgb_sha256=v['rgb_sha256'],source_sha256=v['source_sha256'],fps=v['rate_accounting_fps'],
                FloLPIPS_pairs=[[i,i+1] for i in range(n-1)],FloLPIPS_source_frame_pairs=[[v['source_frame_indices'][i],v['source_frame_indices'][i+1]] for i in range(n-1)],
                num_transitions=n-1,FID_pool_offset=offset,FID_pool_frames=n))
            offset+=n
            print('VERIFIED VIDEO',d,v['video_index'],'frames',n,flush=True)
        pools[d]=dict(frames=offset,GT_file_sha256=item['sha256'],GT_array_sha256=item['array_sha256'],
            bootstrap_sha256=sha(ROOT/'audits'/f'fid_bootstrap_indices_{d}.json'),bootstrap_repeats=20)
    assert len(rows)==31
    write(ROOT/'audits/evaluation_frames_per_video.csv',rows)
    dump(ROOT/'audits/evaluation_frames_per_video.json',dict(status='PASS',videos=rows,pools=pools))
    staged=ROOT/'audits/evaluation_protocol_verified_pending_activation.json'
    dump(staged,dict(status='PASS',user_correction='Use frozen V6.9 per-video frame indices and pools; UVG/U-Long 64; VIRAT 96/33; VIRAT 20 fps',
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),qp_semantics_sha256=sha(ROOT/'qp_semantics_audit.json'),
        source_experiment=str(V69),pools=pools,all_source_pixels_verified=True,
        unchanged_FID_protocol=True,unchanged_bootstrap_indices=True))
    # Keep the live scheduler blocked until point reuse has finished. The only
    # in-memory difference from the sealed reuse worker is its gate filename.
    src=(ROOT/'reuse.py').read_text()
    src=src.replace("ROOT/'audits/evaluation_protocol_resolution.json'","ROOT/'audits/evaluation_protocol_verified_pending_activation.json'")
    ns=dict(__file__=str(ROOT/'reuse.py'),__name__='v610_staged_reuse')
    exec(compile(src,str(ROOT/'reuse.py'),'exec'),ns);ns['main']()
    frozen(True)
    evidence=load(staged);evidence.update(activated_unix=time.time(),baseline_reuse_sha256=sha(ROOT/'audits/baseline_reuse.json'),
        activation_script_sha256=sha(Path(__file__)),staged_reuse_source_sha256=sha(ROOT/'reuse.py'))
    dump(gate,evidence)
    print('EVALUATION PROTOCOL VERIFIED AND RELEASED',flush=True)

if __name__=='__main__':
    import traceback
    try:main()
    except Exception:
        dump(ROOT/'audits/evaluation_activation_error.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
