"""Freeze validation and final source protocols without using test metrics."""
from fractions import Fraction
from v61_io import *
def main():
    import torch
    torch.set_num_threads(2);frozen();sys.path.insert(0,str(V41));import metric_runtime as metrics
    validation=[]
    for i,r in enumerate(load(ROOT/'validation_manifest.json')['videos']):
        assert sha(r['path'])==r['sha256'];frames=metrics.eco.video_frames(r['path'],32)
        h=hashlib.sha256()
        for frame in frames:h.update(frame.contiguous().numpy().tobytes())
        validation.append(dict(r,dataset='validation',video_index=i,name=Path(r['filename']).stem,frames=32,width=1920,height=1080,rate_fps=float(Fraction(str(r['fps']))),tensor_sha256=h.hexdigest()))
    dump(ROOT/'validation_sources.json',dict(status='PASS',videos=validation))
    canonical=load(V52/'canonical_source_audit.json');loader=module('v61_canonical',V52/'canonical_loader.py');final=[]
    for r in canonical['records']:
        d=Path(r['canonical_dir']);assert [sha(d/f'frame_{i:06d}.png') for i in range(64)]==r['canonical_frame_sha256'];assert loader.rgb_hash(d)==r['whole_sequence_rgb_hash']
        final.append(dict(dataset=r['dataset'],video_index=r['video_index'],name=r['name'],canonical_dir=str(d),frames=64,width=1920,height=1080,rate_fps=30.0,rgb_sha256=r['whole_sequence_rgb_hash']))
    v720=load(V720/'config.json');assert v720['experiment']=='B' and len(v720['videos'])==8
    for r in v720['videos']:
        assert sha(r['path'])==r['source_sha256'];final.append(dict(r,dataset='virat720'))
    assert len(final)==23
    dump(ROOT/'final_sources.json',dict(status='PASS',videos=final,canonical_audit_sha256=sha(V52/'canonical_source_audit.json'),virat_source_config_sha256=sha(V720/'config.json')))
    dump(ROOT/'baseline_reuse_audit.json',dict(status='PENDING_PARITY',threshold=.12,official_none_results_excluded=True,
        ulong_uvg=dict(candidate='V5-A.2 Original',source_hashes_verified=True,frames=64,fps=30.0,padding=[1920,1088],metric_crop=[1920,1080],
            base_checkpoints=load(V52/'checkpoint_audit.json')['original_base_checkpoints'],metric_implementation_sha256=sha(V52/'metric_implementation_audit.json'),
            threshold_evidence='gvc_hooks.load_models default .12; fresh threshold-.12 parity points QP0 and QP9 required on each dataset before any reuse'),
        virat720=dict(reuse=False,reason='Existing 720p baseline uses threshold=None; fresh .12 Original required')))
    print('EVALUATION SOURCES PASS',flush=True)
if __name__=='__main__':main()
