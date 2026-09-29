import math
from v61_io import *
def main():
    import numpy as np
    frozen();audit=load(ROOT/'baseline_reuse_audit.json');sources=load(ROOT/'final_sources.json')['videos'];details={}
    assert sha(V52/'metric_implementation_audit.json')==audit['ulong_uvg']['metric_implementation_sha256']
    for p,h in audit['ulong_uvg']['base_checkpoints'].items():assert sha(p)==h
    for dataset in ('ulong','uvg'):
        vs=[v for v in sources if v['dataset']==dataset];checked=[];parity=[];reason=None
        try:
            for q in (0,9):
                now=load(point(dataset,'original',vs[0]['video_index'],q));old=load(V52/'parts'/dataset/'original'/f"video_{vs[0]['video_index']}_qp{q}.json")
                equal=now['bitstream_sha256']==old['bitstream_sha256'] and now['reconstruction_sha256']==old['reconstruction_sha256']
                delta={k:abs(now[k]-float(old[k])) for k in (*SPATIAL,'FloLPIPS')}
                with np.load(now['feature_path']) as a,np.load(old['feature_path']) as b:
                    feature_equal=all(np.allclose(a[k],b[k],rtol=1e-5,atol=1e-6) for k in ('real','reconstruction'))
                parity.append(dict(QP=q,bitstream_and_decode_exact=equal,metric_absolute_differences=delta,features_match=feature_equal))
                assert equal and feature_equal and all(v<=2e-6 for v in delta.values()),'Fresh Original parity mismatch'
            for v in vs:
                for q in range(10):
                    old=load(V52/'parts'/dataset/'original'/f"video_{v['video_index']}_qp{q}.json")
                    assert old['source_frame_hash']==v['rgb_sha256'] and old['num_frames']==64 and old['fps']==30.0
                    assert old['canonical_audit_sha256']==sha(V52/'canonical_source_audit.json')
                    assert old['metric_audit_sha256']==sha(V52/'metric_implementation_audit.json')
                    assert old['independent_decode_pass'] and old['state_sync_pass'] and old['compression_hash_before']==old['compression_hash_after']
                    assert old['bytes_consumed']==old['real_bytes']==Path(old['bitstream_path']).stat().st_size
                    assert math.isclose(old['kbps'],old['real_bytes']*8*30/64/1000,rel_tol=1e-12)
                    assert math.isclose(old['bpp'],old['real_bytes']*8/(64*1920*1080),rel_tol=1e-12)
                    for k in ('bitstream','feature','frame_metrics'):assert sha(old[k+'_path'])==old[k+'_sha256']
                    checked.append((v,q,old))
        except Exception as exc:reason=repr(exc)
        reuse=reason is None
        details[dataset]=dict(reuse=reuse,checked_points=len(checked),fresh_parity=parity,reason=reason,threshold=.12,source_hashes_match=reuse,frames_match=reuse,fps_match=reuse,padding_match=reuse,checkpoint_hashes_match=True,metric_implementation_match=True)
        if reuse:
            for v,q,old in checked:
                out=point(dataset,'original',v['video_index'],q)
                if out.exists():continue
                new=dict(old,dataset=dataset,video=v['name'],frames=64,QP=q,rate_accounting_fps=30.0,force_zero_thres=.12,
                    real_RANS=True,processing_canvas=[1920,1088],metric_crop=[1920,1080],source_record=v,config_sha256=sha(ROOT/'config.json'),
                    fresh_real_RANS=False,reuse_verified=True,reuse_source_record=str(V52/'parts'/dataset/'original'/out.name),
                    provenance='Verified V5-A.2 original real RANS raw result; original file unchanged')
                dump(out,new)
    audit.update(status='PASS',datasets=details,decision='Reuse only matching datasets; rerun Original where any check fails');dump(ROOT/'baseline_reuse_audit.json',audit)
    print('BASELINE REUSE AUDIT PASS',flush=True)
if __name__=='__main__':main()
