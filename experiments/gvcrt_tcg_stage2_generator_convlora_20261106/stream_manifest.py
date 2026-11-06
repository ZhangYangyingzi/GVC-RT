"""Freeze source/byte/hash identities of the shared streams without new test scoring."""
from common_s2 import *
from evaluate import videos,historical_input

def main():
    rows=[]
    for split in ['development','final']:
        for v in videos(split):
            for qp in [0,5,9]:
                p,path,trace=historical_input(v,qp)
                assert trace is not None
                r=p['result'];assert r['bpp']==path.stat().st_size*8/(64*v['height']*v['width'])
                rows.append(dict(split=split,sequence=v['sequence'],QP=qp,stream_path=str(path),stream_sha256=sha(path),
                    real_bytes=path.stat().st_size,bpp=r['bpp'],source_sha256=v['source_sha256'],source_RGB_sha256=v['rgb_sha256'],
                    source_frame_indices=v['source_frame_indices'],point_path=p['audit_source_path'],point_sha256=sha(p['audit_source_path']),
                    codec_feature_trace_available=True))
    assert len(rows)==231 and len({(r['split'],r['sequence'],r['QP']) for r in rows})==231
    dump(ROOT/'bitstream_manifest.json',dict(status='PASS',quality_scoring=False,streams=rows,shared_baseline_candidate=True))
    print('CANONICAL_STREAM_MANIFEST_PASS',len(rows))

if __name__=='__main__':main()
