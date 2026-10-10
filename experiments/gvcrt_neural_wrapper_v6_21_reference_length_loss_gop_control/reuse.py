"""Only source IP=-1 baseline points; strict read-only prior validation."""
from io21 import *
def main():
    from adapter import validate,cache_key
    oldio=module('gop21_old_io',V20/'io20.py');oldio.frozen()
    old=module('gop21_old_adapter_validate',V20/'adapter.py');good=[];bad=[]
    for d in DATASETS:
        for v in sources(d):
            assert v==next(x for x in oldio.sources(d) if x['video_index']==v['video_index'])
            for method,oldm in (('original','original'),('native_initial','native_i_adapt_1000')):
                for q in range(10):
                    p=oldio.point(d,oldm,v['video_index'],q)
                    try:
                        r=load(p);old.validate(r,v,q,oldm)
                        assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:64]
                        r.update(method=method,IP=-1,reused=True,reused_from=str(p),reused_sha256=sha(p),reuse_verified=True,cache_key=cache_key(v,q,method,-1),I_frames=1,P_frames=63,source_method=oldm,source_protocol_sha256=sha(V20/'protocol.json'),reuse_dependency_sha256=sha(ROOT/'audits/dependencies.json'))
                        frame=read(r['frame_metrics_path'])
                        r.update(I_bits_including_associated_headers=int(frame[0]['real_bits']),P_bits_including_headers=sum(int(x['real_bits']) for x in frame[1:]),I_payload_bits=int(frame[0]['payload_bits']),P_payload_bits=sum(int(x['payload_bits']) for x in frame[1:]),header_and_SPS_bits=r['total_bits']-sum(int(x['payload_bits']) for x in frame))
                        validate(r,v,q,method,-1);dump(point(d,method,v['video_index'],q,-1),r)
                        good.append(dict(path=str(p),sha256=sha(p),method=method,dataset=d,video=v['name'],QP=q,IP=-1))
                    except (AssertionError,KeyError,FileNotFoundError) as e:bad.append(dict(path=str(p),reason=repr(e),method=method,dataset=d,video=v['name'],QP=q,IP=-1))
    dump(ROOT/'audits/baseline_reuse.json',dict(status='PASS',expected=320,reused=len(good),records=good,rerun=bad))
    print('REUSE',len(good),'/320',flush=True)
if __name__=='__main__':main()

