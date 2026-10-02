"""Unchanged real-RANS Runtime for seven independently factored sender/receiver combinations."""
import argparse,fcntl,math,traceback
from v65_io import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('uvg','ulong'),required=True);p.add_argument('--video',type=int,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);cfg=frozen();assert torch.cuda.device_count()==1
    v=next(x for x in videos() if x['dataset']==a.dataset and x['video_index']==a.video)
    lock=(ROOT/'parts'/f'factorial_{sid(v)}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    e=engine();frames=frames_for(v);assert len(frames)==64
    methods=list(METHODS) if not a.smoke else list(METHODS)
    qps=QPS if not a.smoke else (0,)
    parts=[]
    for method in methods:
        runtime=e.Runtime(cfg,method,torch.device('cuda:0'))
        for q in qps:
            path=point(v,method,q)
            if path.exists():r=load(path);assert r['status']=='PASS' and sha(r['bitstream_path'])==r['bitstream_sha256'];parts.append(r);continue
            sub=Path(v['dataset'])/v['name']/method/f'qp{q}'
            stream=ROOT/'bitstreams'/sub.with_suffix('.bin');feature=ROOT/'features'/sub.with_suffix('.npz')
            vis=ROOT/'visualizations/recon'/sub if chosen(v) else None
            t=time.time();r,rows=runtime.run(frames,q,stream,feature,recon_dir=vis)
            framefile=path.with_suffix('.frames.csv');write(framefile,rows)
            trans=feature.with_suffix('.transitions.csv')
            r.update(status='PASS',dataset=v['dataset'],sequence=v['name'],video_index=v['video_index'],method=method,QP=q,external_qp=q,
                actual_qps=[x['actual_qp'] for x in rows],source_sha256=v['source_sha256'],source_rgb_sha256=v['rgb_sha256'],
                source_frame_indices=v['source_frame_indices'],frame_metrics_path=str(framefile),frame_metrics_sha256=sha(framefile),
                transitions_path=str(trans),transitions_sha256=sha(trans),total_bits=r['real_bytes']*8,physical_gpu=a.gpu,elapsed_seconds=time.time()-t,
                wrapper_hash=None if method=='M00_original' or '_BG_only' in method else cfg['checkpoints'][METHODS[method][0]]['module_hashes']['wrapper'],
                bridge_hash=r['loaded_receiver_module_hashes']['bridge'],generator_hash=r['loaded_receiver_module_hashes']['generator'],
                recon_dir=str(vis) if vis else None,visual_frame_indices=visual_indices(v) if vis else [])
            expected=load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:v['frames']]
            assert r['actual_qps']==expected and r['compression_hash_before']==r['compression_hash_after']==cfg['compression_hash']
            assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'] and r['real_bytes']==r['bytes_consumed']
            assert all(math.isfinite(r[k]) for k in (*METRICS,'kbps','bpp'))
            assert len(rows)==64 and sha(stream)==r['bitstream_sha256']
            if vis:
                import shutil
                for i in range(64):
                    if i not in visual_indices(v):(vis/f'frame_{i:06d}.png').unlink()
            dump(path,r);parts.append(r)
            print('DONE',v['dataset'],v['video_index'],method,q,'seconds',round(r['elapsed_seconds'],1),flush=True)
        del runtime
    if a.smoke:
        audit=bitstream_checks(v,parts,qps);assert all(x['status']=='PASS' for x in audit)
        dump(ROOT/'audits/runtime_smoke.json',dict(status='PASS',dataset=a.dataset,video_index=a.video,methods=methods,QP=0,bitstream_checks=audit,
            compression_hashes=sorted({r['compression_hash_before'] for r in parts}),independent_decode=True,real_RANS=True))
        factor=load(ROOT/'audits/runtime_factorization_audit.json');factor.update(status='PASS',runtime_smoke_audit_sha256=sha(ROOT/'audits/runtime_smoke.json'))
        dump(ROOT/'audits/runtime_factorization_audit.json',factor)
    else:dump(ROOT/'parts'/f'factorial_{sid(v)}_done.json',dict(status='PASS',points=21,dataset=v['dataset'],video_index=v['video_index']))
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs'/f'factorial_failure_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
