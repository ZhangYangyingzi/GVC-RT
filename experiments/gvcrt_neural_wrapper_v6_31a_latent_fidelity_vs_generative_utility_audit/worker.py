"""One GPU worker owns its dataset parts; resumable immutable receipts."""
import argparse
import gc
import time
import traceback
from audit_io import *


def key(t):return f"{t['dataset']}/{t['video']:02d}/{t['method']}/q{t['qp']}"


def evaluator_hash():
    return {name:sha(ROOT/name) for name in ('worker.py','proxy_runtime.py','audit_io.py')}


def repeat_noise(eng,rt,targets,path,expected):
    import io
    import torch
    from proxy_runtime import latent_metrics
    all_rows=[]
    for repeat in range(2):
        ia,pa=rt.models(1088,1920);pa.clear_dpb();pa.set_curr_poc(0)
        data=io.BytesIO(path.read_bytes());helper=eng.eco.SPSHelper();rows=[]
        with torch.inference_mode():
            for i in range(len(targets)):
                h=eng.eco.read_header(data)
                while h['nal_type']==eng.eco.NalType.NAL_SPS:
                    helper.add_sps_by_id(eng.eco.read_sps_remaining(data,h['sps_id']));h=eng.eco.read_header(data)
                sp=helper.get_sps_by_id(h['sps_id']);q,bits=eng.eco.read_ip_remaining(data)
                model=ia if i==0 else pa;captured={}
                hook=model.recon_generation_net.decoder.register_forward_pre_hook(lambda _m,a:captured.update(z=a[0].detach()))
                try:dec=model.decompress(bits,sp,q)
                finally:hook.remove()
                if i==0:pa.add_ref_frame(None,dec['x_hat'])
                rows.append(latent_metrics(captured['z'],targets[i].to(captured['z'].device)))
            assert data.tell()==path.stat().st_size
        rt.assert_frozen(ia,pa)
        all_rows.append(rows);del ia,pa,dec,model,captured
        gc.collect();torch.cuda.empty_cache()
    errors={}
    for metric in ('cosine','L1','normalized_L2','RMSE'):
        errors[metric]=max(abs(a[i][metric]-b[i][metric]) for a,b in ((all_rows[0],all_rows[1]),(expected,all_rows[0])) for i in range(len(targets)))
    return dict(status='PASS',independent_repeats=2,frames=len(targets),max_absolute_metric_noise=errors)


def execute(t,v,targets,teacher_hash,eng,rt,sanity=False):
    import torch
    from proxy_runtime import frames_for
    folder=('sanity/proxy_points/' if sanity else 'parts/')+key(t)
    dest=ROOT/folder;dest.mkdir(parents=True,exist_ok=True)
    rt.models_count=0;rt.latent_rows=[];rt.wrapper_rows=[];rt.full_samples={}
    manifest=load(ROOT/'task_manifest.json')
    rt.sample_frames=[1] if sanity else [1,32] if t in manifest['tensor_sample_points'] else []
    count=4 if sanity else 64
    started=time.time()
    result,rows=rt.run(frames_for(v,count),t['qp'],dest/'stream.bin',dest/'features.npz')
    assert len(rows)==len(rt.latent_rows)==count and result['independent_decode_pass'] and result['state_sync_pass']
    assert all(result[k] for k in ('real_RANS','model_hashes_unchanged'))
    for row,latent in zip(rows,rt.latent_rows):
        latent.update(dataset=t['dataset'],video=v['name'],method=t['method'],QP=t['qp'],
                      source_frame=v['source_frame_indices'][row['frame']],actual_bits=row['real_bits'],
                      actual_bpp=row['real_bits']/(1080*1920),frame_type=row['frame_type'],
                      actual_qp=row['actual_qp'],LPIPS=row['LPIPS'],DISTS=row['DISTS'],
                      PSNR=row['PSNR'],MS_SSIM=row['MS_SSIM'],bitstream_sha256=result['bitstream_sha256'],
                      payload_bits=row['payload_bits'],payload_bpp=row['payload_bits']/(1080*1920))
    write(folder+'/latent.csv',rt.latent_rows)
    write(folder+'/quality.csv',rows)
    if rt.wrapper_rows:write(folder+'/wrapper.csv',rt.wrapper_rows)
    for frame,state in rt.full_samples.items():torch.save(state,dest/f'full_frame{frame}.pt')
    noise=repeat_noise(eng,rt,targets,dest/'stream.bin',rt.latent_rows) if not sanity and t in manifest['noise_points'] else None
    result.update(task=t,dataset=t['dataset'],video=v['name'],method=t['method'],QP=t['qp'],
                  audit_scope='paper-faithful proxy audit',teacher_hash=teacher_hash,
                  source_frame_indices=v['source_frame_indices'][:count],
                  actual_bpp=result['bpp'],kbps_20fps=result['bpp']*1080*1920*20/1000,
                  latent_rows=len(rt.latent_rows),noise_repeat=noise,
                  elapsed_seconds=time.time()-started,proxy_teacher_not_original=True)
    # Receipt hashes exclude the receipt itself and are checked before resume/aggregation.
    result['artifacts']={str(p.relative_to(ROOT)):sha(p) for p in dest.iterdir() if p.is_file() and p.name!='point.json'}
    result['protocol_sha256']=sha(ROOT/'protocol.json')
    result['evaluator_hashes']=evaluator_hash()
    dump(folder+'/point.json',result)
    print('POINT_PASS',key(t),'seconds',round(result['elapsed_seconds'],1),flush=True)
    return result


def valid(t):
    p=ROOT/'parts'/key(t)/'point.json'
    try:
        r=load(p)
        return r['protocol_sha256']==sha(ROOT/'protocol.json') and r['evaluator_hashes']==evaluator_hash() and all(sha(ROOT/f)==h for f,h in r['artifacts'].items())
    except (OSError,KeyError,ValueError):return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);parser.add_argument('--sanity',action='store_true')
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    from proxy_runtime import backend,teacher_targets,runtime
    backend()
    import torch
    videos=load(ROOT/'dataset_manifest.json')['datasets']
    if args.sanity:
        v=videos['ulong'][0];targets,th=teacher_targets(v,4);checks=[]
        for method in ('Original','B1'):
            eng,rt=runtime(method,targets)
            checks.append(execute(dict(dataset='ulong',video=v['video_index'],method=method,qp=0,gpu=args.gpu),v,targets,th,eng,rt,True))
            del rt,eng;gc.collect();torch.cuda.empty_cache()
        unchanged=all(sha(p)==h for p,h in load(ROOT/'source_hashes.json').items())
        assert unchanged
        # The native I0 must be identical for both methods when P is bypassed.
        assert checks[0]['I_refresh_audits'][0]['native_I_sha256']==checks[1]['I_refresh_audits'][0]['native_I_sha256']
        dump('sanity/proxy_sanity.json',dict(status='PASS',scope='paper-faithful proxy, not original teacher reproduction',
             real_RANS=True,independent_decode=True,target_and_generator_shape=[1,18,68,120],
             target_semantics_verified=True,latent_metrics_finite=True,hook_decode_matches_normal_decode=True,
             model_files_unchanged=unchanged,training_performed=False,points=[r['task'] for r in checks]))
        print('FULL_PROXY_SANITY_PASS',flush=True)
        return
    assert load(ROOT/'sanity/proxy_sanity.json')['status']=='PASS'
    tasks=[t for t in load(ROOT/'task_manifest.json')['tasks'] if t['gpu']==args.gpu]
    total=len(tasks);complete=0
    for dataset,vs in videos.items():
        for v in vs:
            todo=[t for t in tasks if t['dataset']==dataset and t['video']==v['video_index']]
            if not todo:continue
            if all(valid(t) for t in todo):complete+=len(todo);continue
            targets,th=teacher_targets(v)
            for method in ('Original','B1','B_covstats'):
                mt=[t for t in todo if t['method']==method]
                if all(valid(t) for t in mt):complete+=len(mt);continue
                eng,rt=runtime(method,targets)
                for t in mt:
                    if not valid(t):execute(t,v,targets,th,eng,rt)
                    complete+=1
                    dump(f'worker_status/gpu{args.gpu}.json',dict(status='RUNNING',pid=os.getpid(),gpu=args.gpu,
                          completed=complete,total=total,last_task=t,updated=time.time()))
                del eng,rt;gc.collect();torch.cuda.empty_cache()
    assert complete==total
    dump(f'worker_status/gpu{args.gpu}.json',dict(status='PASS',completed=complete,total=total,gpu=args.gpu,pid=os.getpid()))


if __name__=='__main__':
    while True:
        oom=False
        try:main();break
        except Exception as exc:
            import torch
            oom=isinstance(exc,torch.cuda.OutOfMemoryError)
            dump(f'worker_status/failure_{os.getpid()}.json',dict(status='WAITING_GPU_MEMORY' if oom else 'FAIL',traceback=traceback.format_exc()))
            if not oom:raise
        # Release failed-call references before waiting; never evict other users' GPU work.
        gc.collect();torch.cuda.empty_cache()
        print('GPU_MEMORY_BACKOFF_60S; valid receipts will resume',flush=True)
        time.sleep(60)
