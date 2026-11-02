"""Fixed held-out real-RANS validation; no checkpoint selection or tuning."""
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PILOT = REPO/'experiments/gvcrt_tcg_stage1_pilot'
V640 = REPO/'experiments/gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100'
V68 = REPO/'experiments/gvcrt_neural_wrapper_v6_8a_b_continue_vs_interface_preservation'
ENGINE = REPO/'experiments/gvcrt_parallel_v5a3_v60'
LARGE = Path('/Huang_group/zyyz/home_dir/tmp/gvcrt_tcg_stage1_evaluation')
METRICS = ('bpp','kbps','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS')
FIELDS = ('domain','sequence','QP','step','frames','real_bytes',*METRICS,
          'checkpoint_sha256','bitstream_file','bitstream_sha256')
os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='5')
sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def public(value):
    if isinstance(value,dict):
        return {str(k):public(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [public(v) for v in value]
    if isinstance(value,str):
        if value.startswith(str(REPO)+'/'):
            return str(Path(value).relative_to(REPO))
        if value.startswith('/') and '\n' not in value:
            return Path(value).name
        return value.replace(str(REPO),'<repo>').replace('/Huang_group/zyyz','<local>')
    return value


def dump(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(public(value),f,indent=2,allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)


def event(name,**kwargs):
    row=public(dict(event=name,time=time.time(),**kwargs))
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/events.jsonl').open('a') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row),flush=True)


def prepare():
    ROOT.mkdir(exist_ok=True);LARGE.mkdir(parents=True,exist_ok=True)
    audit={}
    for name in ('run_status.json','final_integrity.json','protocol.json','source_manifest.json'):
        audit[name]=load(PILOT/name)
    assert audit['run_status.json']['status']=='PASS'
    assert audit['run_status.json']['committed_updates']==100
    assert audit['final_integrity.json']['status']=='PASS'
    for name,expected in audit['source_manifest.json'].items():
        assert sha(REPO/name)==expected,('Pilot source mismatch',name)
    training=load(PILOT/'protocol.json')
    plan_path=Path(training['plan'])
    assert str(plan_path) in (PILOT/'train.py').read_text() or "V640" in (PILOT/'train.py').read_text()
    assert plan_path==V640/'plans/first100.json'
    assert sha(plan_path)==training['plan_sha256']==audit['final_integrity.json']['plan_sha256']
    plan=load(plan_path)['updates']
    assert len(plan)==100 and [x['step'] for x in plan]==list(range(1,101))
    train_rows=[x['source_row'] for x in plan]
    names={Path(r['video']).stem.casefold() for r in train_rows}
    source_hashes={r['source_sha256'] for r in train_rows}
    manifest_path=V640/'eval_manifest.json'; videos=load(manifest_path)['hevc_b']
    assert len(videos)==2
    assert all(v['name'].casefold() not in names and v['source_sha256'] not in source_hashes for v in videos),'Evaluation overlaps training'
    checkpoint_manifest=[]
    for step in (0,100):
        path=PILOT/f'checkpoints/adaptation_step_{step}.pt'
        checkpoint_manifest.append(dict(filename=path.name,size_bytes=path.stat().st_size,sha256=sha(path)))
    dump(ROOT/'checkpoint_manifest.json',dict(checkpoints=checkpoint_manifest))
    audit['checkpoint_files_verified']=checkpoint_manifest
    audit['source_manifest_hash_checks']='PASS'
    dump(ROOT/'pilot_audit.json',audit)
    dump(ROOT/'split_audit.json',dict(status='PASS',training_updates=100,
         training_plan='plans/first100.json',training_plan_sha256=sha(plan_path),
         training_sequences=sorted({r['video'] for r in train_rows}),
         heldout_sequences=[v['name'] for v in videos],
         sequence_name_overlap=[],source_sha256_overlap=[],
         source_manifest=str(manifest_path.relative_to(REPO)),source_manifest_sha256=sha(manifest_path)))
    cfg=load(V640/'evaluation_config.json');qps=cfg.get('external_qps',[0,5,9])
    assert qps==[0,5,9]
    protocol=dict(checkpoint_steps=[0,100],physical_GPU=5,QP=qps,
         validation_manifest=str(manifest_path.relative_to(REPO)),validation_manifest_sha256=sha(manifest_path),
         training_plan_sha256=sha(plan_path),expected_points=2*len(videos)*len(qps),
         sequences=[{k:v[k] for k in ('dataset','name','frames','width','height','source_frame_indices','source_sha256','rgb_sha256')} for v in videos],
         IP=-1,GOP='one I at frame 0 followed by 63 P frames',I_mode='native RGB bypass',
         frame_selection='existing v6.40 evaluation manifest, all 64 frames',
         input_order='RGB float32 -> float16 -> replicate pad to multiple of 64 -> 2*x-1',
         codec_precision='FP16, including bridge and generator; established real-RANS evaluation protocol',
         metrics_precision='FP32',force_zero_thres=.12,ec_part=1,
         metrics=list(METRICS),PSNR='-10*log10(sum RGB pixel SSE / (frames*height*width*3))',
         MS_SSIM='mean frame MS-SSIM from existing frame_metrics/core.ms_ssim_rgb',
         LPIPS_DISTS='mean frame scores, including I frame',
         FloLPIPS='existing official-compatible FloLPIPS/PWC, mean of 63 adjacent transitions',
         kbps='full persisted stream bytes * 8 * manifest rate_accounting_fps / frames / 1000',
         bpp='full persisted stream bytes * 8 / (frames * unpadded height * width)',
         rate_accounting_fps={v['name']:v['rate_accounting_fps'] for v in videos},
         delta='step100 minus step0, matched domain/sequence/QP',
         no_tuning=True,no_checkpoint_selection=True,fresh_encode_per_checkpoint=True,
         independent_decode=True,cross_checkpoint_stream_reuse=False,
         TCG_buffer_reset='assert clear at sequence start, each I/GOP boundary, and sequence end for all codec instances',
         large_artifacts='local scratch only; excluded from publication',
         reused_entrypoints=[str((V640/'evaluate.py').relative_to(REPO)),str((V68/'evaluate.py').relative_to(REPO)),
                            str((ENGINE/'engine.py').relative_to(REPO))],
         adaptation='reuse engine Runtime.models, native RANS compress/decompress, stream helpers and spatial/FloLPIPS metrics; fresh streams, no FID')
    dump(ROOT/'protocol.json',protocol)
    # Public manifest contains only repository-relative or basename identifiers.
    sources=[ROOT/'evaluate.py',PILOT/'train.py',PILOT/'loss.py',PILOT/'preflight.py',plan_path,
             V640/'evaluate.py',V640/'adapter35.py',V640/'evaluation_config.json',manifest_path,
             V68/'evaluate.py',V68/'v68_io.py',ENGINE/'engine.py',ENGINE/'parallel_utils.py',
             REPO/'src/models/image_model_gvcrt.py',REPO/'src/models/video_model_gvcrt.py',
             REPO/'src/models/improved_model_gvcrt.py',REPO/'src/utils/stream_helper.py',
             REPO/'experiments/gvcrt_neural_wrapper_v4_1_high_convergence_metrics/metric_runtime.py',
             REPO/'experiments/gvcrt_neural_wrapper_v4_progressive_joint/eval_core.py',
             REPO/'experiments/gvcrt_neural_wrapper_v2_joint/core.py']
    dump(ROOT/'source_manifest.json',dict(files=[dict(file=str(p.relative_to(REPO)),size_bytes=p.stat().st_size,sha256=sha(p)) for p in sources]))
    write_csv(ROOT/'metrics.csv',[],FIELDS)
    return protocol,videos,checkpoint_manifest


def evaluate(protocol,videos,checkpoints):
    import torch
    sys.path.insert(0,str(REPO/'experiments/gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
    sys.path.insert(0,str(REPO));sys.path.insert(0,str(ENGINE));sys.path.insert(0,str(V640))
    from adapter35 import frames_for
    from src.utils.common import set_torch_env
    set_torch_env();torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable on physical GPU 5')
    torch.empty(1,device='cuda:0');torch.cuda.synchronize()
    import engine
    core,eco=engine.core,engine.eco
    quality=core.quality_models(torch.device('cuda:0'))
    flow,perceptual=engine.metrics.flo.load_models(torch.device('cuda:0'))
    metric_hashes=dict(quality=[core.module_hash(m) for m in quality],flow=core.module_hash(flow),FloLPIPS=core.module_hash(perceptual))
    dump(ROOT/'metric_audit.json',dict(module_hashes=metric_hashes,
         flo_official_commit=engine.metrics.flo.OFFICIAL_COMMIT,
         flo_source_sha256=engine.metrics.flo.OFFICIAL_SOURCE_SHA256,
         PWC_weights_sha256=engine.metrics.flo.PWC_WEIGHT_SHA256))
    m=load(ROOT/'source_manifest.json')
    dependency_paths=[Path(engine.metrics.flo.__file__),engine.metrics.flo.PWC_WEIGHTS,
         engine.metrics.flo.OFFICIAL_ROOT/'flolpips.py',engine.metrics.flo.OFFICIAL_ROOT/'pwcnet.py',
         Path(sys.modules['gvc_hooks'].__file__)]
    m['external_dependencies']=[dict(file=p.name,sha256=sha(p),size_bytes=p.stat().st_size) for p in dependency_paths]
    dump(ROOT/'source_manifest.json',m)
    results=[]
    for step,cp in zip((0,100),checkpoints):
        path=PILOT/'checkpoints'/cp['filename']
        assert sha(path)==cp['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['plan_cursor']==state['committed_updates']==step
        assert state['protocol']['plan_sha256']==protocol['training_plan_sha256']
        class Runtime(engine.Runtime):
            def __init__(self):
                self.device=torch.device('cuda:0');self.wrapper=None
                self.config=dict(force_zero_thres=.12);self.payload=state['models']
            def models(self,h,w):
                im,pm=super().models(h,w)
                pm.recon_generation_net.decoder.set_temporal_cascade_grad(False)
                assert all(not p.requires_grad for mod in (im,pm) for p in mod.parameters())
                return im,pm
        runtime=Runtime()
        for v in videos:
            if not Path(v['input_dir']).is_dir():
                raise FileNotFoundError('Validation frames unavailable: '+v['name'])
            frames=frames_for(v)
            for qp in protocol['QP']:
                key=f"step{step}_{v['dataset']}_{v['name']}_qp{qp}"
                event('POINT_STARTED',point=key,completed_points=len(results))
                result=run_point(engine,runtime,quality,flow,perceptual,frames,v,qp,step,cp,key)
                results.append(result)
                write_csv(ROOT/'metrics.csv',results,FIELDS)
                dump(ROOT/'run_status.json',dict(status='RUNNING',completed_points=len(results),expected_points=protocol['expected_points'],physical_GPU=5))
                event('POINT_PASS',point=key,completed_points=len(results))
            del frames
        del state,runtime
    assert len(results)==protocol['expected_points']
    for cp in checkpoints:
        p=PILOT/'checkpoints'/cp['filename']
        assert p.stat().st_size==cp['size_bytes'] and sha(p)==cp['sha256'],'Checkpoint mutated'
    assert metric_hashes==dict(quality=[core.module_hash(m) for m in quality],flow=core.module_hash(flow),FloLPIPS=core.module_hash(perceptual))
    return results


def run_point(engine,runtime,quality,flow,perceptual,frames,v,qp,step,cp,key):
    import torch
    core,eco=engine.core,engine.eco
    h,w=v['height'],v['width'];ch,cw=(h+63)//64*64,(w+63)//64*64
    n=len(frames);assert n==64
    sps=dict(sps_id=0,height=ch,width=cw,ec_part=1,use_ada_i=0)
    resets=[]
    def reset(pm,label):
        populated=[getattr(pm.recon_generation_net.decoder,k)._previous is not None for k in ('tcg_low','tcg_mid','tcg_out')]
        pm.clear_dpb();pm.set_curr_poc(0)
        assert not pm.dpb
        assert all(getattr(pm.recon_generation_net.decoder,k)._previous is None for k in ('tcg_low','tcg_mid','tcg_out'))
        resets.append(dict(boundary=label,previous_populated=populated,cleared=True))
    def populated(pm):
        assert all(getattr(pm.recon_generation_net.decoder,k)._previous is not None for k in ('tcg_low','tcg_mid','tcg_out'))
    begun=time.time();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        ie,pe=runtime.models(ch,cw);idc,pdc=runtime.models(ch,cw)
        before=[core.module_hash(m) for m in (ie,pe,idc,pdc)]
        reset(pe,'encoder_sequence_start');reset(pdc,'sync_decoder_sequence_start')
        encoded_hashes=[];actual=[];buffer=io.BytesIO();eco.write_sps(buffer,sps)
        for i,cpu in enumerate(frames):
            x=engine.replicate_pad(cpu.cuda().half(),ch-h,cw-w)*2.0-1.0
            q=qp if i==0 else pe.shift_qp(qp,eco.INDEX_MAP[i%8]);actual.append(q)
            if i==0:
                encoded=ie.compress(x,q);decoded=idc.decompress(encoded['bit_stream'],sps,q)
                assert torch.equal(encoded['x_hat'],decoded['x_hat'])
                reset(pe,'encoder_GOP_I_0');reset(pdc,'sync_decoder_GOP_I_0')
                pe.add_ref_frame(None,decoded['x_hat']);pdc.add_ref_frame(None,decoded['x_hat'])
            else:
                encoded=pe.compress(x,q);decoded=pdc.decompress(encoded['bit_stream'],sps,q)
                assert torch.equal(pe.dpb[0].feature,pdc.dpb[0].feature),'RANS encoder/decoder feature mismatch'
            eco.write_ip(buffer,i==0,0,q,encoded['bit_stream'])
            encoded_hashes.append(eco.tensor_sha(decoded['x_hat']))
            if i%16==0:event('ENCODE_PROGRESS',point=key,frame=i)
        populated(pdc)
        assert before==[core.module_hash(m) for m in (ie,pe,idc,pdc)]
        reset(pe,'encoder_sequence_end');reset(pdc,'sync_decoder_sequence_end')
        data=buffer.getvalue();stream_path=LARGE/'bitstreams'/f'{key}.bin'
        stream_path.parent.mkdir(parents=True,exist_ok=True)
        tmp=stream_path.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(stream_path)
        del ie,pe,idc,pdc;torch.cuda.empty_cache()
        ia,pa=runtime.models(ch,cw);before=[core.module_hash(m) for m in (ia,pa)]
        reset(pa,'independent_decoder_sequence_start')
        stream=io.BytesIO(stream_path.read_bytes());helper=eco.SPSHelper()
        hashes=[];rows=[];trans=[];previous_gt=previous_out=None
        for i,cpu in enumerate(frames):
            position=stream.tell();header=eco.read_header(stream)
            while header['nal_type']==eco.NalType.NAL_SPS:
                helper.add_sps_by_id(eco.read_sps_remaining(stream,header['sps_id']));header=eco.read_header(stream)
            current=helper.get_sps_by_id(header['sps_id']);q,bits=eco.read_ip_remaining(stream)
            assert current==sps and q==actual[i]
            assert (header['nal_type']==eco.NalType.NAL_I)==(i==0)
            if i==0:
                decoded=ia.decompress(bits,current,q);reset(pa,'independent_decoder_GOP_I_0');pa.add_ref_frame(None,decoded['x_hat'])
            else:decoded=pa.decompress(bits,current,q)
            hashes.append(eco.tensor_sha(decoded['x_hat']));assert hashes[-1]==encoded_hashes[i]
            out=core.unit(decoded['x_hat'],h,w).float();gt=cpu.cuda()
            spatial=eco.frame_metrics(out,gt,quality)
            assert all(math.isfinite(float(x)) for x in spatial.values())
            rows.append(dict(frame=i,external_qp=qp,actual_qp=q,real_bits=(stream.tell()-position)*8,payload_bits=len(bits)*8,**spatial))
            if previous_gt is not None:
                diff=flow(previous_gt,gt)-flow(previous_out,out)
                magnitude=diff.square().sum(1,keepdim=True).sqrt().sum()
                assert torch.isfinite(magnitude) and magnitude>0,'FloLPIPS undefined motion weights'
                value=float(perceptual(previous_gt,previous_out,diff,normalize=True).item())
                assert math.isfinite(value)
                trans.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=value))
            previous_gt,previous_out=gt,out
            if i%16==0:event('DECODE_METRICS_PROGRESS',point=key,frame=i)
        populated(pa);reset(pa,'independent_decoder_sequence_end')
        assert before==[core.module_hash(m) for m in (ia,pa)]
        assert len(rows)==64 and len(trans)==63 and stream.tell()==len(data)
        assert sum(r['real_bits'] for r in rows)==len(data)*8
        write_csv(ROOT/'frames'/f'{key}.csv',rows)
        write_csv(ROOT/'transitions'/f'{key}.csv',trans)
        mse=sum(r['pixel_SSE'] for r in rows)/(n*h*w*3)
        result=dict(domain=v['dataset'],sequence=v['name'],QP=qp,step=step,frames=n,real_bytes=len(data),
                    bpp=len(data)*8/(n*h*w),kbps=len(data)*8*v['rate_accounting_fps']/n/1000,
                    PSNR=-10*math.log10(max(mse,1e-15)),
                    **{k:sum(r[k] for r in rows)/n for k in ('MS_SSIM','LPIPS','DISTS')},
                    FloLPIPS=sum(r['FloLPIPS'] for r in trans)/63,
                    checkpoint_sha256=cp['sha256'],bitstream_file=stream_path.name,bitstream_sha256=sha(stream_path))
        assert all(math.isfinite(result[k]) for k in METRICS)
        dump(ROOT/'points'/f'{key}.json',dict(status='PASS',result=result,
             actual_qps=actual,source_rgb_sha256=v['rgb_sha256'],source_frame_indices=v['source_frame_indices'],
             real_RANS=True,fresh_encode=True,independent_decode=True,state_sync_pass=True,
             cross_checkpoint_bitstream_reuse=False,bytes_consumed=stream.tell(),
             model_hashes_unchanged=True,TCG_reset_audits=resets,
             reconstruction_sha256=eco.sha256_bytes(''.join(hashes).encode()),
             physical_GPU=5,seconds=time.time()-begun,
             peak_memory_reserved_bytes=torch.cuda.max_memory_reserved()))
        del ia,pa;torch.cuda.empty_cache()
        return result


def finish(status,protocol,results,error=None):
    deltas=[]
    if status=='PASS':
        for sequence in protocol['sequences']:
            for q in protocol['QP']:
                pair=[next(r for r in results if r['sequence']==sequence['name'] and r['QP']==q and r['step']==s) for s in (0,100)]
                a,b=pair
                deltas.append(dict(domain=a['domain'],sequence=a['sequence'],QP=q,
                                   **{f'delta_{k}':b[k]-a[k] for k in METRICS}))
    write_csv(ROOT/'deltas.csv',deltas,('domain','sequence','QP',*[f'delta_{k}' for k in METRICS]))
    summary=dict(status=status,completed_points=len(results),expected_points=protocol.get('expected_points',12),
                 delta_definition='step100 minus step0',deltas=deltas)
    if error:summary['error']=error
    dump(ROOT/'summary.json',summary)
    dump(ROOT/'run_status.json',dict(status=status,exit_code=0 if status=='PASS' else 1,
         completed_points=len(results),expected_points=protocol.get('expected_points',12),physical_GPU=5,
         evaluation_complete=status=='PASS',error=error))
    lines=['# TCG Stage 1 fixed evaluation','',f'Status: {status}',
           f'Completed points: {len(results)} / {protocol.get("expected_points",12)}','',
           'Checkpoint steps: 0, 100. GPU: 5. QP: 0, 5, 9.',
           'BasketballDrive and BQTerrace; 64 frames each; native 1920x1080 metrics; IP=-1.',
           'Separate fresh RANS encode and independent decode for each checkpoint and point.',
           'TCG buffers are asserted clear at sequence and I/GOP boundaries.',
           'Protocol and formulas: protocol.json. Numerical details: metrics.csv and deltas.csv.','']
    if error:lines.extend(['Error: '+public(error),''])
    if results:
        lines.extend(['| Sequence | QP | Step | bpp | kbps | PSNR | MS-SSIM | LPIPS | DISTS | FloLPIPS |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for r in results:
            lines.append('| '+r['sequence']+' | '+str(r['QP'])+' | '+str(r['step'])+' | '+' | '.join(f'{float(r[k]):.9f}' for k in METRICS)+' |')
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    event('EVALUATION_FINISHED',status=status,completed_points=len(results))


if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    lock=(ROOT/'evaluate.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol={};results=[]
    try:
        protocol,videos,checkpoints=prepare()
        dump(ROOT/'run_status.json',dict(status='RUNNING',completed_points=0,expected_points=12,physical_GPU=5))
        event('PREFLIGHT_PASS',checkpoint_audit=True,split_audit=True)
        results=evaluate(protocol,videos,checkpoints)
        finish('PASS',protocol,results)
    except Exception as exc:
        # Preserve any point results completed before the failure.
        if (ROOT/'metrics.csv').exists():
            with (ROOT/'metrics.csv').open() as f:results=list(csv.DictReader(f))
        message=type(exc).__name__+': '+str(exc)
        status='BLOCKED' if isinstance(exc,FileNotFoundError) or 'CUDA' in message else 'FAILED'
        dump(ROOT/'error.json',dict(status=status,error=message,traceback=traceback.format_exc()))
        finish(status,protocol,results,message)
        traceback.print_exc();sys.exit(1)
