"""Viewing-only preview of the preselected first development clip at step100.

Independently decodes the already measured64-frame QP5 stream; no optimizer,
metric recomputation, checkpoint selection or training configuration changes.
"""
from pathlib import Path
import sys
EXPERIMENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXPERIMENT))
from common_s2 import *
import io
import traceback

OUT=ROOT/'preview_step100'
MODELS=['GT','Original','A2000','A','B','C','D']

def command(args):
    subprocess.run(args,check=True,stdin=subprocess.DEVNULL)

def label(group):
    return {'GT':'GT reference','Original':'Original GVC-RT','A2000':'A2000 s0.5'}.get(group,f'{group} +100 s0.5')

def render(data,proof,key,profile,group,rank):
    import torch
    from experiment_model import construct,restore
    from model_context import assert_reset
    from temporal import alignment_field
    from videos import open_lossless
    from evaluate import backend
    row=profile['row'];seq=profile['sequence'];h,w=proof['native_shape'];fps=row['source_fps'];step=100 if group in GROUPS else 0
    checkpoint=ROOT/group/'training/checkpoints/step_0100.pt' if group in GROUPS else INIT if group=='A2000' else None
    identity=dict(model=group,step=step,source_cache_identity=key,source_cache_SHA256=proof['cache_sha256'],checkpoint_SHA256=sha(checkpoint) if checkpoint else None,
        stream_SHA256=proof['stream_SHA256'],external_QP=5,source_frame_indices=row['source_frame_indices'],source_manifest_SHA256=sha(ROOT/'source_manifest.json'),preview_script_SHA256=sha(Path(__file__)),protocol_SHA256=sha(ROOT/'protocol.json'))
    receipt=OUT/'manifests'/f'{group}.json'
    if receipt.exists():
        saved=load(receipt);assert saved['identity']==identity
        assert sha(saved['lossless_path'])==saved['lossless_SHA256'] and sha(saved['preview_path'])==saved['preview_SHA256']
        return saved
    c=None;reference=[]
    if group!='GT':
        point=load(ROOT/f'evaluation/{group}/step_{step:04d}/points/{seq}_qp5.json')
        assert point['status']=='PASS' and point['result']['bitstream_SHA256']==proof['stream_SHA256'] and point['result']['real_bytes']==proof['real_bytes']
        for name,value in point['record_SHA256'].items():assert sha(ROOT/name)==value
        if checkpoint is not None:assert point['identity']['checkpoint_SHA256']==identity['checkpoint_SHA256']
        with (ROOT/f'evaluation/{group}/step_{step:04d}/frames/{seq}_qp5.csv').open() as f:reference=list(csv.DictReader(f))
        assert len(reference)==64
    if group not in ['GT','Original']:
        c=construct('A' if group=='A2000' else group,False,rank)
        if group in GROUPS:restore(c,torch.load(checkpoint,map_location='cpu',weights_only=True))
        assert c['optimizer'] is None
        assert_reset(c);before=tensor_hash(c['pm'].state_dict().items())
    backend(False,rank)
    from src.utils import stream_helper as sh
    reader=io.BytesIO(Path(proof['stream_path']).read_bytes());helper=sh.SPSHelper();idx=[0];device=torch.device('cuda',rank)
    if c is not None:
        original_add=c['pm'].add_ref_frame
        def fixed_reference(feature=None,frame=None,increase_poc=True):
            return original_add(feature,data['original'][idx[0]].to(device),increase_poc)
        c['pm'].add_ref_frame=fixed_reference
    kbps=proof['real_bytes']*8*float(__import__('fractions').Fraction(fps))/64/1000
    suffix=f'GT_reference' if group=='GT' else f'{group}_step{step}_QP5_stream{proof["real_bytes"]}B_{kbps:.1f}kbps'
    path=OUT/f'{seq}_{suffix}.mkv';preview=path.with_suffix('.mp4');writer=open_lossless(path,w,h,fps)
    frames=[];first_rgb=None;start=time.time();torch.cuda.reset_peak_memory_stats(rank)
    with torch.no_grad():
        for i in range(64):
            idx[0]=i;header=sh.read_header(reader)
            while header['nal_type']==sh.NalType.NAL_SPS:
                helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
            sps=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader)
            assert q==proof['actual_qps'][i] and (header['nal_type']==sh.NalType.NAL_I)==(i==0)
            source_hash=hashlib.sha256(data['GT'][i].numpy().tobytes()).hexdigest()
            assert source_hash==proof['codec_trace'][i]['source_RGB_SHA256']
            if group=='GT':
                rgb=data['GT'][i].numpy().tobytes();record=dict(frame=i,source_RGB_SHA256=source_hash)
            else:
                if c is None:
                    raw=data['original'][i];out=((raw.float().clamp(-1,1)+1)/2)[...,:h,:w]
                else:
                    if i==0:
                        raw=c['im'].decompress(payload,sps,q)['x_hat'];c['pm'].add_ref_frame(None,raw)
                    else:
                        gen=c['models']['generator'];gen.set_temporal_cascade_grad(False)
                        for name in ['tcg_low','tcg_mid','tcg_out']:
                            getattr(gen,name).alignment=alignment_field(data,i,device) if group in 'CD' else None
                        raw=c['pm'].decompress(payload,sps,q)['x_hat']
                        assert tensor_sha(c['pm'].dpb[0].feature)==proof['codec_trace'][i]['codec_feature_SHA256']
                        assert tensor_sha(c['pm'].dpb[0].frame)==tensor_sha(data['original'][i])
                    out=c['core'].unit(raw,h,w)
                raw_hash=tensor_sha(raw);out_hash=tensor_sha(out)
                assert raw_hash==reference[i]['reconstruction_SHA256'],('RAW_PREVIEW_MISMATCH',group,i)
                assert out_hash==reference[i]['RGB_tensor_SHA256'],('RGB_PREVIEW_MISMATCH',group,i)
                assert reference[i]['source_RGB_SHA256']==source_hash and int(reference[i]['actual_QP'])==q
                rgb=out[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().cpu().numpy().tobytes()
                record=dict(frame=i,source_RGB_SHA256=source_hash,reconstruction_SHA256=raw_hash,RGB_tensor_SHA256=out_hash,actual_QP=q)
            record['viewing_RGB8_SHA256']=hashlib.sha256(rgb).hexdigest();frames.append(record)
            if first_rgb is None:first_rgb=rgb
            writer.stdin.write(rgb)
            dump(OUT/'status.json',dict(status='RUNNING',PID=os.getpid(),model=group,frame=i,time=time.time(),training_updates=0))
    writer.stdin.close();assert writer.wait()==0 and reader.tell()==proof['real_bytes']
    first=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'])
    assert first==first_rgb,'LOSSLESS_RGB_COLOR_ROUNDTRIP_FAILED'
    if c is not None:
        assert tensor_hash(c['pm'].state_dict().items())==before
        assert_reset(c);del c
    torch.cuda.empty_cache()
    command(['ffmpeg','-v','error','-nostdin','-y','-i',str(path),'-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','libx264','-crf','16','-preset','medium','-threads','2','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv','-movflags','+faststart',str(preview)])
    write_csv(OUT/'frames'/f'{group}.csv',frames)
    saved=dict(status='PASS',identity=identity,model=group,label=label(group),frames=64,width=w,height=h,fps=fps,codec_bytes=proof['real_bytes'],codec_kbps=kbps,
        lossless_path=str(path),lossless_SHA256=sha(path),preview_path=str(preview),preview_SHA256=sha(preview),preview_lossy_viewing_only=True,
        frame_records=str(OUT/'frames'/f'{group}.csv'),matches_saved_measured_reconstruction_and_RGB_every_frame=group!='GT',lossless_RGB_roundtrip='PASS',weights_unchanged=True,
        training_updates=0,seconds=time.time()-start,physical_GPU=physical_gpu(rank),timing_interfered_by_existing_training=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(rank))
    dump(receipt,saved);return saved

def combined(entries,proof,seq,cropped=False):
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';args=['ffmpeg','-v','error','-nostdin','-y']
    for e in entries:args+=['-i',e['lossless_path']]
    pw,ph=(640,360) if cropped else (960,540)
    filters=[]
    for i,e in enumerate(entries):
        op='crop=640:360:(iw-640)/2:(ih-360)/2' if cropped else 'scale=960:540'
        text=e['label']+(' native center crop' if cropped else ' scaled overview')
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS,{op},drawtext=fontfile={font}:text={text}:x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7[v{i}]')
    layout='|'.join(f'{(i%4)*pw}_{(i//4)*ph}' for i in range(len(entries)))
    filters.append(''.join(f'[v{i}]' for i in range(len(entries)))+f'xstack=inputs={len(entries)}:layout={layout}:fill=black[out]')
    name='native_center_crop' if cropped else 'scaled_overview'
    file=OUT/f'{seq}_QP5_step100_stream{proof["real_bytes"]}B_{name}_comparison.mp4'
    command(args+['-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[out]','-c:v','libx264','-crf','16','-preset','medium','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(file)])
    image=file.with_suffix('.png')
    command(['ffmpeg','-v','error','-nostdin','-y','-i',str(file),'-vf','select=eq(n\\,32)','-frames:v','1',str(image)])
    return dict(path=str(file),SHA256=sha(file),keyframe=str(image),native_crop=cropped,panel_geometry=[pw,ph],frames=64)

def main():
    env();OUT.mkdir(exist_ok=True)
    with lock('locks/quick_preview_100.lock'):
        from freeze import verify
        assert verify()
        profile=load(ROOT/'manifests/quick_development.json')[0]
        assert profile['GT_category_rule']=='low_motion_high_texture'
        from cache_codec import cache_paths
        key,path,rp=cache_paths(profile['row'],5,False);proof=load(rp)
        assert proof['status']=='PASS' and sha(path)==proof['cache_sha256'] and sha(proof['stream_path'])==proof['stream_SHA256']
        import torch
        torch.set_num_threads(1);torch.cuda.set_device(0)
        data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        entries=[render(data,proof,key,profile,g,0) for g in MODELS]
        comparisons=[combined(entries,proof,profile['sequence'],crop) for crop in [False,True]]
        assert verify()
        dump(OUT/'manifest.json',dict(status='COMPLETE',purpose='Viewing-only intermediate checkpoint preview; not a new metric evaluation or checkpoint selection',
            sequence=profile['sequence'],GT_only_preselected_category=profile['GT_category_rule'],external_QP=5,frames=64,checkpoint_step=100,models=entries,comparisons=comparisons,
            training_updates=0,frozen_protocol_unchanged=True,physical_GPU=4,source_script_SHA256=sha(Path(__file__)),time=time.time()))
        dump(OUT/'status.json',dict(status='COMPLETE',PID=os.getpid(),frames=64,models=len(entries),manifest=str(OUT/'manifest.json'),training_updates=0,time=time.time()))

if __name__=='__main__':
    try:main()
    except Exception:
        env();dump(OUT/'status.json',dict(status='FAILED',traceback=traceback.format_exc(),PID=os.getpid(),time=time.time()));raise
