"""Four preselected native128-frame development demonstrations, local only."""
from common_s2 import *
from cache_codec import cache_paths
from experiment_model import construct,restore
from model_context import assert_reset
import argparse,io

def open_lossless(path,w,h,fps):
    path.parent.mkdir(parents=True,exist_ok=True)
    return subprocess.Popen(['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',str(fps),'-i','pipe:0','-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2',str(path)],stdin=subprocess.PIPE,stderr=(ROOT/'logs/video_encode_errors.log').open('ab'))

def export(profile,group,rank):
    import torch
    env();row=dict(profile['row']);row.update(long_video=True,source_frame_indices=list(range(128)),canonical_indices=list(range(128)))
    key,path,rp=cache_paths(row,5,False);proof=load(rp);assert sha(path)==proof['cache_sha256'];data=torch.load(path,map_location='cpu',weights_only=True,mmap=True);info=data['info'];h,w=info['native_shape']
    seq=profile['sequence'];step=0 if group in ['GT','Original','A2000','B300'] else load(ROOT/group/'training/completion.json')['committed_updates']
    dest=ROOT/'videos'/f'{seq}_QP5_{group}_step{step}_bytes{info["real_bytes"]}.mkv';manifest=ROOT/'videos/manifests'/f'{seq}_{group}.json'
    if manifest.exists():
        a=load(manifest);assert sha(a['path'])==a['sha256'] and a['source_cache_identity']==key;return
    c=None
    if group not in ['GT','Original']:
        c=construct(group,False,rank)
        if group not in ['A2000','B300']:restore(c,torch.load(ROOT/group/f'training/checkpoints/step_{step:04d}.pt',map_location='cpu',weights_only=True))
        assert_reset(c);before=tensor_hash(c['pm'].state_dict().items())
    from model_context import backend
    backend(False,rank)
    from src.utils import stream_helper as sh
    reader=io.BytesIO(Path(info['stream_path']).read_bytes());helper=sh.SPSHelper();idx=[0]
    if c is not None:
        old_add=c['pm'].add_ref_frame
        def fixed(feature=None,frame=None,increase_poc=True):return old_add(feature,data['original'][idx[0]].to(rank),increase_poc)
        c['pm'].add_ref_frame=fixed
    writer=open_lossless(dest,w,h,row['source_fps']);first_rgb=None;frames=[];start=time.time()
    with torch.no_grad():
        for i in range(128):
            idx[0]=i;header=sh.read_header(reader)
            while header['nal_type']==sh.NalType.NAL_SPS:helper.add_sps_by_id(sh.read_sps_remaining(reader,header['sps_id']));header=sh.read_header(reader)
            sps=helper.get_sps_by_id(header['sps_id']);q,payload=sh.read_ip_remaining(reader);assert q==info['actual_qps'][i]
            if group=='GT':rgb=data['GT'][i].numpy().tobytes()
            elif group=='Original':rgb=((data['original'][i].float().clamp(-1,1)+1)/2)[0,:,:h,:w].permute(1,2,0).mul(255).round().clamp(0,255).byte().numpy().tobytes()
            else:
                if i==0:raw=c['im'].decompress(payload,sps,q)['x_hat'];c['pm'].add_ref_frame(None,raw)
                else:
                    for n in ['tcg_low','tcg_mid','tcg_out']:getattr(c['models']['generator'],n).alignment=__import__('temporal').alignment_field(data,i,rank) if False else None
                    raw=c['pm'].decompress(payload,sps,q)['x_hat'];assert tensor_sha(c['pm'].dpb[0].feature)==info['codec_trace'][i]['codec_feature_SHA256']
                out=c['core'].unit(raw,h,w);rgb=out[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().cpu().numpy().tobytes()
            if first_rgb is None:first_rgb=rgb
            writer.stdin.write(rgb);frames.append(hashlib.sha256(rgb).hexdigest())
    writer.stdin.close();assert writer.wait()==0;assert reader.tell()==info['real_bytes']
    # Verify lossless RGB channel order, avoiding implicit YUV interpretation.
    first=subprocess.check_output(['ffmpeg','-v','error','-i',str(dest),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'])
    assert first==first_rgb,'LOSSLESS_RGB_ROUNDTRIP_COLOR_MISMATCH'
    if c is not None:assert tensor_hash(c['pm'].state_dict().items())==before;assert_reset(c)
    preview=dest.with_suffix('.mp4')
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(dest),'-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','libx264','-crf','16','-preset','medium','-threads','2','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv','-movflags','+faststart',str(preview)],check=True)
    dump(manifest,dict(status='PASS',sequence=seq,model=group,step=step,frames=128,source_frame_indices=list(range(128)),external_QP=5,s=.5 if group not in ['GT','Original'] else None,
        source_cache_identity=key,path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,preview_path=str(preview),preview_sha256=sha(preview),preview_lossy_viewing_only=True,
        pixel_format='input RGB24 / lossless FFV1 bgr0; exact decoded RGB24 verified',RGB8_frame_hashes=frames,quantization='float metrics unaffected; videos rounded to8-bit',fps=row['source_fps'],stream_SHA256=info['stream_SHA256'],stream_bytes=info['real_bytes'],
        codec_features_match=True,seconds=time.time()-start,alignment_cache_generation_wall_seconds_including_RANS_and_GT_RAFT=info['original_base_and_RAFT_seconds'] if group in 'CD' else 0.,physical_GPU=physical_gpu(rank)))

def montage():
    env();font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';models=['GT','Original','A2000','A','B','C','D'];all_rows=[]
    for profile in load(ROOT/'manifests/long_videos.json'):
        seq=profile['sequence'];entries=[load(ROOT/'videos/manifests'/f'{seq}_{g}.json') for g in models]
        cmd=['ffmpeg','-v','error','-nostdin','-y']
        for a in entries:cmd+=['-i',a['path']]
        labels=[f'{a["model"]} step{a["step"]} QP5' for a in entries]
        filters=[f'[{i}:v]setpts=PTS-STARTPTS,scale=960:540,drawtext=fontfile={font}:text={label}:x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.75[v{i}]' for i,label in enumerate(labels)]
        filters+=[''.join(f'[v{i}]' for i in range(7))+'xstack=inputs=7:layout=0_0|960_0|1920_0|2880_0|0_540|960_540|1920_540:fill=black[out]']
        joined=ROOT/'videos'/f'{seq}_QP5_synchronized_lossless_view960x540.mkv'
        subprocess.run(cmd+['-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[out]','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2',str(joined)],check=True)
        preview=joined.with_suffix('.mp4')
        subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(joined),'-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','libx264','-crf','16','-preset','medium','-threads','2','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-movflags','+faststart',str(preview)],check=True)
        all_rows.append(dict(sequence=seq,GT_category_rule=profile['GT_category_rule'],lossless=str(joined),sha256=sha(joined),preview=str(preview),preview_sha256=sha(preview),individual_native_resolution_lossless=[a['path'] for a in entries],panel_resize=[960,540],frames=128))
        # Fixed temporal center-line and fixed frame64; no model-quality selection.
        import cv2,numpy as np
        from PIL import Image,ImageDraw
        slices=[];keys=[]
        for a in entries:
            cap=cv2.VideoCapture(a['path']);lines=[];fixed=None
            for i in range(128):
                ok,bgr=cap.read();assert ok;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);lines.append(rgb[rgb.shape[0]//2].copy())
                if i==64:fixed=cv2.resize(rgb,(480,270),interpolation=cv2.INTER_AREA)
            cap.release();slices.append(np.stack(lines));keys.append(fixed)
        out=ROOT/'plots/keyframes';out.mkdir(exist_ok=True)
        canvas=Image.new('RGB',(480*4,300*2),'black');draw=ImageDraw.Draw(canvas)
        for i,(image,label) in enumerate(zip(keys,labels)):
            x=(i%4)*480;y=(i//4)*300;canvas.paste(Image.fromarray(image),(x,y+25));draw.text((x+4,y+4),label,fill='white')
        canvas.save(out/f'{seq}_frame64.png')
        strip=Image.new('RGB',(1920,150*7),'black');draw=ImageDraw.Draw(strip)
        for i,(sl,label) in enumerate(zip(slices,labels)):
            strip.paste(Image.fromarray(sl).resize((1920,128)),(0,i*150+20));draw.text((4,i*150+3),label+'; fixed native center row,128frames',fill='white')
        strip.save(out/f'{seq}_center_line_time_slice.png')
    dump(ROOT/'videos/manifest.json',dict(status='PASS',videos=all_rows,all_videos_local_only=True,GT_categories_fixed_before_training=True))
    (ROOT/'videos/README.md').write_text('Local-only128-frame development videos. Individual FFV1 clips retain native geometry; synchronized grids use960×540 panels and fixed labels. H264 previews are for viewing only. All neural metrics use float32 decoded tensors. See manifest.json for hashes and exact paths.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group');p.add_argument('--rank',type=int);p.add_argument('--montage',action='store_true');a=p.parse_args();env()
    if a.montage:montage()
    else:
        for profile in load(ROOT/'manifests/long_videos.json'):export(profile,a.group,a.rank)
