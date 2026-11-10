"""Native FFV1 records and unscaled fixed foliage/texture ROI comparisons."""
from common_s2 import *
import argparse
MODELS=['GT','Original','B300_original_reference','B300_noLoRA_step0','G0_selected','G1_selected']

def ffmpeg_probe():
    import numpy as np
    from videos import open_lossless
    d=ROOT/'tmp/ffmpeg_probe';d.mkdir(parents=True,exist_ok=True);paths=[]
    for i in range(6):
        p=d/f'{i}.mkv';w=open_lossless(p,16,16,30);w.stdin.write(np.full((2,16,16,3),i*30,dtype=np.uint8).tobytes());w.stdin.close();assert w.wait()==0;paths.append(p)
    args=['ffmpeg','-v','error','-nostdin','-y']
    for p in paths:args+=['-i',str(p)]
    graph=';'.join(f'[{i}:v]setpts=PTS-STARTPTS[v{i}]' for i in range(6))+';' + ''.join(f'[v{i}]' for i in range(6))+'xstack=inputs=6:layout=0_0|16_0|0_16|16_16|0_32|16_32:shortest=1[out]'
    dest=d/'stack.mkv';subprocess.run(args+['-filter_complex_threads','1','-filter_complex',graph,'-map','[out]','-c:v','ffv1','-pix_fmt','bgr0','-threads','1',str(dest)],check=True)
    dump(ROOT/'audits/ffmpeg_compatibility.json',dict(status='PASS',filter=graph,unsupported_fill_option_omitted=True,SHA256=sha(dest),time=time.time()))

def export(profile,rank):
    env();import torch,numpy as np
    from cache_codec import cache_paths
    from pipeline import wait_rows
    from evaluation import Evaluator
    from experiment_model import reconstruct
    from model_context import assert_reset
    from videos import open_lossless
    row=dict(profile['row']);row.update(long_video=True,source_frame_indices=list(range(128)),canonical_indices=list(range(128)))
    wait_rows([row],False,5);key,p,rp=cache_paths(row,5,False)
    with lock('locks/lease_'+key+'.lock',blocking=True):
        assert sha(p)==load(rp)['cache_sha256'];d=torch.load(p,map_location='cpu',weights_only=True,mmap=True);info=d['info'];h,w=info['native_shape'];assert info['frames']==128
        seq=profile['sequence'];out=[];ev=Evaluator(rank)
        for model in MODELS:
            manifest=ROOT/f'media/manifests/{seq}_{model}.json'
            if manifest.exists():
                a=load(manifest);assert sha(a['path'])==a['SHA256'] and a['stream_SHA256']==info['stream_SHA256'];out.append(a);continue
            step=load(ROOT/'selection.json')['groups'][GROUPS[0] if model=='G0_selected' else GROUPS[1]]['step'] if model.endswith('_selected') else 300 if model=='B300_original_reference' else 0
            c=ev.context(model,step) if model not in ['GT','Original'] else None
            if c:assert_reset(c);before=tensor_hash(c['pm'].state_dict().items())
            oldname={'B300_original_reference':'B300','Original':'Original','GT':'GT'}.get(model);oldpath=EF_SOURCE/f'videos/manifests/{seq}_{oldname}.json' if oldname else None;old=load(oldpath) if oldpath and oldpath.exists() else None
            if old:
                assert old['status']=='PASS' and old['frames']==128 and old['external_QP']==5 and old['source_frame_indices']==list(range(128)) and old['codec_features_match'] and old['stream_SHA256']==info['stream_SHA256'] and old['stream_bytes']==info['real_bytes'] and sha(old['path'])==old['sha256'] and str(old['fps'])==str(row['source_fps'])
                dest=Path(old['path']);writer=None
            else:
                dest=ROOT/f'media/native/{seq}_QP5_{model}_step{step:04d}.mkv';temp=dest.with_name(dest.stem+'.partial.mkv');writer=open_lossless(temp,w,h,row['source_fps'])
            hashes=[];first=None
            with torch.no_grad():
                for i in range(128):
                    if model=='GT':rgb=d['GT'][i].numpy().tobytes()
                    else:
                        feature=d['features'][i];assert (None if i==0 else tensor_sha(feature))==info['codec_trace'][i]['codec_feature_SHA256']
                        raw=d['original'][i].to(rank) if i==0 or c is None else reconstruct(c,feature,info['actual_qps'][i]);rgb=(((raw.float().clamp(-1,1)+1)/2)[0,:,:h,:w].permute(1,2,0)*255).round().clamp(0,255).byte().cpu().numpy().tobytes()
                    rh=hashlib.sha256(rgb).hexdigest();hashes.append(rh)
                    if first is None:first=rgb
                    if old:assert old['RGB8_frame_hashes'][i]==rh,('REUSED_VIDEO_RGB_MISMATCH',seq,model,i)
                    else:writer.stdin.write(rgb)
            if writer:
                writer.stdin.close();assert writer.wait()==0;temp.replace(dest)
            decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(dest),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1']);assert decoded==first
            if c:assert tensor_hash(c['pm'].state_dict().items())==before;assert_reset(c)
            record=dict(status='PASS',sequence=seq,model=model,step=step,QP=5,frames=128,width=w,height=h,fps=row['source_fps'],source_frame_indices=list(range(128)),source_SHA256=row['source_sha256'],path=str(dest),SHA256=sha(dest),bytes=dest.stat().st_size,stream_SHA256=info['stream_SHA256'],stream_bytes=info['real_bytes'],codec_trace_SHA256=digest(info['codec_trace']),RGB8_frame_hashes=hashes,lossless='FFV1 bgr0, exact RGB24 first-frame roundtrip',preview_used_for_metrics=False,reused_from=str(oldpath) if old else None,checkpoint_SHA256=B300_SHA if model=='B300_original_reference' else sha(ROOT/'checkpoints/B300_noLoRA_step0.pt') if model=='B300_noLoRA_step0' else load(ROOT/'selection.json')['groups'][GROUPS[0] if model=='G0_selected' else GROUPS[1]]['checkpoint_SHA256'] if model.endswith('_selected') else None)
            dump(manifest,record);out.append(record)
            if c:ev.contexts.clear();del c;torch.cuda.empty_cache()
        return out

def montage(seq,records):
    import numpy as np
    from PIL import Image,ImageDraw
    roi=load(ROOT/'visual_ROIs.json')[seq];x,y,w,h=[roi[k] for k in ['x','y','width','height']];first,last=roi['first'],roi['last'];base=ROOT/'media/ROI';base.mkdir(parents=True,exist_ok=True);proof=base/f'{seq}.json'
    if proof.exists():a=load(proof);assert sha(a['path'])==a['SHA256'];return a
    args=['ffmpeg','-v','error','-nostdin','-y'];filters=[];expected=[]
    for i,r in enumerate(records):
        args+=['-i',r['path']];label=ROOT/f'tmp/roi_label_{i}.txt';label.write_text(f'{r["model"]} step {r["step"]} QP5\nsource_frame %{{n}} ROI ({x},{y},{w},{h})')
        filters.append(f'[{i}:v]trim=start_frame={first}:end_frame={last+1},setpts=PTS-STARTPTS,crop={w}:{h}:{x}:{y},pad={w}:{h+48}:0:48:black,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile={label}:fontcolor=white:fontsize=10:x=3:y=3:start_number={first}[v{i}]')
    layout='|'.join(f'{(i%2)*w}_{(i//2)*(h+48)}' for i in range(6));filters.append(''.join(f'[v{i}]' for i in range(6))+f'xstack=inputs=6:layout={layout}:shortest=1[out]')
    path=base/f'{seq}_QP5_unscaled_pixel_ROI_FFV1.mkv';temp=path.with_name(path.stem+'.partial.mkv');subprocess.run(args+['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2',str(temp)],check=True);temp.replace(path)
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1']);grid=np.frombuffer(raw,np.uint8).reshape((3*(h+48),2*w,3))
    canvas=Image.new('RGB',(w*5,6*(h+40)));draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(records):
        allraw=subprocess.check_output(['ffmpeg','-v','error','-i',r['path'],'-vf',f'trim=start_frame={first}:end_frame={last+1},crop={w}:{h}:{x}:{y}','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1']);frames=np.frombuffer(allraw,np.uint8).reshape((last-first+1,h,w,3))
        assert np.array_equal(frames[0],grid[(i//2)*(h+48)+48:(i//2+1)*(h+48),(i%2)*w:(i%2+1)*w]),'ROI_GRID_CHANGED_PIXELS'
        for j,index in enumerate([48,56,64,72,79]):
            ox=j*w;oy=i*(h+40);draw.text((ox+2,oy+2),f'{r["model"]} step{r["step"]} QP5 frame{index}',fill='white');canvas.paste(Image.fromarray(frames[index-first]),(ox,oy+40))
    png=base/f'{seq}_unscaled_selected_frames.png';canvas.save(png)
    a=dict(status='PASS',sequence=seq,path=str(path),SHA256=sha(path),frames=last-first+1,source_frames=[first,last],fps=records[0]['fps'],ROI=roi,coordinates_relative_to_original_frame=True,ROI_scale=1,native_pixels_unchanged=True,labels_in_separate_border=True,grid_model_order=MODELS,still_path=str(png),still_SHA256=sha(png));dump(proof,a);return a

def main():
    env();assert load(ROOT/'selection.json')['status']=='FROZEN';ffmpeg_probe();all_records=[];rois=[]
    for profile in load(ROOT/'manifests/long_videos.json'):
        rec=export(profile,0);all_records+=rec;rois.append(montage(profile['sequence'],rec))
        dump(ROOT/'audits/media_progress.json',dict(status='RUNNING',sequences=len(rois),native_clips=len(all_records),time=time.time()))
    dump(ROOT/'video_manifest.json',dict(status='PASS',native_clips=all_records,synchronized_unscaled_ROI=rois,subjective_blind_test_claimed=False,not_used_for_metrics=True,time=time.time()))
if __name__=='__main__':main()
