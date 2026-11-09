"""Final local media, artifact hashes and completion checks; never retrain."""
from common_s2 import *
import argparse

def ffmpeg_probe():
    dest=ROOT/'tmp/ffmpeg_filter_probe.mp4';font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    filters=f'[0:v]drawtext=fontfile={font}:text=frame_%{{n}}:fontsize=12[a];[a][1:v]xstack=inputs=2:layout=0_0|64_0[out]'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-f','lavfi','-i','color=red:s=64x64:r=4:d=1','-f','lavfi','-i','color=blue:s=64x64:r=4:d=1','-filter_complex_threads','1','-filter_complex',filters,'-map','[out]','-c:v','libx264','-threads','1',str(dest)],check=True)
    dump(ROOT/'audits/ffmpeg_compatibility.json',dict(status='PASS',xstack_explicit_layout=True,drawtext_frame_counter=True,probe_SHA256=sha(dest),time=time.time()))

def videos():
    import torch
    from videos import export
    from pipeline import wait_rows
    from cache_codec import cache_paths
    env();ffmpeg_probe();models=['GT','Original','A2000','B300','E','F'];all_entries=[]
    for profile in load(ROOT/'manifests/long_videos.json'):
        seq=profile['sequence'];row=dict(profile['row']);row.update(long_video=True,source_frame_indices=list(range(128)),canonical_indices=list(range(128)))
        entries=[]
        for g in models:
            pg='B' if g=='B300' else g
            source=PARENT_EXPERIMENT/'videos/manifests'/f'{seq}_{pg}.json'
            target=ROOT/'videos/manifests'/f'{seq}_{g}.json'
            if g in ['GT','Original','A2000','B300'] and source.exists():
                a=load(source);assert a['status']=='PASS' and sha(a['path'])==a['sha256']
                key,_,_=cache_paths(row,5,False);assert a['source_cache_identity']==key
                if g=='B300':assert a['step']==300
                dump(target,dict(a,model=g,reused_from=str(source),reused_SHA256=sha(source)))
            else:
                wait_rows([row],5,False);export(profile,g,0 if g!='F' else 2)
            entries.append(load(target))
        # Native lossless clips remain individual. Preview is a labeled downsize
        # grid only; fixed contiguous native ROIs are exported separately.
        font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        labels=[f'{e["model"]} step{e["step"]} QP5 source_frame_%{{n}}' for e in entries]
        filters=[f'[{i}:v]setpts=PTS-STARTPTS,scale=640:360,drawtext=fontfile={font}:text={label}:x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.7[v{i}]' for i,label in enumerate(labels)]
        filters+=[''.join(f'[v{i}]' for i in range(6))+'xstack=inputs=6:layout=0_0|640_0|1280_0|0_360|640_360|1280_360[out]']
        cmd=['ffmpeg','-v','error','-nostdin','-y']
        for e in entries:cmd+=['-i',e['path']]
        preview=ROOT/'videos'/f'{seq}_QP5_synchronized.mp4'
        subprocess.run(cmd+['-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[out]','-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(preview)],check=True)
        # Fixed native center ROI and48..79 contiguous source frames, identical
        # across models. Reuse an inherited preregistered ROI when available.
        roi=load(ROOT/'visual_ROIs.json')[seq]
        for e,label in zip(entries,labels):
            directory=ROOT/'videos/ROI'/seq/e['model'];directory.mkdir(parents=True,exist_ok=True)
            vf=f'select=between(n\\,48\\,79),crop=256:256:{roi["x"]}:{roi["y"]}'
            subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',e['path'],'-vf',vf,'-vsync','0','-frames:v','32','-threads','1',str(directory/'frame_%03d.png')],check=True)
            dump(directory/'identity.json',dict(model=e['model'],step=e['step'],QP=5,source_frame_indices=list(range(48,80)),ROI=roi,source_video_SHA256=e['sha256'],PNG_SHA256={p.name:sha(p) for p in sorted(directory.glob('*.png'))}))
        all_entries.append(dict(sequence=seq,preview=str(preview),preview_SHA256=sha(preview),native_lossless=entries,ROI=roi))
    dump(ROOT/'videos/manifest.json',dict(status='PASS',videos=all_entries,preview_excluded_from_metrics_and_rate=True))

def integrity():
    import torch
    from summarize import rows,should_stop
    env();rr=rows();checks=[];cp=[]
    for g in GROUPS:
        done=load(ROOT/g/'training/completion.json');step=done['committed_updates'];assert done['status']=='PASS' and step in [200,300]
        if step==200:assert should_stop(g)
        expected=list(range(1,step+1));receipts=sorted((ROOT/g/'training/receipts').glob('*.json'));assert [int(p.stem) for p in receipts]==expected
        for k in range(100,step+1,100):
            points=[a for a in rr if a['model']==g and a['step']==k];assert len(points)==36
            for a in points:
                b=next(b for b in rr if b['model']=='B300' and b['QP']==a['QP'] and b['sequence']==a['sequence'])
                for name in ['real_bytes','bpp','bitstream_SHA256','codec_trace_SHA256','source_RGB_SHA256']:assert a[name]==b[name],('CODEC_PAIR_IDENTITY',g,k,name)
            checks.append(dict(group=g,step=k,points=36,paired_bpp_stream_source_codec_trace_exact=True))
        p=Path(done['final_checkpoint']);saved=torch.load(p,map_location='cpu',weights_only=True);assert saved['committed_updates']==step and saved['B300_SHA256']==B300_SHA
        assert sum(t.numel() for t in saved['trainable_state'].values())==34774916
        assert all(int(s['step'])==step for s in saved['optimizer']['state'].values())
        cp.append(dict(group=g,path=str(p),SHA256=sha(p),additional_updates=step,parameters=34774916))
    assert len(list((ROOT/'baseline_detail').glob('*.json')))==36
    assert load(ROOT/'audits/step0_consistency.json')['status']=='PASS'
    assert all(load(ROOT/f'audits/engineering_{g}.json')['status']=='PASS' for g in GROUPS)
    assert load(ROOT/'videos/manifest.json')['status']=='PASS'
    assert sha(B300)==B300_SHA
    for n,h in load(ROOT/'source_manifest.json')['code'].items():assert sha(ROOT/n)==h,('SOURCE_HASH',n)
    dump(ROOT/'checkpoint_manifest.json',cp)
    result=dict(status='PASS',complete=True,training_groups=2,checks=checks,checkpoints=cp,B300_SHA256_verified=True,frozen_codec_and_bitstream_consistency='PASS',final_test_points=0,time=time.time())
    dump(ROOT/'final_integrity.json',result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--videos',action='store_true');a=p.parse_args();videos() if a.videos else integrity()
