import argparse
import hashlib
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from diag_utils import *
from sources import decode,FILTER
from media import writer,finish,panel,probe

def smoke():
    s=samples()[0];assert s['dataset']=='uvg'
    text=(ROOT/'logs/ffmpeg_scale_configuration.txt').read_text()
    print(text,flush=True);print('FILTER CONFIGURATION',FILTER,flush=True)
    a,ha,ca=decode(s,False,1);b,hb,cb=decode(s,True,1)
    expected=old.load(f'parts/source_uvg_original_{s["video_index"]}.json')['ffmpeg_command']
    assert ca[ca.index('-frames:v')+1]=='1' and b.shape==(1,1080,1920,3)
    for name,value in [('CURRENT',a),('BT709_EXPLICIT',b)]:
        Image.fromarray(value[0],'RGB').save(ROOT/'color_conversion'/f'smoke_frame0_{name}.png')
    dump('conversion_smoke.json',dict(status='PASS',filter=FILTER,current_command=ca,explicit_command=cb,
        source_reference_command=expected,input_range='limited',input_matrix='BT.709 explicit / CURRENT ffmpeg default',
        output_range='full',output_pixel_format='rgb24',current_sha256=ha,explicit_sha256=hb,
        mean_absolute_RGB_difference=float(np.abs(a.astype(float)-b).mean()/255),
        max_RGB_difference=float(np.abs(a.astype(float)-b).max()/255)))
    print('ONE FRAME CONVERSION SMOKE PASS',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke-only',action='store_true');args=p.parse_args()
    if args.smoke_only:smoke();return
    assert load('conversion_smoke.json')['status']=='PASS'
    records=[]
    for s in samples():
        if s['dataset']!='uvg':continue
        a,ha,ca=decode(s,False,96,rawpath(s,'source'))
        b,hb,cb=decode(s,True,96,rawpath(s,'source',smoke=True))
        expected=old.load(f'parts/source_uvg_original_{s["video_index"]}.json')
        assert ca==expected['ffmpeg_command'] and ha==expected['source_hash']
        for conv,h,cmd in [('current',ha,ca),('709',hb,cb)]:
            dump(f'parts/source_{tag(s)}_{conv}.json',dict(status='PASS',RGB24_sha256=h,command=cmd,num_frames=96))
        paths=[];commands=[];rgb_sum=np.zeros((2,3));diff_sum=np.zeros(3);max_diff=np.zeros(3)
        dests=[ROOT/'color_conversion'/f'{s["name"]}_{suf}.mp4' for suf in ('BT601_current','BT709_explicit','BT601_vs_BT709')]
        processes=[]
        for i,dest in enumerate(dests):
            proc,cmd=writer(dest,1920,1080 if i<2 else 600,s['video']['fps'])
            processes.append(proc);commands.append(cmd)
        keys=[];sheets=ROOT/'contact_sheets/color_conversion'/s['name'];sheets.mkdir(parents=True,exist_ok=True)
        for i in range(96):
            processes[0].stdin.write(a[i].tobytes());processes[1].stdin.write(b[i].tobytes())
            pair=Image.new('RGB',(1920,600))
            pair.paste(panel(a[i],'CURRENT / ffmpeg default BT.601',960),(0,0))
            pair.paste(panel(b[i],'BT709_EXPLICIT / limited to full RGB',960),(960,0))
            processes[2].stdin.write(pair.tobytes())
            if i in (0,24,48,95):
                pair.save(sheets/f'frame_{i:03d}.png')
                Image.fromarray(a[i],'RGB').save(sheets/f'frame_{i:03d}_CURRENT.png')
                Image.fromarray(b[i],'RGB').save(sheets/f'frame_{i:03d}_BT709.png')
                keys.append(pair.resize((960,300),Image.Resampling.LANCZOS))
            rgb_sum[0]+=a[i].sum((0,1),dtype=np.float64)/255;rgb_sum[1]+=b[i].sum((0,1),dtype=np.float64)/255
            delta=np.abs(a[i].astype(np.int16)-b[i].astype(np.int16))
            diff_sum+=delta.sum((0,1),dtype=np.float64)/255;max_diff=np.maximum(max_diff,delta.max((0,1))/255)
        for proc in processes:finish(proc)
        sheet=Image.new('RGB',(1920,600))
        for i,img in enumerate(keys):sheet.paste(img,((i%2)*960,(i//2)*300))
        sheet.save(sheets/'contact_sheet.png')
        probes=[probe(d) for d in dests];assert all(int(v['nb_read_frames'])==96 for v in probes)
        pixels=96*1080*1920
        records.append(dict(sequence=s['name'],status='PASS',current_command=ca,explicit_command=cb,
            current_input_matrix='ffmpeg auto/default, recorded BT.601; exact old command and RGB hash verified',
            explicit_input_matrix='BT.709',input_range='limited',output_range='full',output_pixel_format='rgb24',
            filter_configuration=FILTER,mean_RGB_CURRENT=(rgb_sum[0]/pixels).tolist(),
            mean_RGB_BT709=(rgb_sum[1]/pixels).tolist(),mean_absolute_RGB_difference=(diff_sum/pixels).tolist(),
            max_RGB_difference=max_diff.tolist(),units='RGB [0,1]',CURRENT_RGB_sha256=ha,BT709_RGB_sha256=hb,
            current_matches_frozen_experiment=True,videos=[str(d) for d in dests],video_encoding_commands=commands,probes=probes))
        del a,b
        print('COLOR CONVERSION PASS',s['name'],flush=True)
    dump('uvg_color_conversion_audit.json',dict(status='PASS',one_frame_smoke_pass=True,conversions=records,
        interpretation='BT.709 is an explicit diagnostic assumption, not a claim about untagged raw-file color metadata'))

if __name__=='__main__':main()
