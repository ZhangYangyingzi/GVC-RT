"""Audit fixed task coverage and write public tables, figures, previews and report."""
import csv, hashlib, json, math, shutil
from pathlib import Path

P=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,rows):
    assert rows
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def interpolate(points,target,key):
    ordered=sorted(points,key=lambda p:p['actual_bpp'])
    pair=next(((a,b) for a,b in zip(ordered[:-1],ordered[1:])
               if a['actual_bpp']<=target<=b['actual_bpp']),None)
    assert pair is not None
    a,b=pair
    lo,hi=a['actual_bpp'],b['actual_bpp']
    assert lo<=target<=hi and hi>lo
    t=(target-lo)/(hi-lo)
    return a[key]+t*(b[key]-a[key])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    all_train=[]
    for branch in 'ABCD':
        path=P/f'training_{branch}.csv'
        rows=list(csv.DictReader(path.open()))
        assert len(rows)==300 and [int(r['step']) for r in rows]==list(range(1,301))
        assert all(int(r['optimizer_internal_step'])==int(r['step']) and float(r['parameter_delta'])>0 for r in rows)
        all_train+=rows
    write(P/'training_log.csv',all_train)
    items=[]
    for branch in 'ABCD':
        for step in (0,100,300):
            for source in ('UVal0','UVal1'):
                for qp in (2,7):
                    path=P/'server_artifacts/evaluation'/f'{branch}_s{step}_{source}_q{qp}.json'
                    assert path.is_file(),path
                    row=json.loads(path.read_text());assert row['status']=='PASS' and row['independent_decode']
                    assert sha(row['stream_path'])==row['stream_sha256']
                    items.append(row)
        for source in ('ParkScene','Jockey'):
            for qp in (2,7):
                path=P/'server_artifacts/evaluation'/f'{branch}_s300_{source}_q{qp}.json'
                assert path.is_file(),path
                row=json.loads(path.read_text());assert row['status']=='PASS' and row['independent_decode']
                assert sha(row['stream_path'])==row['stream_sha256']
                items.append(row)
    assert len(items)==64
    for path in sorted((P/'server_artifacts/evaluation').glob('*.json')):
        stem=path.stem
        if stem.endswith(('_q0','_q1','_q3','_q4','_q5','_q6','_q8','_q9')):
            row=json.loads(path.read_text());assert row['status']=='PASS' and row['step']==300
            assert sha(row['stream_path'])==row['stream_sha256']
            items.append(row)
    for branch in 'ABCD':
        for source in ('UVal0','UVal1','ParkScene','Jockey'):
            extra=[r for r in items if r['branch']==branch and r['sequence']==source and r['external_qp'] not in (2,7)]
            assert len(extra)<=2
    columns=('branch','step','sequence','dataset','external_qp','actual_bpp','kbps_20fps',
             'real_total_bytes','payload_bytes','header_bytes','estimated_P_main_latent_bits',
             'estimated_P_hyper_latent_bits','latent_rate_attribution','LPIPS','DISTS','FloLPIPS',
             'PSNR','encoding_seconds','decoding_seconds','peak_MiB','generator_parameters',
             'quant_parameters','condition_parameters','source_fps','sampling_fps','source_frame_indices',
             'FID','independent_decode','stream_sha256','checkpoint_sha256')
    write(P/'evaluation.csv',[{k:json.dumps(r[k]) if k=='source_frame_indices' else r.get(k,'') for k in columns} for r in items])
    compare=[]
    for source in ('UVal0','UVal1','ParkScene','Jockey'):
        for first,second in (('A','B'),('A','C'),('A','D'),('B','D'),('C','D')):
            a=[r for r in items if r['step']==300 and r['sequence']==source and r['branch']==first]
            b=[r for r in items if r['step']==300 and r['sequence']==source and r['branch']==second]
            assert len(a)>=2 and len(b)>=2
            low=max(min(x['actual_bpp'] for x in a),min(x['actual_bpp'] for x in b))
            high=min(max(x['actual_bpp'] for x in a),max(x['actual_bpp'] for x in b))
            if high<=low:
                compare.append({'sequence':source,'first':first,'second':second,'status':'no_overlap',
                                'overlap_low_bpp':low,'overlap_high_bpp':high,'method':'no_extrapolation'})
                continue
            target=(low+high)/2
            row={'sequence':source,'first':first,'second':second,'status':'overlap',
                 'overlap_low_bpp':low,'overlap_high_bpp':high,'comparison_bpp':target,
                 'method':'linear_interpolation_within_two_observed_QP_points; descriptive_only'}
            for metric in ('LPIPS','DISTS','FloLPIPS','PSNR'):
                av,bv=interpolate(a,target,metric),interpolate(b,target,metric)
                row[f'{metric}_first']=av;row[f'{metric}_second']=bv;row[f'delta_{metric}_second_minus_first']=bv-av
            compare.append(row)
    write(P/'matched_rate_comparisons.csv',compare)
    fig=P/'figures';fig.mkdir(exist_ok=True)
    plt.figure(figsize=(10,5))
    for branch in 'ABCD':
        rows=[r for r in all_train if r['branch']==branch]
        y=[float(r['loss']) for r in rows]
        smooth=[sum(y[max(0,i-9):i+1])/min(i+1,10) for i in range(len(y))]
        plt.plot(range(1,301),smooth,label=branch)
    plt.xlabel('Effective optimizer update');plt.ylabel('Training objective, trailing 10 updates')
    plt.legend();plt.tight_layout();plt.savefig(fig/'training_curve.png',dpi=150);plt.close()
    for source in ('UVal0','UVal1','ParkScene','Jockey'):
        plt.figure(figsize=(8,5))
        for branch in 'ABCD':
            rows=sorted((r for r in items if r['step']==300 and r['sequence']==source and r['branch']==branch),key=lambda r:r['actual_bpp'])
            plt.plot([r['actual_bpp'] for r in rows],[r['LPIPS'] for r in rows],'o-',label=branch)
        plt.xlabel('Actual full-stream bpp');plt.ylabel('LPIPS (lower is better)')
        plt.title(source+'; step 300; 64 native-I causal frames');plt.legend()
        plt.tight_layout();plt.savefig(fig/f'actual_rate_LPIPS_{source}.png',dpi=150);plt.close()
    preview=fig/'previews';preview.mkdir(exist_ok=True)
    for branch in 'ABCD':
        key=f'{branch}_s300_UVal0_q2'
        src=P/'previews'/f'{key}_frame15.png'
        assert src.is_file(),src
        shutil.copy2(src,preview/src.name)
        for suffix in ('quant','condition'):
            src=P/'previews'/f'{key}_{suffix}.png'
            if src.exists():shutil.copy2(src,preview/src.name)
        clip=P/'previews'/f'{key}_GT_left_reconstruction_right.mp4'
        if clip.exists():
            import cv2
            row=next(r for r in items if r['branch']==branch and r['step']==300 and r['sequence']=='UVal0' and r['external_qp']==2)
            reader=cv2.VideoCapture(str(clip))
            writer=cv2.VideoWriter(str(preview/clip.name),cv2.VideoWriter_fourcc(*'mp4v'),4,(512,256))
            assert reader.isOpened() and writer.isOpened()
            while True:
                ok,frame=reader.read()
                if not ok:break
                cv2.putText(frame,f'{branch} step 300  {row["actual_bpp"]:.5f} bpp',
                            (8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2,cv2.LINE_AA)
                writer.write(frame)
            reader.release();writer.release()
    gt=P/'previews/UVal0_gt_frame15.png';assert gt.is_file();shutil.copy2(gt,preview/gt.name)
    images=[]
    for branch in 'ABCD':
        key=f'{branch}_s300_UVal0_q2'
        image=Image.open(preview/f'{key}_frame15.png').convert('RGB')
        row=next(r for r in items if r['step']==300 and r['sequence']=='UVal0' and r['branch']==branch and r['external_qp']==2)
        tile=Image.new('RGB',(256,286),'white');tile.paste(image,(0,30))
        ImageDraw.Draw(tile).text((6,8),f'{branch} step 300  {row["actual_bpp"]:.5f} bpp',fill='black')
        images.append(tile)
    canvas=Image.new('RGB',(1024,286),'white')
    for i,image in enumerate(images):canvas.paste(image,(256*i,0))
    canvas.save(fig/'fixed_crop_A_B_C_D.png')
    artifacts=[]
    for branch in 'ABCD':
        for step in (0,100,300):
            path=P/'server_artifacts'/branch/f'step_{step}.pt'
            artifacts.append({'type':'checkpoint','branch':branch,'step':step,'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)})
    for r in items:
        artifacts.append({'type':'stream','branch':r['branch'],'step':r['step'],'sequence':r['sequence'],
                          'external_qp':r['external_qp'],'path':str(Path(r['stream_path']).resolve()),
                          'bytes':r['real_total_bytes'],'sha256':r['stream_sha256']})
    (P/'binary_artifacts_index.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    means={}
    for branch in 'ABCD':
        subset=[r for r in items if r['branch']==branch and r['step']==300 and r['sequence'] in ('UVal0','UVal1') and r['external_qp'] in (2,7)]
        means[branch]={k:sum(r[k] for r in subset)/len(subset) for k in ('actual_bpp','LPIPS','DISTS','FloLPIPS','PSNR')}
    synchronized=sum(r.get('all_64_reference_states_synchronized',False) for r in items)
    lines=['# Results','',
           '## Training and coding checks','',
           'Four branches completed 300 verified optimizer updates. Preflight identity, real-stream and gradient checks passed. Every listed evaluation decoded a persisted RANS file with freshly instantiated decoder models.',
           f'{synchronized}/{len(items)} evaluation tasks also compared all 64 encoder and independent-decoder causal reference states by hash. Earlier tasks ran before that extra full-sequence assertion was added; their independent decode and two-frame preflight checks passed.','',
           '## Fixed validation, step 300','',
            'The following means combine two validation sources and the two predeclared external QPs as descriptive summaries; each QP has its own actual rate. Supplemental QPs are excluded from this mean.','',
           '| Group | Mean actual bpp | LPIPS | DISTS | FloLPIPS | PSNR dB |','|---|---:|---:|---:|---:|---:|']
    for branch,v in means.items():
        lines.append(f'| {branch} | {v["actual_bpp"]:.6f} | {v["LPIPS"]:.5f} | {v["DISTS"]:.5f} | {v["FloLPIPS"]:.5f} | {v["PSNR"]:.3f} |')
    lines+=['',f'Matched-rate overlap exists in {sum(r["status"]=="overlap" for r in compare)}/{len(compare)} method pairs across the four sequences. `matched_rate_comparisons.csv` evaluates only the overlapping bpp interval in each individual sequence with descriptive linear interpolation. Same-QP means above do not demonstrate equal-rate advantage. Two points are not a formal BD-rate estimate.','',
            '### Matched-rate LPIPS differences at overlap midpoint','',
            'Positive values mean the second method has higher LPIPS at the displayed actual bpp.','',
            '| Sequence | Comparison | Bpp | ΔLPIPS second − first |','|---|---|---:|---:|']
    for source in ('UVal0','UVal1','ParkScene','Jockey'):
        for second in ('B','C','D'):
            r=next(x for x in compare if x['sequence']==source and x['first']=='A' and x['second']==second)
            if r['status']=='overlap':
                lines.append(f'| {source} | A vs {second} | {r["comparison_bpp"]:.6f} | {r["delta_LPIPS_second_minus_first"]:+.5f} |')
            else:lines.append(f'| {source} | A vs {second} | — | no overlap |')
    lines+=['',
            '## Descriptive cross-dataset checks, step 300','',
            '| Sequence | Group | QP | Actual bpp | LPIPS | DISTS | FloLPIPS | PSNR dB |','|---|---|---:|---:|---:|---:|---:|---:|']
    for source in ('ParkScene','Jockey'):
        for branch in 'ABCD':
            for qp in (2,7):
                r=next(x for x in items if x['sequence']==source and x['branch']==branch and x['external_qp']==qp)
                lines.append(f'| {source} | {branch} | {qp} | {r["actual_bpp"]:.6f} | {r["LPIPS"]:.5f} | {r["DISTS"]:.5f} | {r["FloLPIPS"]:.5f} | {r["PSNR"]:.3f} |')
    lines+=['',
            '## Data and measurement limits','',
            '- This is a 300-update trend run with two validation source identities; no convergence or statistical significance claim is made.',
            '- ParkScene and Jockey are fixed descriptive checks only and did not choose checkpoints or training settings.',
            '- The inherited G_basic_100 checkpoint has mixed U-Long/UVG historical exposure. Clean historical nonexposure of evaluation datasets cannot be established.',
            '- Real total bytes, payload bytes and headers are exact. The RANS payload jointly encodes y and z, so exact separate y/z byte counts are unavailable; the table marks analytical P-only component rates as estimates.',
            '- FID is not computed.','']
    (P/'REPORT.md').write_text('\n'.join(lines))
    print('FINALIZED',len(items),len(compare),flush=True)


if __name__=='__main__':main()
