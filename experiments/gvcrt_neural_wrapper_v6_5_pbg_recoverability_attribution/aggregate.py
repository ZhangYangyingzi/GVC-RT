"""Numerical tables and fixed-frame montages; no interpretation."""
import math,statistics,traceback
from collections import defaultdict
from v65_io import *

def stats(values):
    vals=[float(x) for x in values if x not in (None,'')]
    assert all(math.isfinite(x) for x in vals)
    return dict(mean=statistics.mean(vals) if vals else None,median=statistics.median(vals) if vals else None,std=statistics.pstdev(vals) if vals else None,n=len(vals))

def main():
    cfg=frozen(True);raw=[];frames=[];checks=[];drift=[];effects=[];proxy=[];pv=[];methodhash={}
    for v in videos():
        own=[]
        for method in METHODS:
            for q in QPS:
                r=load(point(v,method,q));assert r['status']=='PASS'
                for key in ('bitstream','feature','frame_metrics','transitions'):assert sha(r[key+'_path'])==r[key+'_sha256']
                assert r['source_rgb_sha256']==v['rgb_sha256'] and r['source_sha256']==v['source_sha256'] and r['source_frame_indices']==v['source_frame_indices']
                assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:64]
                assert all(r[k] for k in ('real_RANS','independent_decode_pass','state_sync_pass'))
                assert r['compression_hash_before']==r['compression_hash_after']==cfg['compression_hash']
                assert all(math.isfinite(r[k]) for k in (*METRICS,'kbps','bpp'))
                hashes={k:r[k] for k in ('wrapper_hash','bridge_hash','generator_hash')};hashes['compression_core_hash']=r['compression_hash_before']
                hashes['wrapper_kind']='identity' if METHODS[method][0] is None else 'learned'
                if method in methodhash:assert methodhash[method]==hashes
                methodhash[method]=hashes;raw.append(r);own.append(r)
                fr=read(r['frame_metrics_path']);assert len(fr)==64 and [int(x['frame']) for x in fr]==list(range(64))
                assert sum(int(x['real_bits']) for x in fr)==8*r['real_bytes']==8*r['bytes_consumed']
                for x in fr:
                    assert all(math.isfinite(float(x[k])) for k in METRICS if k!='FloLPIPS')
                    frames.append(dict(dataset=v['dataset'],sequence=v['name'],video_index=v['video_index'],method=method,QP=q,**x))
                for metric in ('DISTS','LPIPS','PSNR','SSIM'):
                    vals=[float(x[metric]) for x in fr];first=statistics.mean(vals[:16]);last=statistics.mean(vals[-16:]);mean=statistics.mean(vals)
                    slope=sum((i-31.5)*(z-mean) for i,z in enumerate(vals))/sum((i-31.5)**2 for i in range(64))
                    drift.append(dict(dataset=v['dataset'],sequence=v['name'],video_index=v['video_index'],method=method,QP=q,metric=metric,first_quarter_mean=first,last_quarter_mean=last,last_minus_first=last-first,linear_slope=slope))
        checks.extend(bitstream_checks(v,own))
        for q in QPS:
            lookup={r['method']:r for r in own if r['QP']==q}
            for theta in ('V62','V64'):
                for metric in METRICS:
                    base,p,b,full=[lookup[m][metric] for m in ('M00_original',theta+'_P_only',theta+'_BG_only',theta+'_full')]
                    differences=dict(P_effect_originalBG=p-base,BG_effect_identityP=b-base,BG_effect_learnedP=full-p,P_effect_adaptedBG=full-b,interaction=full-p-b+base)
                    for effect,value in differences.items():effects.append(dict(dataset=v['dataset'],sequence=v['name'],video_index=v['video_index'],QP=q,theta=theta,metric=metric,metric_direction='lower_is_better' if metric in ('LPIPS','DISTS','FloLPIPS') else 'higher_is_better',effect=effect,value=value))
        for theta in ('V62','V64'):
            p=load(ROOT/'parts/proxy'/f'{sid(v)}_{theta}.json');assert p['status']=='PASS' and p['source_rgb_sha256']==v['rgb_sha256'] and sha(p['csv'])==p['csv_sha256']
            rows=read(p['csv']);assert len(rows)==64;proxy.extend(rows)
            meta=dict(dataset=v['dataset'],sequence=v['name'],video_index=v['video_index'],theta=theta)
            numeric=[k for k,z in rows[0].items() if k not in ('frame','video_index') and z not in ('',None) and _numeric(z)]
            numeric+= [k for k,z in rows[1].items() if k not in numeric and ('_change' in k or k.startswith(('temporal_RGB','flow_magnitude','compensated_residual'))) and z not in ('',None) and _numeric(z)]
            record=dict(meta)
            for key in numeric:
                s=stats([x.get(key) for x in rows]);record[key]=s['mean'];record[key+'_valid_frames']=s['n']
            for label in ('x','proxy'):record['camera_valid_pairs_'+label]=sum(x.get('camera_status_'+label)=='PASS' for x in rows)
            pv.append(record)
    write(ROOT/'audits/bitstream_factorial_audit.csv',checks);assert len(checks)==180 and all(x['status']=='PASS' for x in checks)
    assert len(raw)==315 and len(frames)==20160 and len(proxy)==1920 and len(pv)==30
    for theta in ('V62','V64'):
        assert methodhash[theta+'_P_only']['wrapper_hash']==methodhash[theta+'_full']['wrapper_hash']
        for key in ('bridge_hash','generator_hash'):
            assert methodhash['M00_original'][key]==methodhash[theta+'_P_only'][key]
            assert methodhash[theta+'_BG_only'][key]==methodhash[theta+'_full'][key]
    dump(ROOT/'audits/method_factorization.json',dict(status='PASS',methods=methodhash,receiver_hash_dtype='loaded float16',wrapper_hash_dtype='float32',identity_wrapper_hash=None))
    groups=defaultdict(list)
    for r in effects:groups[tuple(r[k] for k in ('dataset','theta','QP','metric','metric_direction','effect'))].append(r['value'])
    es=[dict(zip(('dataset','theta','QP','metric','metric_direction','effect'),key),**stats(vals)) for key,vals in groups.items()]
    ps=[]
    for dataset in ('uvg','ulong'):
        for theta in ('V62','V64'):
            selected=[r for r in pv if r['dataset']==dataset and r['theta']==theta]
            keys=sorted({k for r in selected for k,z in r.items() if isinstance(z,(int,float)) and k!='video_index' and not k.endswith('_valid_frames') and not k.startswith('camera_valid')})
            for key in keys:ps.append(dict(dataset=dataset,theta=theta,metric=key,aggregation_unit='video_mean',**stats([r.get(key) for r in selected])))
    for name,rows in [('factorial_raw',raw),('factorial_per_frame',frames),('temporal_drift',drift),('factorial_effects_per_video',effects),('factorial_effects_dataset_summary',es),('proxy_per_frame',proxy),('proxy_per_video',pv),('proxy_dataset_summary',ps)]:write(ROOT/'results'/f'{name}.csv',rows)
    write(ROOT/'results/ulong_vs_uvg_attribution.csv',[dict(part='proxy',**r) for r in ps]+[dict(part='factorial',**r) for r in es])
    from PIL import Image,ImageDraw
    montage_count=0
    for v in videos():
        if not chosen(v):continue
        for q in QPS:
            for i in visual_indices(v):
                d=ROOT/'visualizations/proxy'/sid(v)/f'frame_{i:06d}'
                panels=[(name,d/(name+'.png')) for name in ('GT','V62','V64','abs_V62','abs_V64')]
                panels += [(m,ROOT/'visualizations/recon'/v['dataset']/v['name']/m/f'qp{q}'/f'frame_{i:06d}.png') for m in METHODS]
                canvas=Image.new('RGB',(1920*3,(1080+28)*4),'white');draw=ImageDraw.Draw(canvas)
                for n,(label,path) in enumerate(panels):
                    with Image.open(path) as im:assert im.size==(1920,1080);canvas.paste(im,((n%3)*1920,(n//3)*1108+28))
                    draw.text(((n%3)*1920+8,(n//3)*1108+6),label,fill='black')
                out=ROOT/'visualizations/montages'/v['dataset']/v['name']/f'qp{q}_frame{i:06d}.png';out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out);montage_count+=1
    assert montage_count==84
    frozen(True)
    dump(ROOT/'final_integrity.json',dict(status='PASS',no_training=True,no_old_experiments_modified=True,checkpoint_sha256_pass=True,compression_core_identical=True,all_expected_runs_complete=True,all_metrics_finite=True,real_RANS=True,independent_decode=True,bitstream_factorial_audit_pass=True,all_source_frames_match_existing_manifest=True,factorial_points=len(raw),proxy_frames=len(proxy),montages=montage_count,undefined_diagnostic_policy='Explicit empty cells and status; excluded from means with valid counts',finished_unix=time.time()))
    print('AGGREGATE PASS',flush=True)

def _numeric(x):
    try:float(x);return True
    except (ValueError,TypeError):return False

if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'final_integrity.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
