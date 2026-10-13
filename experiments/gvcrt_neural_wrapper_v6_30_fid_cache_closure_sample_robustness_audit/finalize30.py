"""Mechanical aggregation, six fixed report sections, and completion gate."""
from audit_io import *
from run_all import tasks,gts,cpu_task
import numpy as np
def closure_report():
    vs={d:sources(d) for d in DATASETS};ts=tasks('closure',vs);fr=[];pr=[];gt=[];missing=[]
    for t in gts(vs):
        if not valid(t):missing.append(task_id(t));continue
        r=load(output(t));gt.append(dict(task=task_id(t),status=r['status'],sealed_GT_path=r['sealed_GT_path'],sealed_GT_sha256=r['sealed_GT_sha256']))
        fr.extend(dict(dataset=t['dataset'],sequence=next(v['name'] for v in vs[t['dataset']] if v['video_index']==t['video']),method='GT',QP=-1,**x) for x in r['records'])
    for t in ts:
        if not valid(t):missing.append(task_id(t));continue
        r=load(output(t));fr.extend(r['records']);pr.append(dict(dataset=t['dataset'],video=t['video'],method=t['method'],QP=t['qp'],status=r['status'],frames=64,bytes_consumed=r['bytes_consumed'],reconstruction_hash_match=r['reconstruction_hash_match'],max_relative_L2=max(x['relative_L2'] for x in r['records']),max_abs=max(x['max_abs'] for x in r['records']),failed_UIDs=[x['UID'] for x in r['records'] if not x['PASS']],reason=r['reason'],fresh_feature_sha256=r['fresh_feature_sha256'],bitstream_sha256=r['bitstream_sha256']))
    pooled=[]
    cs=[cpu_task(d,d+'_old64',[v['video_index'] for v in vs[d]],'closure_pool') for d in DATASETS]
    cs.append(cpu_task('mcljcv','mcl29_old64',[v['video_index'] for v in vs['mcljcv'] if v['name']!='videoSRC17_1920x1080_24'],'closure_pool'))
    for t in cs:
        if not valid(t):missing.append(task_id(t));continue
        pooled.extend(load(load(output(t))['statistics_path'])['closure'])
    passed=len(pr)==420 and len(gt)==35 and len(pooled)==36 and not missing and all(x['status']=='PASS' for x in pr+gt) and all(x['PASS'] for x in pooled)
    write(ROOT/'cache_closure_per_frame.csv',fr,fields=None if fr else ['UID','status','reason'])
    write(ROOT/'cache_closure_per_point.csv',pr,fields=None if pr else ['dataset','method','QP','status'])
    write(ROOT/'pooled_fid_fresh_vs_cached.csv',pooled,fields=None if pooled else ['dataset','method','QP','PASS'])
    dump(ROOT/'audits/cache_closure_integrity.json',dict(status='PASS' if passed else 'FAIL',expected=420,completed=len(pr),GT_completed=len(gt),pooled_expected=36,pooled_completed=len(pooled),missing=missing,failed_points=[x for x in pr if x['status']!='PASS'],failed_GT=[x for x in gt if x['status']!='PASS'],failed_pooled=[x for x in pooled if not x['PASS']],max_relative_L2=max((x['relative_L2'] for x in fr),default=None),max_abs=max((x['max_abs'] for x in fr),default=None),feature_row_failures=sum(not x['PASS'] for x in fr),historical_cache_modified=False))
    return passed
def statistics_results():
    out=[]
    for p in sorted((ROOT/'tasks').glob('cpu_*.json')):
        r=load(p)
        if valid(r['task']):out.append(load(r['statistics_path']))
    return out
def finalize(blocked=False):
    closure_report();audit=load(ROOT/'audits/cache_closure_integrity.json');vs={d:sources(d) for d in DATASETS}
    allstats=statistics_results();pooled=[];comparisons=[];grids=[];sensitivity=[];missing=list(audit['missing']);newpoints=[];metriccomp=[];metricgrids=[];subset_count=0
    from statistics_worker import grid_compare,COMPARISONS
    for s in allstats:
        pooled.extend(s['pooled'])
        if s['kind']=='subset':
            subset_count+=1
            for q in QPS:
                a=next(r for r in s['pooled'] if r['method']=='A_continue_1000' and r['QP']==q);b=next(r for r in s['pooled'] if r['method']=='B_covstats_1000' and r['QP']==q)
                sensitivity.append(dict(pool=s['pool'],dataset=a['dataset'],videos=s['videos'],QP=q,type='same-QP raw diagnostic, not equal-rate',A_FID=a['FID'],B_FID=b['FID'],delta=b['FID']-a['FID'],relative_percent=100*(b['FID']-a['FID'])/a['FID'],A_bpp=a['bpp'],B_bpp=b['bpp'],actual_bpp_equal=a['bpp']==b['bpp'],samples=a['samples']))
            sensitivity.extend(dict(**r,videos=s['videos'],type='3-QP equal-rate diagnostic') for r in s['comparisons'])
        else:comparisons.extend(s['comparisons']);grids.extend(s['grids'])
    if not blocked:
        nvs={d:[v for v in sources(d,True) if v['status']=='AVAILABLE'] for d in DATASETS};newts=tasks('new',nvs)
        for t in newts:
            if not valid(t):missing.append(task_id(t));continue
            r=load(output(t));newpoints.append(r['result'])
            if r['status']!='PASS':missing.append(task_id(t)+':FAIL')
        expected_new=len(newts)
        for d,label,exclude in (('hevc_b','hevc_b',False),('mcljcv','mcljcv',False),('mcljcv','mcl29',True)):
            selected=[v for v in nvs[d] if not exclude or v['name']!='videoSRC17_1920x1080_24'];ids=[v['video_index'] for v in selected]
            if not ids:continue
            for kind in ('matched_old64','new64','combined_two_clips'):
                t=cpu_task(d,label+'_'+kind,ids,kind)
                if not valid(t):missing.append(task_id(t));continue
                # Per-video LPIPS/DISTS/FloLPIPS diagnostics remain equal-video aggregates.
                seqrows=[]
                for v in selected:
                    curves={m:[] for m in METHODS}
                    for m in METHODS:
                        for q in QPS:
                            oldr=load(point(d,m,v['video_index'],q));nt=dict(kind='new',dataset=d,video=v['video_index'],method=m,qp=q);nr=load(output(nt))['result']
                            if kind=='matched_old64':r=oldr
                            elif kind=='new64':r=nr
                            else:r=dict(QP=q,bpp=(oldr['real_bytes']+nr['real_bytes'])*8/(128*1920*1080),**{k:(oldr[k]+nr[k])/2 for k in ('LPIPS','DISTS','FloLPIPS')})
                            curves[m].append(dict(QP=q,bpp=r['bpp'],**{k:r[k] for k in ('LPIPS','DISTS','FloLPIPS')}))
                    for metric in ('LPIPS','DISTS','FloLPIPS'):
                        c,g=grid_compare(curves,COMPARISONS,label+'_'+kind,metric)
                        for row in c:row.update(dataset=d,sequence=v['name'],video_index=v['video_index'],aggregation='per-video; not pooled features');seqrows.append(row)
                        metricgrids.extend(dict(**row,dataset=d,sequence=v['name']) for row in g)
                metriccomp.extend(seqrows)
                for metric in ('LPIPS','DISTS','FloLPIPS'):
                    for m,ref in COMPARISONS:
                        rows=[r for r in seqrows if r['metric']==metric and r['method']==m and r['reference']==ref];available=[r for r in rows if r['status']=='available']
                        if len(available)!=len(selected):comparisons.append(dict(pool=label+'_'+kind,dataset=d,metric=metric,method=m,reference=ref,status='unavailable',reason='at least one video lacks overlap',available=len(available),expected=len(selected)));continue
                        am=float(np.mean([r['method_mean'] for r in rows]));bm=float(np.mean([r['reference_mean'] for r in rows]))
                        comparisons.append(dict(pool=label+'_'+kind,dataset=d,metric=metric,method=m,reference=ref,status='available',method_mean=am,reference_mean=bm,delta=am-bm,relative_percent=100*(am-bm)/bm,better_grid_fraction=float(np.mean([r['better_grid_fraction'] for r in rows])),sequences=len(rows),aggregation='equal-video mean of individual common-rate-grid means',diagnostic='3-QP stability diagnostic; no extrapolation'))
        if subset_count!=26:missing.append(f'subset statistics completed {subset_count}/26')
    else:expected_new=None
    write(ROOT/'fid_pooled.csv',pooled,fields=None if pooled else ['pool','method','QP','FID','status'])
    write(ROOT/'pool_comparisons.csv',comparisons,fields=None if comparisons else ['pool','method','reference','status','reason'])
    write(ROOT/'common_rate_grid.csv',grids+metricgrids,fields=None if grids or metricgrids else ['pool','index','bpp'])
    write(ROOT/'sample_sensitivity.csv',sensitivity,fields=None if sensitivity else ['pool','status','reason'])
    write(ROOT/'equal_rate_per_sequence.csv',metriccomp,fields=None if metriccomp else ['pool','sequence','metric','status'])
    if newpoints:write(ROOT/'new_clip_metrics/per_sequence_all_qp.csv',newpoints)
    summaries=[]
    for d in DATASETS:
        values=[x for x in sensitivity if x.get('dataset')==d and x.get('type','').startswith('same-QP')]
        for q in QPS:
            rr=[x for x in values if x['QP']==q]
            if rr:summaries.append(dict(dataset=d,QP=q,subsets=len(rr),delta_min=min(x['delta'] for x in rr),delta_max=max(x['delta'] for x in rr),negative=sum(x['delta']<0 for x in rr),positive=sum(x['delta']>0 for x in rr),zero=sum(x['delta']==0 for x in rr),not_confidence_interval=True))
    write(ROOT/'sample_sensitivity_summary.csv',summaries,fields=None if summaries else ['dataset','QP','status','reason'])
    unchanged=[]
    for p,h in load(ROOT/'source_hashes.json')['hashes'].items():
        if sha(p)!=h:unchanged.append(p)
    missing.extend('historical source changed: '+p for p in unchanged)
    status='BLOCKED' if blocked else 'PASS' if audit['status']=='PASS' and len(newpoints)==expected_new and not missing else 'FAIL'
    integrity=dict(status=status,updates=0,closure_expected=420,closure_completed=audit['completed'],closure_pass=audit['status']=='PASS',pooled_closure_expected=36,pooled_closure_completed=audit['pooled_completed'],new_expected=expected_new,new_completed=len(newpoints),subset_expected=26,subset_completed=subset_count,missing=missing,stages2_3_blocked=blocked,historical_sources_unchanged=not unchanged,calculation_complete=status=='PASS',gain_retention='numeric evidence in pool_comparisons.csv; independent of integrity PASS',publication_status='separate publication_status.json')
    if (ROOT/'new_clips_manifest.json').exists():
        integrity['skipped_new_clips']=[dict(dataset=d,sequence=v['name'],reason=v['reason']) for d in DATASETS for v in sources(d,True) if v['status']=='SKIPPED']
    index=[]
    for folder in ('features','bitstreams','cache'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and p.suffix not in ('.lock','.tmp'):index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    dump(ROOT/'binary_artifacts_index.json',dict(files=index,server_only=True))
    def line(r):return f"- {r.get('pool')}: {r.get('method')} − {r.get('reference')}, {r.get('metric','FID')}, delta={r.get('delta')}, relative%={r.get('relative_percent')}, status={r.get('status')}"
    original_gap=[r for r in comparisons if r.get('metric')=='FID' and r.get('reference')=='original' and r.get('method')=='B_covstats_1000']
    # Compute B1-original from the actual recorded common four-model grids, not from QP equality.
    gap=[]
    for s in allstats:
        if s['kind']=='closure_pool':
            curves={m:[r for r in s['pooled'] if r['method']==m] for m in METHODS};gg,_=grid_compare(curves,(('B1_initial','original'),),s['pool']);gap.extend(gg)
    write(ROOT/'original_to_B1_fresh_gap.csv',gap,fields=None if gap else ['pool','status','reason'])
    ba=[r for r in comparisons if r.get('metric')=='FID' and r.get('method')=='B_covstats_1000' and r.get('reference')=='A_continue_1000']
    report=f'''# V6.30 FID cache closure and sample robustness audit\n\nExecution: {status}. All model/optimizer updates: 0. Calculation completion, cache closure, numerical gain retention, and Git publication are separate fields.\n\n## 1. Fresh decode features versus historical caches\n\nClosure: {audit['status']}; video points {audit['completed']}/420; pooled comparisons {audit['pooled_completed']}/36. Maximum relative L2={audit['max_relative_L2']}; maximum absolute feature error={audit['max_abs']}; failed feature rows={audit['feature_row_failures']}. UID-level errors and UNKNOWN reasons are preserved in `cache_closure_per_frame.csv`, `cache_closure_per_point.csv` and `audits/cache_closure_integrity.json`. Tolerances are fixed, no row reordering or historical cache overwrite.\n\n## 2. Original to B1 gap after fresh extraction\n\n3-QP pooled common-actual-bpp diagnostics:\n\n'''+ '\n'.join(line(r) for r in gap)+f'''\n\n## 3. B minus A on fixed new clips and combined pools\n\n'''+ '\n'.join(line(r) for r in ba)+f'''\n\nComparisons use matching-video old/new/combined pools. Combined is two independently I-initialized 64-frame streams, not a continuous 128-frame GOP. Same-QP values are raw diagnostics unless actual bpp matches. Different pools' absolute FID values are not quality improvements. Skipped clips: see frozen new manifest; no alternate clips were selected.\n\n## 4. Direction under video subset sampling\n\nSubset jobs {subset_count}/26. Fixed PCG64(20261030), 10 HEVC three-video combinations; MCL eight distinct ten-video and eight distinct twenty-video subsets. See `subset_manifest.json`, `sample_sensitivity.csv` and `sample_sensitivity_summary.csv` for ranges, sign counts, native actual bpp and all members.\n\n'''+ '\n'.join('- '+json.dumps(r) for r in summaries)+f'''\n\n## 5. Supported evidence and limits\n\nA PASS supports only the specified feature-cache/decode/pooled-statistics consistency checks. Recorded sign counts describe these frozen temporal clips and video subsets. They are not repeated training runs, confidence intervals, significance proofs, or evidence that sample-size bias has been removed. Video correlation, finite samples and a single training realization remain limitations. New temporal clips are from the same test videos, not new scenes. MCL videoSRC17 matches HEVC Kimono1 across test datasets; this is not by itself training-leakage evidence. No FID-infinity fitting or checkpoint selection was performed.\n\n## 6. Evaluation-chain consistency status\n\n{'The fixed chain passed the specified closed-loop checks. This is not a method-success or significant-gain claim.' if audit['status']=='PASS' else 'The specified closed loop did not pass or is incomplete. Stages 2/3 are blocked; recorded mismatches require correction before relying on the affected cached results.'}\n\nPublication: independently recorded in `publication_status.json`; a computation PASS does not claim remote publication. No next-round training is launched.\n'''
    textfile(ROOT/'REPORT.md',report)
    if pooled:
        import subprocess
        dump(ROOT/'audits/plot_payload.json',dict(pooled=pooled,newpoints=newpoints,methods=METHODS,datasets=DATASETS))
        subprocess.run(['/data1/anaconda3_new/anaconda_program/bin/python','-B',str(ROOT/'plot30.py')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    integrity.update(report_sha256=sha(ROOT/'REPORT.md'),plot_files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'plots').rglob('*.png')})
    dump(ROOT/'final_integrity.json',integrity)
    return integrity
