"""Independent CPU review; writes only in this analysis folder."""
from pathlib import Path
import json,csv,gzip,hashlib,math,collections
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def load(p):
    p=Path(p)
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def readcsv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def writecsv(fn,rows):
    with (OUT/fn).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
audit={'scope':'CPU only; no new training or metric-network inference','failures':[]}
def ck(test,msg):
    if not test:audit['failures'].append(msg)
rows=readcsv(ROOT/'metrics.csv')
models=collections.defaultdict(dict)
for r in rows:models[r['model']][(r['domain'],r['sequence'],r['QP'])]=r
ck(len(rows)==900 and all(len(r)==225 for r in models.values()),'final count')
ck(all(set(r)==set(next(iter(models.values()))) for r in models.values()),'paired point keys')
metrics=['LPIPS','DISTS','FloLPIPS','PSNR','MS_SSIM','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']
cand='ULong_continuation_s05';paired=[]
for ds,q in sorted({(r['dataset'],r['QP']) for r in rows}):
    keys=sorted(k for k,r in models[cand].items() if r['dataset']==ds and r['QP']==q)
    for ref in ['A2000_s05','Original_GVC_RT','TCG_B1500']:
        for m in metrics:
            a=np.array([float(models[cand][k][m]) for k in keys]);b=np.array([float(models[ref][k][m]) for k in keys]);low=m not in ['PSNR','MS_SSIM']
            paired.append(dict(dataset=ds,QP=q,reference=ref,metric=m,n_videos=len(keys),candidate_mean=float(a.mean()),reference_mean=float(b.mean()),absolute_delta=float(a.mean()-b.mean()),relative_percent=float(100*(a.mean()/b.mean()-1)),improved_videos=int(np.sum(a<b if low else a>b))))
writecsv('paired_dataset_qp_recomputed.csv',paired)
base=readcsv(ROOT/'reference_A2000_s05/evaluation/development_baseline/metrics.csv');dev=readcsv(ROOT/'development_metrics.csv');anchor=1.;bad=0;traj=[]
for step in sorted({int(r['additional_step']) for r in dev}):
    rr=[r for r in dev if int(r['additional_step'])==step];d={};ck(len(rr)==96,f'dev count {step}')
    for q in ['0','5','9']:
        a=[r for r in rr if r['QP']==q];b=[r for r in base if r['QP']==q];ck(len(a)==len(b)==32,f'dev QP {step}/{q}')
        for m in ['FloLPIPS','DISTS']:d[m+'_'+q]=(sum(float(r[m]) for r in a)/len(a))/(sum(float(r[m]) for r in b)/len(b))
    s=sum(d['FloLPIPS_'+q] for q in ['0','5','9'])/3;qualified=s<1 and all(d['DISTS_'+q]<=1 for q in ['0','5','9'])
    if qualified and s<anchor-.0005:anchor=s;bad=0
    else:bad+=1
    ck(abs(s-float(rr[0]['S']))<1e-12 and str(qualified)==rr[0]['qualified'] and bad==int(rr[0]['bad_check_count']),f'dev controller {step}')
    traj.append(dict(additional_step=step,S=s,qualified=qualified,bad_count=bad,**d))
sel=load(ROOT/'selection.json');best=min((r for r in traj if r['qualified']),key=lambda r:r['S'])
ck(best['additional_step']==sel['selected_additional_step']==750,'selected step');ck(traj[-1]['additional_step']==1750 and bad==4,'early stop')
ck(sha(sel['checkpoint'])==sel['checkpoint_SHA256'],'selected checkpoint SHA')
writecsv('development_selection_recomputed.csv',traj);audit['selection']=dict(selected_additional_step=750,selected_absolute_step=2750,last_additional_step=1750,best_S=best['S'])
manifest={(v['domain'],v['sequence']):v for v in load(ROOT/'manifests/final_videos.json')+load(ROOT/'manifests/MCL_JCV_30.json')}
groups=collections.defaultdict(list);streams={};errors=collections.defaultdict(float);counts=collections.Counter();weights=set();segments=[]
for st in load(ROOT/'final_states.json')['states']:
    batch=ROOT/st['batch'];files=list((batch/'points').glob('*.json'));ck(len(files)==225,'point count '+st['name'])
    for f in files:
        p=load(f);r=p['result'];k=(r['domain'],r['sequence'],str(r['QP']));top=models[st['name']][k];v=manifest[k[:2]]
        ck(p['status']=='PASS' and p['real_RANS'] and p['independent_decode'],'point flags '+str(f))
        ck(all(str(r[x])==str(top[x]) for x in metrics+['bpp','real_bytes','bitstream_sha256','source_RGB_sha256']),'top metrics '+str(f))
        if st['name']=='Original_GVC_RT':
            pv=p['model_provenance'];ck(pv['original_codec_only'] and not any(pv[x] for x in ['pilot_loaded','TCG_loaded','LoRA_loaded','Router_instantiated']),'original provenance '+str(f))
        if st['name']==cand:ck(p['lora_checkpoint_sha256']==sel['checkpoint_SHA256'] and p['s_eval']==.5,'selected state '+str(f))
        mh=json.dumps(p['metric_hashes'],sort_keys=True);weights.add(mh);stream=Path(p['stream_path'])
        if stream not in streams:streams[stream]=sha(stream)
        ck(streams[stream]==r['bitstream_sha256'] and stream.stat().st_size==int(r['real_bytes']),'stream '+str(f))
        records={n:readcsv(batch/n/(f.stem+'.csv')) for n in ['frames','transitions','codec_trace']}
        hashes=p.get('record_file_SHA256',p.get('reuse_provenance',{}).get('record_file_SHA256',{}))
        for n,h in hashes.items():ck(sha(batch/n/(f.stem+'.csv'))==h,'record SHA '+str(f)+'/'+n)
        if 'reuse_provenance' in p:
            rp=p['reuse_provenance'];ck(sha(rp['source_point'])==rp['source_point_SHA256'],'reuse source '+str(f))
        fr=records['frames'];tr=records['transitions'];ct=records['codec_trace']
        ck(len(fr)==64 and len(tr)==63 and len(ct)==64,'record count '+str(f));counts['frames']+=len(fr);counts['transitions']+=len(tr)
        ck([int(x['frame']) for x in fr]==list(range(64)),'frame order '+str(f))
        ck([(int(x['from_frame']),int(x['to_frame'])) for x in tr]==[(i,i+1) for i in range(63)],'transition order '+str(f))
        ck(sum(int(x['real_bits']) for x in fr)==8*int(r['real_bytes']) and int(fr[0]['real_bits'])>0,'I/frame bits '+str(f))
        ck(8*int(r['real_bytes'])/(64*v['width']*v['height'])==r['bpp'],'bpp '+str(f))
        recalc={'PSNR':-10*math.log10(sum(float(x['pixel_SSE']) for x in fr)/(64*v['width']*v['height']*3))}
        recalc.update({m:sum(float(x[m]) for x in fr)/64 for m in ['MS_SSIM','LPIPS','DISTS']})
        recalc.update({m:sum(float(x[m]) for x in tr)/63 for m in ['FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC']})
        for m,val in recalc.items():
            err=abs(val-float(r[m]));errors[m]=max(errors[m],err);ck(err<1e-10,'aggregation '+str(f)+'/'+m)
        groups[k].append(dict(bpp=r['bpp'],bytes=r['real_bytes'],stream=r['bitstream_sha256'],source=r['source_sha256'],rgb=r['source_RGB_sha256'],actual_QPs=p['actual_qps'],indices=p['source_frame_indices'],metric_weights=mh,codec_trace=ct,GT_flow=[x['GT_flow_sha256'] for x in tr],source_RGB=[x['source_frame_RGB_sha256'] for x in fr]))
        for lo,hi in [(1,16),(17,32),(33,48),(49,63)]:
            ff=[x for x in fr if lo<=int(x['frame'])<=hi];tt=[x for x in tr if lo<=int(x['to_frame'])<=hi]
            segments.append(dict(model=st['name'],dataset=top['dataset'],sequence=top['sequence'],QP=r['QP'],window=f'{lo}-{hi}',DISTS=sum(float(x['DISTS']) for x in ff)/len(ff),LPIPS=sum(float(x['LPIPS']) for x in ff)/len(ff),FloLPIPS=sum(float(x['FloLPIPS']) for x in tt)/len(tt)))
for k,ps in groups.items():ck(len(ps)==4 and all(p==ps[0] for p in ps),'paired stream/source/feature/flow '+str(k))
audit['raw_records']=dict(points=900,unique_paired_points=len(groups),**counts,stream_paths_hashed=len(streams),metric_weight_sets=len(weights),aggregation_max_abs_error=dict(errors));writecsv('within_clip_metrics.csv',segments)
hist=readcsv(ROOT/'training_history.csv');bykey={r['sample_key']:r for r in hist};coverage=[]
for stop in [750,1750]:
    rr=[r for r in hist if int(r['additional_step'])<=stop];starts=[int(r['first_frame']) for r in rr]
    coverage.append(dict(additional_step=stop,samples=len(rr),unique_videos=len({r['video'] for r in rr}),unique_source_groups=len({r['source_group'] for r in rr}),unique_windows=len({(r['video'],r['first_frame']) for r in rr}),QP_counts=dict(sorted(collections.Counter(r['external_QP'] for r in rr).items())),domains=dict(collections.Counter(r['domain'] for r in rr)),nonzero_starts=sum(x>0 for x in starts),start_min=min(starts),start_max=max(starts)))
ck(len(hist)==len(bykey)==7000,'sample uniqueness');frozen=set();nread=0
for step,f in enumerate(sorted((ROOT/'receipts').glob('update_*.json.gz')),1):
    r=load(f);ck(r['step']==step and r['absolute_LoRA_optimizer_step']==2000+step,'receipt step '+str(step))
    ck(all(r[x] for x in ['rank_synchronized','optimizer_synchronized','only_LoRA_changed','frozen_unchanged']),'training freeze '+str(step))
    ck(r['gradient_normalization_denominator']==4 and r['global_clip_count']==r['optimizer_step_count']==1,'reduction '+str(step))
    for rank in r['physical_ranks']:frozen.add(rank['frozen_sha256'])
    for s in r['logical_samples']:
        h=bykey[s['sample_key']];ap=ROOT/'audits/window_reads'/(s['sample_key']+'.json.gz');a=load(ap);nread+=1;first=int(h['first_frame'])
        ck(sha(ap)==s['read_audit_sha256'],'read SHA '+s['sample_key'])
        ck(s['domain']=='ulong' and s['frozen_byte_exact'] and s['buffers_cleared'],'sample freeze '+s['sample_key'])
        ck(a['status']=='PASS' and a['source_frame_indices']==s['source_frame_indices']==list(range(first,first+64)),'native frames '+s['sample_key'])
        ck(a['real_contiguous_frames']==64 and a['no_resize_repetition_padding_or_stitch'],'native semantics '+s['sample_key'])
        ck(a['crop']==s['crop'] and a['source_sha256']==s['source_sha256'],'native crop/source '+s['sample_key'])
ck(step==1750 and nread==7000 and len(frozen)==1,'training receipt/frozen count')
audit['sampling']=dict(coverage=coverage,receipts=step,native_read_audits=nread,frozen_hash_count=len(frozen),caveat='New exposures, not necessarily historically unseen videos; no repeat-group visit occurred, so temporal-bin revisit cycling was not exercised.')
sm=load(ROOT/'source_manifest.json');hcounts={}
for group in ['code','protected_sources','immutable_manifests']:
    for fn,h in sm[group].items():
        p=Path(fn) if Path(fn).is_absolute() else ROOT/fn;ck(p.exists() and sha(p)==h,'source SHA '+fn)
    hcounts[group]=len(sm[group])
rh=load(ROOT/'result_hashes.json')['SHA256'];stale=[fn for fn,h in rh.items() if not (ROOT/fn).exists() or sha(ROOT/fn)!=h]
audit['source_hash_checks']=hcounts;audit['result_hash_manifest_mismatches']=stale
pr=load(ROOT/'publication_remote_receipt.json');bloberrors=[]
for fn,d in pr['core_blobs'].items():
    b=(ROOT/fn).read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+bytes([0])+b).hexdigest()
    if sha(ROOT/fn)!=d['SHA256'] or blob!=d['git_blob']:bloberrors.append(fn)
ck(not bloberrors,'published core blobs');audit['publication']=dict(remote_HEAD=pr['remote_HEAD'],core_files_checked=len(pr['core_blobs']),core_blob_mismatches=bloberrors,live_remote_head_confirmed='bb2f504a0ecaa390a0ba58682320fc30c3142221')
audit['status']='PASS' if not audit['failures'] else 'FAIL'
(OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n');print(json.dumps(audit,ensure_ascii=False))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
datasets=['ULong_new_holdout_32','ULong_historical_6','HEVC_B_5','MCL_JCV_30','UVG_historical_HoneyBee_Jockey_2'];names=['ULong holdout (32)','ULong historical (6)','HEVC-B (5)','MCL-JCV (30)','UVG (2)']
fig,axs=plt.subplots(1,3,figsize=(15,4.6))
for ax,m in zip(axs,['FloLPIPS','DISTS','LPIPS']):
    for qi,(q,color) in enumerate(zip(['0','5','9'],['#2166ac','#e68a00','#8e44ad'])):
        vals=[next(x['relative_percent'] for x in paired if x['dataset']==ds and x['QP']==q and x['reference']=='A2000_s05' and x['metric']==m) for ds in datasets]
        ax.bar(np.arange(5)+(qi-1)*.23,vals,width=.23,label='QP '+q,color=color)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(5),names,rotation=30,ha='right');ax.set_title(m+' (lower is better)');ax.set_ylabel('Change vs A2000 s=0.5 (%)');ax.grid(axis='y',alpha=.2);ax.legend()
fig.suptitle('ULong-only continuation +750, s=0.5 | Same bitstreams | Negative = improvement');fig.tight_layout();fig.savefig(OUT/'incremental_changes_vs_A2000.png',dpi=180);plt.close(fig)
