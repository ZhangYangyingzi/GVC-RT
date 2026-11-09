"""Factual tables, fixed-scope aggregates, curves and protocol stop decisions."""
from common_s2 import *

TEMP='MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP'
METRICS=['PSNR','LPIPS','DISTS','FloLPIPS',TEMP,'MCResidual_RGB_RAFT_pixelcenter_v1_PtoP','GT_multiscale_detail_error','GT_Sobel_strength','reconstruction_Sobel_strength','WarpingError_GTFlow_PWC']

def rows():
    out=[]
    for p in sorted((ROOT/'evaluation').glob('*/*/points/*.json')):
        a=load(p);assert a['status']=='PASS';r=dict(a['result']);r['scope']='ULong开发子集12视频'
        if r['model']=='B300':
            d=ROOT/'baseline_detail'/p.name
            if d.exists():r['GT_multiscale_detail_error']=load(d)['GT_multiscale_detail_error']
        out.append(r)
    return out

def mean(rr,k):
    v=[r[k] for r in rr if isinstance(r.get(k),(float,int))]
    return sum(v)/len(v) if v else None

def stop_conditions(group,step):
    rr=rows();b=[r for r in rr if r['model']=='B300'];a=[r for r in rr if r['model']==group and r['step']==step]
    if len(b)!=36 or len(a)!=36:return dict(available=False)
    temporal=mean(a,TEMP)>=mean(b,TEMP);qp=[]
    for q in [0,5,9]:
        aa=[r for r in a if r['QP']==q];bb=[r for r in b if r['QP']==q]
        qp.append(dict(QP=q,LPIPS_ratio=mean(aa,'LPIPS')/mean(bb,'LPIPS'),DISTS_ratio=mean(aa,'DISTS')/mean(bb,'DISTS')))
    spatial=any(x['LPIPS_ratio']>1.01 or x['DISTS_ratio']>1.01 for x in qp)
    return dict(available=True,group=group,step=step,mean_PtoP_Sobel_not_lower_than_B300=temporal,any_QP_LPIPS_or_DISTS_over_1percent=spatial,QP=qp,eligible_for_200_stop=temporal and spatial)

def should_stop(group):return all(stop_conditions(group,s).get('eligible_for_200_stop',False) for s in [100,200])

def report():
    env();rr=rows()
    if rr:write_csv(ROOT/'development_metrics.csv',rr)
    aggregates=[]
    for g,s in sorted({(r['model'],r['step']) for r in rr}):
        r=[a for a in rr if a['model']==g and a['step']==s]
        for q in [0,5,9,None]:
            selected=[a for a in r if q is None or a['QP']==q]
            aggregates.append(dict(model=g,step=s,QP='all_0_5_9' if q is None else q,points=len(selected),scope='ULong开发子集12视频',bpp=mean(selected,'bpp'),**{k:mean(selected,k) for k in METRICS}))
    if aggregates:write_csv(ROOT/'aggregate_by_QP.csv',aggregates)
    training=[]
    for g in GROUPS:
        for p in sorted((ROOT/g/'training/receipts').glob('*.json')):
            a=load(p);t=[v for rank in a['ranks'] for win in rank['records'] for v in win['trace']]
            training.append(dict(group=g,step=a['additional_step'],loss=a['loss'],seconds=a['seconds'],cache_wait_seconds=a['cache_wait_seconds'],gradient_norm=a['gradient_norm'],**{k:mean(t,k) for k in ['LPIPS','DISTS','RGB_L1','RAFT_motion','detail','RGB_temporal','Sobel_temporal']}))
    if training:write_csv(ROOT/'training_history.csv',training)
    stop={g:{str(s):stop_conditions(g,s) for s in [100,200]} for g in GROUPS};dump(ROOT/'stopping_history.json',stop)
    states={g:load(ROOT/g/'training/progress.json') if (ROOT/g/'training/progress.json').exists() else dict(committed_updates=0) for g in GROUPS}
    text=['# GVC-RT E/F B300 execution report','',f'Factual snapshot UTC epoch: {time.time()}.', '', 'Status: '+('COMPLETE' if (ROOT/'status.json').exists() and load(ROOT/'status.json').get('complete') else 'RESULTS_READY_PUBLICATION_PENDING' if (ROOT/'final_integrity.json').exists() and load(ROOT/'final_integrity.json').get('status')=='PASS' else 'INCOMPLETE'),'',f'Initialization: B300 `{B300}`; SHA256 `{B300_SHA}`. Source commit `{SOURCE_COMMIT}`.', '', 'E GPU0/1; F GPU2/3. Fresh AdamW; global batch4 (2 windows/rank), maximum300 additional updates. Independent original codec reference trajectory. No final test evaluation.', '', 'Scope: ULong开发子集12视频, QP0/5/9,64 frames. Original/A2000/B300 inherited36-point records retained with explicit provenance. B300 supplemental detail is reported only when actually computed.', '', f'Incremental cache: initially one GPU producer; now {load(ROOT/"scheduler_config.json")["gpu_cache_workers"]} controlled producers on GPUs {load(ROOT/"scheduler_config.json").get("cache_GPUs",[0])}, authorized by the user.16 distinct startup windows,32-window prefetch, bounded bytes and backpressure. Rank readiness coordinated at full optimizer boundaries. Both cursors protect cache retention.', '']
    for g in GROUPS:
        n=states[g]['committed_updates'];pts=len([r for r in rr if r['model']==g]);done=ROOT/g/'training/completion.json'
        text += [f'{g}: additional optimizer updates={n}; completed development points={pts}; stop reason={load(done)["stop_reason"] if done.exists() else "not stopped/completed"}.']
    text+=['','No method superiority conclusion is made. MP4 preview bytes are excluded from R-D data. All neural codec bpp values originate from verified real RANS streams.','', 'See status.json for current stage, audits for measured checks, development_metrics.csv and aggregate_by_QP.csv for bounded-scope results, training_history.csv for formal optimizer updates.']
    (ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')
    plots(rr,aggregates)

def plots(rr,aggregates):
    if not rr:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (ROOT/'plots').mkdir(exist_ok=True)
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(7,4))
        for g,s in sorted({(r['model'],r['step']) for r in rr}):
            vals=[a for a in aggregates if a['model']==g and a['step']==s and isinstance(a['QP'],int) and a.get(metric) is not None]
            if vals:ax.plot([v['bpp'] for v in vals],[v[metric] for v in vals],'o-',label=f'{g} step{s}')
        ax.set(xlabel='real RANS bpp',ylabel=metric,title='ULong development subset:12 videos (QP0/5/9)');ax.legend(fontsize=7) if ax.lines else None;fig.tight_layout();fig.savefig(ROOT/'plots'/f'RD_{metric}.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        for g in ['E','F']:
            vals=[a for a in aggregates if a['model']==g and a['QP']=='all_0_5_9' and a.get(metric) is not None]
            if vals:ax.plot([v['step'] for v in vals],[v[metric] for v in vals],'o-',label=g)
        if ax.lines:
            ax.set(xlabel='additional optimizer updates from B300',ylabel=metric,title='ULong development subset:12 videos,36 points');ax.legend();fig.tight_layout();fig.savefig(ROOT/'plots'/f'steps_{metric}.png',dpi=150)
        plt.close(fig)
if __name__=='__main__':report()
