"""Raw fixed-endpoint values and common-rate interpolation; no interpretation."""
import ast,math,statistics,traceback
from io21 import *
PAIRS=(('B_covstats_1000','A_continue_1000'),('A_continue_1000','B1_initial'),('B_covstats_1000','B1_initial'),('B_covstats_1000','original'))
def comparisons(dataset,sequence,metric,rows,pooled=False,methods=METHODS,pairs=PAIRS):
    source=(V15/'report.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='comparisons')
    code=ast.get_source_segment(source,node).replace('expected_models=6','expected_models=len(METHODS)').replace('available_models=6','available_models=len(METHODS)').replace('six-model','all-model')
    ns=dict(globals(),METHODS=methods,PAIRS=pairs);exec(compile(code,str(V15/'report.py'),'exec'),ns)
    results,grid=ns['comparisons'](dataset,sequence,metric,rows,pooled)
    for r in results:
        r['relative_change_percent']=100*r['delta']/r['reference_mean'] if r['status']=='available' and r['reference_mean'] else None
        r['relative_change_definition']='100*(method_mean-reference_mean)/reference_mean; NOT BD-rate'
    return results,grid
def summarize(d,per,n,pairs=PAIRS):
    out=[]
    for method,reference in pairs:
        for metric in ('LPIPS','DISTS','FloLPIPS'):
            group=[r for r in per if r['method']==method and r['reference']==reference and r['metric']==metric];ok=[r for r in group if r['status']=='available'];complete=len(group)==len(ok)==n
            r=dict(dataset=d,method=method,reference=reference,metric=metric,status='available' if complete else 'unavailable',expected_sequences=n,available_sequences=len(ok),rate_axis='per-video actual bpp',aggregation='equal video weight; unavailable if any sequence unavailable',grid_points_per_sequence=100,shared_interval='all methods common per-video interval; see per-sequence/grid',reason='' if complete else 'one or more videos unavailable')
            for k in ('method_mean','reference_mean','delta','better_grid_fraction','relative_change_percent'):r[k]=statistics.mean(t[k] for t in ok) if complete and all(t[k] is not None for t in ok) else None
            r['mean_per_video_relative_change_percent']=r.pop('relative_change_percent')
            r['relative_change_percent']=100*(r['method_mean']-r['reference_mean'])/r['reference_mean'] if complete and r['reference_mean'] else None
            r['relative_change_definition']='100*(dataset method_mean-dataset reference_mean)/dataset reference_mean; NOT BD-rate';out.append(r)
    return out
def window_rows(r,fr,tr):
    out=[]
    for lo,hi in ((0,1),(1,8),(8,16),(16,32),(32,48),(48,64)):
        fs=fr[lo:hi];ts=[x for x in tr if lo<=int(x['to_frame'])<hi];ni=sum(x['frame_type']=='I' for x in fs)
        out.append(dict(IP=r['IP'],dataset=r['dataset'],method=r['method'],sequence=r['sequence'],QP=r['QP'],window_1based=f'{lo+1}-{hi}',start_frame0=lo,end_frame0_exclusive=hi,I_frames=ni,P_frames=len(fs)-ni,transition_count=len(ts),transition_rule='destination frame within window, includes cross-refresh',**{k:statistics.mean(float(x[k]) for x in fs) for k in ('LPIPS','DISTS','PSNR','SSIM','MS_SSIM')},FloLPIPS=statistics.mean(float(x['FloLPIPS']) for x in ts) if ts else None,real_bits=sum(int(x['real_bits']) for x in fs),payload_bits=sum(int(x['payload_bits']) for x in fs),rate_axis='full64 actual bpp',full64_bpp=r['bpp']))
    return out

