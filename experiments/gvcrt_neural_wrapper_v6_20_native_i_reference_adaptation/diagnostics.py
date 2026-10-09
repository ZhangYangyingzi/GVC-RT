"""Frame/window metrics, real-rate differences and first-frame panels only."""
import statistics,math
from io20 import *
WINDOWS={'1':(1,1),'2-8':(2,8),'9-16':(9,16),'17-32':(17,32),'33-48':(33,48),'49-64':(49,64),'2-64':(2,64)}
PRIMARY=('LPIPS','DISTS','FloLPIPS')
def main():
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    from PIL import Image,ImageDraw
    from report import comparisons,PAIRS
    from prepare import selected
    perframe=[];allwindows=[];rawdrift=[];errors=[]
    for d in DATASETS:
        windowrows=[];differences=[];grids=[];ratedrifts=[];ratedriftgrids=[]
        for v in sources(d):
            by={};curves={};drifts={}
            for m in METHODS:
                for q in range(10):
                    r=load(point(d,m,v['video_index'],q));fr=read(r['frame_metrics_path']);tr=read(r['transitions_path']);flow={int(x['to_frame'])+1:float(x['FloLPIPS']) for x in tr};cumulative=0
                    for f in fr:
                        i=int(f['frame'])+1;cumulative+=int(f['real_bits']);perframe.append(dict(dataset=d,sequence=v['name'],method=m,external_QP=q,frame_index=i,source_frame_index=v['source_frame_indices'][i-1],actual_QP=int(f['actual_qp']),actual_bits=int(f['real_bits']),payload_bits=int(f['payload_bits']),cumulative_bits=cumulative,sequence_actual_bpp=r['bpp'],LPIPS=float(f['LPIPS']),DISTS=float(f['DISTS']),FloLPIPS=flow.get(i),PSNR=float(f['PSNR']),SSIM=float(f['SSIM'])))
                    assert cumulative==r['total_bits']
                    for metric in PRIMARY:
                        for window,(lo,hi) in WINDOWS.items():
                            values=[flow[i] for i in range(lo,hi+1) if i in flow] if metric=='FloLPIPS' else [float(fr[i-1][metric]) for i in range(lo,hi+1)]
                            mean=statistics.mean(values) if values else None;row=dict(dataset=d,sequence=v['name'],method=m,external_QP=q,QP=q,metric=metric,window=window,mean=mean,std=statistics.pstdev(values) if values else None,count=len(values),bpp=r['bpp'],rate_scope='entire64frames_real_bytes',status='defined' if values else 'undefined_no_first_frame_transition');windowrows.append(row);by[m,q,metric,window]=row
                            curves.setdefault((metric,window),[]).append(dict(method=m,QP=q,bpp=r['bpp'],**{metric:mean}))
                        early=by[m,q,metric,'2-8']['mean'];late=by[m,q,metric,'49-64']['mean'];drifts[m,q,metric]=(early,late)
            for m in METHODS:
                for q in range(10):
                    for metric in PRIMARY:
                        early,late=drifts[m,q,metric];oe,ol=drifts['original',q,metric];rawdrift.append(dict(dataset=d,sequence=v['name'],method=m,external_QP=q,metric=metric,early_2_8=early,late_49_64=late,late_minus_early=late-early,original_drift=ol-oe,excess_drift_vs_original=(late-early)-(ol-oe),comparison='same_external_QP_diagnostic'))
            for (metric,window),rows in curves.items():
                if metric=='FloLPIPS' and window=='1':continue
                table,grid=comparisons(d,v['name'],metric,rows)
                for x in table:x['window']=window;x['rate_scope']='entire64frames_real_bytes'
                for x in grid:x['window']=window;x['rate_scope']='entire64frames_real_bytes'
                differences.extend(table);grids.extend(grid)
            for metric in PRIMARY:
                early=[x for x in grids if x['sequence']==v['name'] and x['metric']==metric and x['window']=='2-8'];late=[x for x in grids if x['sequence']==v['name'] and x['metric']==metric and x['window']=='49-64']
                if not early or not late:
                    for m in METHODS:ratedrifts.append(dict(dataset=d,sequence=v['name'],metric=metric,method=m,status='unavailable',reason='no five-method common bitrate interval'))
                    continue
                assert len(early)==len(late)==100
                records=[]
                for e,l in zip(early,late):
                    assert e['bpp']==l['bpp'];orig=l['original']-e['original']
                    for m in METHODS:records.append(dict(dataset=d,sequence=v['name'],metric=metric,method=m,grid_index=e['grid_index'],bpp=e['bpp'],early_2_8=e[m],late_49_64=l[m],late_minus_early=l[m]-e[m],original_drift=orig,excess_drift_vs_original=l[m]-e[m]-orig,comparison='five-method_common_real_rate_PCHIP',rate_scope='entire64frames_real_bytes'))
                ratedriftgrids.extend(records)
                for m in METHODS:
                    rs=[x for x in records if x['method']==m];ratedrifts.append(dict(dataset=d,sequence=v['name'],metric=metric,method=m,status='available',grid_points=100,common_min_bpp=early[0]['bpp'],common_max_bpp=early[-1]['bpp'],**{k:statistics.mean(x[k] for x in rs) for k in ('early_2_8','late_49_64','late_minus_early','original_drift','excess_drift_vs_original')}))
            if selected(v):
                for q in (0,4,9):
                    caches={m:load(point(d,m,v['video_index'],q))['first_frame_audit'] for m in METHODS[3:]};arrays={}
                    for m,c in caches.items():
                        assert sha(c['visualization_cache_path'])==c['visualization_cache_sha256']
                        with np.load(c['visualization_cache_path']) as f:
                            if m=='wrapped_i_control_1000':arrays['GT']=f['GT'][0]
                            arrays['P(GT) '+m]=f['P_GT'][0]
                            arrays[m]=f['reconstruction'][0]
                    panel=Image.new('RGB',(5*384,250),'white');draw=ImageDraw.Draw(panel)
                    for column,(name,array) in enumerate(arrays.items()):
                        diff=array-arrays['GT'];means=diff.mean(axis=(1,2));maes=np.abs(diff).mean(axis=(1,2));errors.append(dict(dataset=d,sequence=v['name'],external_QP=q,image=name,mean_error_R=float(means[0]),mean_error_G=float(means[1]),mean_error_B=float(means[2]),MAE_R=float(maes[0]),MAE_G=float(maes[1]),MAE_B=float(maes[2]),MAE=float(np.abs(diff).mean()),value_range='RGB_float_0_1',statistics_before_visualization_rounding=True))
                        rgb=np.rint(np.clip(array.transpose(1,2,0),0,1)*255).astype(np.uint8);pic=Image.fromarray(rgb).resize((384,216),Image.Resampling.LANCZOS);panel.paste(pic,(column*384,34));draw.text((column*384+6,8),name,fill='black')
                    dest=ROOT/'plots/first_frames'/f'{d}_{v["name"]}_qp{q}.png';dest.parent.mkdir(parents=True,exist_ok=True);panel.save(dest)
        allwindows.extend(windowrows);folder=ROOT/'results'/d
        for name,rows in [('window_metrics',windowrows),('window_equal_rate_per_sequence',differences),('window_common_rate_grid',grids),('temporal_drift_equal_rate',ratedrifts),('temporal_drift_equal_rate_grid',ratedriftgrids)]:
            write(folder/(name+'.csv'),rows);dump(folder/(name+'.json'),rows)
        summary=[]
        for m,b in PAIRS:
            for metric in PRIMARY:
                for window in WINDOWS:
                    rs=[x for x in differences if x['method']==m and x['reference']==b and x['metric']==metric and x['window']==window];ok=[x for x in rs if x['status']=='available'];summary.append(dict(dataset=d,method=m,reference=b,metric=metric,window=window,status='available' if len(ok)==len(sources(d)) else 'unavailable',expected_sequences=len(sources(d)),available_sequences=len(ok),delta=statistics.mean(x['delta'] for x in ok) if len(ok)==len(sources(d)) else None,method_mean=statistics.mean(x['method_mean'] for x in ok) if len(ok)==len(sources(d)) else None,reference_mean=statistics.mean(x['reference_mean'] for x in ok) if len(ok)==len(sources(d)) else None,better_grid_fraction=statistics.mean(x['better_grid_fraction'] for x in ok) if len(ok)==len(sources(d)) else None,aggregation='equal sequence weights',rate_scope='entire64frames_real_bytes'))
        write(folder/'window_equal_rate_summary.csv',summary);dump(folder/'window_equal_rate_summary.json',summary)
    for path,rows in [('evaluation/per_frame_metrics',perframe),('results/window_metrics',allwindows),('results/temporal_drift_per_sequence',rawdrift),('results/first_frame_RGB_errors',errors)]:write(ROOT/(path+'.csv'),rows);dump(ROOT/(path+'.json'),rows)
    assert len(perframe)==800*64 and len(errors)==4*3*5
    dump(ROOT/'audits/diagnostic_integrity.json',dict(status='PASS',per_frame_rows=len(perframe),window_rows=len(allwindows),FloLPIPS_assignment='transition target frame; frame1 undefined',rate_accounting='all64frames including SPS and headers',first_frame_RGB_rows=len(errors),first_frame_visualizations=12));print('DIAGNOSTICS PASS',flush=True)
