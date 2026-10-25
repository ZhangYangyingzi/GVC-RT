"""Aggregate only verified task files; produce descriptive v3 comparisons."""
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
sys.path.insert(0,str(P.parents[1]))
import task
from src.utils import stream_helper as stream_helper

VIDEOS=('UVal0','UVal1','ParkScene','Jockey')
METRICS=('LPIPS','DISTS','FloLPIPS','PSNR')


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def metric_delta(metric,method,reference):
    return method-reference


def payload_digest(wrapped):
    _,_,native=task.read_header(wrapped)
    buffer=io.BytesIO(native);h=hashlib.sha256();count=0;frames=0
    while buffer.tell()<len(native):
        header=stream_helper.read_header(buffer)
        if header['nal_type']==stream_helper.NalType.NAL_SPS:
            stream_helper.read_sps_remaining(buffer,header['sps_id'])
            continue
        qp,bits=stream_helper.read_ip_remaining(buffer)
        h.update(bits);count+=len(bits);frames+=1
    assert frames==64 and buffer.tell()==len(native)
    return h.hexdigest(),count


def expected_tasks():
    specs=[('A',s,q,2) for s in VIDEOS for q in (2,3,4,5,6,7)]
    specs += [('MAP',s,q,2) for s in VIDEOS for q in (2,4,6,7)]
    specs += [(m,s,q,2) for s in ('UVal1','Jockey') for q in (2,7) for m in ('MEAN','ROLL','IDENTITY')]
    specs += [('A',s,9,2) for s in VIDEOS]+[('MAP','UVal0',9,2)]
    choices=read_csv(P/'rate_match_selection.csv')
    specs += [(r['mode'],r['sequence'],int(r['external_qp']),int(r['alpha_index']))
              for r in choices if int(r['alpha_index'])!=2]
    assert len(specs)==57+sum(int(r['alpha_index'])!=2 for r in choices)
    return specs


def load_results():
    specs=expected_tasks();out=[];frames=[];trans=[];segments=[];diagnostics=[]
    checkpoint_shas={m:task.sha(root/'server_artifacts'/branch/'step_300.pt')
                     for m,root,branch in [('A',task.V1,'A'),('H11',task.V2,'H11')]}
    for mode,source,qp,alpha in specs:
        key=task.key(mode,source,qp,alpha)
        path=P/'server_artifacts/evaluation'/f'{key}.json'
        row=json.loads(path.read_text())
        stream=Path(row['stream_path'])
        assert row['status']=='PASS' and stream.exists() and task.sha(stream)==row['stream_sha256']
        assert row['mode']==mode and row['sequence']==source and row['external_qp']==qp and row['alpha_index']==alpha
        assert row['real_total_bytes']==stream.stat().st_size
        payload_sha,payload_count=payload_digest(stream.read_bytes())
        assert payload_count==row['codec_payload_bytes']
        row['codec_payload_sha256']=payload_sha
        assert row['real_total_bytes']==row['codec_payload_bytes']+row['native_header_bytes']+row['experiment_header_bytes']
        assert row['experiment_header_bytes']==16 and row['checkpoint_sha256']==checkpoint_shas['A' if mode=='A' else 'H11']
        row['sampling_fps']=row.get('sampling_fps',task.source_record(source).get('sampled_fps',row['source_fps']))
        row['estimated_P_y_bits']=row.get('estimated_P_y_bits',row.get('estimated_P_main_latent_bits',''))
        row['estimated_P_z_bits']=row.get('estimated_P_z_bits',row.get('estimated_P_hyper_latent_bits',''))
        row['actual_P_payload_bits']=8*row['P_payload_bytes']
        if row['estimated_P_y_bits']!='' and row['estimated_P_z_bits']!='':
            row['estimated_P_total_bits']=float(row['estimated_P_y_bits'])+float(row['estimated_P_z_bits'])
            row['P_estimate_minus_payload_bits']=row['estimated_P_total_bits']-row['actual_P_payload_bits']
        out.append(row)
        fs=read_csv(P/'server_artifacts/frames'/f'{key}.csv')
        ts=read_csv(P/'server_artifacts/transitions'/f'{key}.csv')
        assert len(fs)==64 and len(ts)==63
        assert sum(int(float(f['payload_bytes'])) for f in fs)==row['codec_payload_bytes']
        assert sum(int(float(f['payload_bytes'])) for f in fs[1:])==row['P_payload_bytes']
        for f in fs:
            f['native_bytes']=int(float(f.get('native_bytes',f.get('real_bytes',0))))
            f['experiment_header_bytes']=16 if int(f['frame'])==0 else 0
            f['actual_total_frame_bytes']=f['native_bytes']+f['experiment_header_bytes']
            frames.append({'task_key':key,'mode':mode,'sequence':source,'external_qp':qp,'alpha_index':alpha,**f})
        assert sum(x['actual_total_frame_bytes'] for x in fs)==row['real_total_bytes']
        for t in ts: trans.append({'task_key':key,'mode':mode,'sequence':source,'external_qp':qp,'alpha_index':alpha,**t})
        diag_path=P/'server_artifacts/diagnostics'/f'{key}.csv'
        if diag_path.exists():
            for d in read_csv(diag_path):
                diagnostics.append({'task_key':key,'mode':mode,'sequence':source,'external_qp':qp,'alpha_index':alpha,**d})
        for label,items in [('first15',fs[:15]),('remaining49',fs[15:])]:
            seg={'task_key':key,'segment':label,'frames':len(items),'payload_bytes':sum(int(float(x['payload_bytes'])) for x in items),
                 'native_bytes':sum(x['native_bytes'] for x in items),
                 'actual_total_bytes':sum(x['actual_total_frame_bytes'] for x in items)}
            for m in ('LPIPS','DISTS'):
                seg[m]=sum(float(x[m]) for x in items)/len(items)
            seg['PSNR']=-10*math.log10(max(sum(float(x['pixel_SSE']) for x in items)/(len(items)*row['height']*row['width']*3),1e-15))
            relevant=[x for x in ts if (int(x['to_frame'])<15 if label=='first15' else int(x['to_frame'])>=15)]
            seg['FloLPIPS']=sum(float(x['FloLPIPS']) for x in relevant)/len(relevant)
            segments.append(seg)
    assert len(out)==len(specs) and len({(r['mode'],r['sequence'],r['external_qp'],r['alpha_index']) for r in out})==len(out)
    task.write_csv(P/'evaluation.csv',out)
    task.write_csv(P/'per_frame.csv',frames)
    task.write_csv(P/'per_transition.csv',trans)
    task.write_csv(P/'frame_segment_summaries.csv',segments)
    task.write_csv(P/'quant_diagnostics.csv',diagnostics)
    return out


def compare(out):
    rows=[];sensitivity=[]
    for source in VIDEOS:
        points=sorted([x for x in out if x['mode']=='A' and x['sequence']==source],key=lambda x:x['actual_bpp'])
        for qp in (2,7):
            target=next(x for x in out if x['mode']=='MAP' and x['sequence']==source and x['external_qp']==qp)
            b=target['actual_bpp']
            nearest=min(points,key=lambda x:abs(x['actual_bpp']-b))
            lower=max((x for x in points if x['actual_bpp']<=b),key=lambda x:x['actual_bpp'],default=None)
            upper=min((x for x in points if x['actual_bpp']>=b),key=lambda x:x['actual_bpp'],default=None)
            for metric in METRICS:
                row={'comparison':'A_vs_MAP','sequence':source,'MAP_QP':qp,'metric':metric,
                     'MAP_actual_bpp':b,'MAP_measured_quality':target[metric],
                     'nearest_A_QP':nearest['external_qp'],'nearest_A_actual_bpp':nearest['actual_bpp'],
                     'nearest_A_measured_quality':nearest[metric],
                     'nearest_rate_difference_bpp':nearest['actual_bpp']-b,
                     'nearest_measured_delta_MAP_minus_A':target[metric]-nearest[metric]}
                if lower and upper and lower is not upper:
                    lo,hi=lower['actual_bpp'],upper['actual_bpp']
                    linear=lower[metric]+(upper[metric]-lower[metric])*(b-lo)/(hi-lo)
                    logarithmic=lower[metric]+(upper[metric]-lower[metric])*(math.log(b)-math.log(lo))/(math.log(hi)-math.log(lo))
                    dlin=target[metric]-linear;dlog=target[metric]-logarithmic
                    row.update({'interpolation_status':'BRACKETED','A_lower_QP':lower['external_qp'],'A_upper_QP':upper['external_qp'],
                        'A_lower_actual_bpp':lo,'A_upper_actual_bpp':hi,
                        'A_linear_bpp_interpolated_quality':linear,'A_log_bpp_interpolated_quality':logarithmic,
                        'delta_MAP_minus_A_linear':dlin,'delta_MAP_minus_A_log':dlog,
                        'interpolation_delta_sign_agrees':(dlin>=0)==(dlog>=0)})
                    sensitivity.append({'sequence':source,'MAP_QP':qp,'metric':metric,'linear_delta':dlin,'log_delta':dlog,
                                        'sign_agrees':(dlin>=0)==(dlog>=0),'interpolation_sensitivity':abs(dlin-dlog)})
                    sparse=[x for x in points if x['external_qp'] in (2,7,9)]
                    slo=max((x for x in sparse if x['actual_bpp']<=b),key=lambda x:x['actual_bpp'],default=None)
                    shi=min((x for x in sparse if x['actual_bpp']>=b),key=lambda x:x['actual_bpp'],default=None)
                    if slo is not None and shi is not None and slo is not shi:
                        lb,hb=slo['actual_bpp'],shi['actual_bpp']
                        sl=slo[metric]+(shi[metric]-slo[metric])*(b-lb)/(hb-lb)
                        sg=slo[metric]+(shi[metric]-slo[metric])*(math.log(b)-math.log(lb))/(math.log(hb)-math.log(lb))
                        sensitivity[-1].update({'sparse_A_QPs':'2;7;9','sparse_lower_QP':slo['external_qp'],
                            'sparse_upper_QP':shi['external_qp'],'sparse_linear_delta':target[metric]-sl,
                            'sparse_log_delta':target[metric]-sg,
                            'dense_vs_sparse_linear_sign_agrees':(dlin>=0)==(target[metric]-sl>=0),
                            'dense_vs_sparse_log_sign_agrees':(dlog>=0)==(target[metric]-sg>=0)})
                else:row['interpolation_status']='NO_BRACKET_NO_EXTRAPOLATION'
                rows.append(row)
        for target in (x for x in out if x['mode']=='MAP' and x['sequence']==source):
            reference=next((x for x in points if x['external_qp']==target['external_qp']),None)
            if reference is not None:
                for metric in METRICS:
                    rows.append({'comparison':'A_vs_MAP_same_QP','sequence':source,
                        'external_qp':target['external_qp'],'metric':metric,'reference':'A','method':'MAP',
                        'reference_actual_bpp':reference['actual_bpp'],'method_actual_bpp':target['actual_bpp'],
                        'rate_difference_bpp':target['actual_bpp']-reference['actual_bpp'],
                        'reference_measured_quality':reference[metric],'method_measured_quality':target[metric],
                        'measured_delta_method_minus_reference':target[metric]-reference[metric]})
    selection=read_csv(P/'rate_match_selection.csv')
    for source in ('UVal1','Jockey'):
        for qp in (2,7):
            mapping={m:next(x for x in out if x['mode']==m and x['sequence']==source and x['external_qp']==qp and x['alpha_index']==2)
                     for m in ('A','MAP','MEAN','ROLL','IDENTITY')}
            for reference,method,kind in [('A','IDENTITY','same_native_stream_generator'),
                                          ('MAP','MEAN','same_QP_spatial_mean'),('MAP','ROLL','same_QP_spatial_roll')]:
                r,m=mapping[reference],mapping[method]
                for metric in METRICS:
                    rows.append({'comparison':kind,'sequence':source,'external_qp':qp,'metric':metric,
                        'reference':reference,'method':method,'reference_actual_bpp':r['actual_bpp'],
                        'method_actual_bpp':m['actual_bpp'],'rate_difference_bpp':m['actual_bpp']-r['actual_bpp'],
                        'reference_measured_quality':r[metric],'method_measured_quality':m[metric],
                        'measured_delta_method_minus_reference':m[metric]-r[metric],
                        'native_stream_sha_equal':r['native_stream_sha256']==m['native_stream_sha256']})
            for method in ('MEAN','ROLL'):
                chosen=next(x for x in selection if x['mode']==method and x['sequence']==source and int(x['external_qp'])==qp)
                m=next(x for x in out if x['mode']==method and x['sequence']==source and x['external_qp']==qp and x['alpha_index']==int(chosen['alpha_index']))
                r=mapping['MAP']
                for metric in METRICS:
                    rows.append({'comparison':'near_rate_'+method,'sequence':source,'external_qp':qp,'metric':metric,
                        'reference':'MAP','method':method,'selected_alpha':m['alpha'],'rate_match_status':chosen['status'],
                        'reference_actual_bpp':r['actual_bpp'],'method_actual_bpp':m['actual_bpp'],
                        'rate_difference_bpp':m['actual_bpp']-r['actual_bpp'],
                        'rate_difference_percent':100*(m['actual_bpp']-r['actual_bpp'])/r['actual_bpp'],
                        'reference_measured_quality':r[metric],'method_measured_quality':m[metric],
                        'measured_delta_method_minus_reference':m[metric]-r[metric]})
    task.write_csv(P/'matched_rate_comparisons.csv',rows)
    task.write_csv(P/'interpolation_sensitivity.csv',sensitivity)
    return rows,sensitivity


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in METRICS:
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for ax,source in zip(axes.flat,VIDEOS):
            for mode,color in [('A','black'),('MAP','#d62728')]:
                points=sorted([x for x in out if x['mode']==mode and x['sequence']==source],key=lambda x:x['actual_bpp'])
                ax.plot([x['actual_bpp'] for x in points],[x[metric] for x in points],'-o',color=color,label=mode)
            ax.set_title(source);ax.set_xlabel('actual bpp incl. 16-byte v3 header')
            ax.set_ylabel(metric+(' (higher better)' if metric=='PSNR' else ' (lower better)'))
            ax.grid(True,alpha=.3);ax.legend()
        fig.tight_layout();(P/'figures').mkdir(exist_ok=True)
        fig.savefig(P/f'figures/RD_{metric}.png',dpi=160);plt.close(fig)


def main():
    assert json.loads((P/'preflight.json').read_text())['status']=='PASS'
    assert json.loads((P/'completion.json').read_text())['status']=='EVALUATIONS_COMPLETE'
    out=load_results();compare(out);plot(out)
    audit={'status':'PASS','protocol_tasks':len(out),'all_stream_SHA_verified':True,
           'all_byte_accounts_exact':True,'all_64_frame_files_present':True,
           'all_63_transition_files_present':True,'new_full_tasks':len(out)-21,
           'reused_native_anchor_tasks':16,'reused_existing_supplement_tasks':5,'preflight_status':'PASS'}
    (P/'delivery_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('FINALIZE_PASS',len(out))


if __name__=='__main__':main()
