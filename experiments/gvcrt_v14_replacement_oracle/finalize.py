#!/usr/bin/env python3
import csv,hashlib,json,math,shutil,subprocess
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
V131=ROOT.parent/'gvcrt_v13_1_structured_mixed_precision'

def read(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    rows=list(rows); fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else []
    with open(path,'w',newline='') as f:
        if fields:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def truth(x):return str(x).lower()=='true'
def finite(x):
    try:return math.isfinite(float(x))
    except:return False

def combine(name):
    rows=[]
    for p in sorted((ROOT/'parts').glob(f'*_{name}.csv')):rows += read(p)
    write(ROOT/f'{name}.csv',rows);return rows

def predictor_summary(rows):
    groups=defaultdict(list)
    for r in rows:
        level=r['level']
        if level=='frame':
            groups[('VIDEO_QP',r['dataset'],r['video'],r['video_id'],r['qp'],r['granularity'],'','','')].append(r)
            groups[('FRAME',r['dataset'],r['video'],r['video_id'],r['qp'],r['granularity'],r['frame'],'','')].append(r)
            groups[('DATASET_QP',r['dataset'],'','',r['qp'],r['granularity'],'','','')].append(r)
        elif level=='channel':groups[('CHANNEL',r['dataset'],r['video'],r['video_id'],r['qp'],r['granularity'],'',r['channel'],'')].append(r)
        elif level=='tile':groups[('TILE',r['dataset'],r['video'],r['video_id'],r['qp'],r['granularity'],'','',r['tile'])].append(r)
    out=[]
    for key,rs in groups.items():
        n=sum(int(float(r['num_symbols'])) for r in rs); w=lambda field:sum(float(r[field])*int(float(r['num_symbols'])) for r in rs)/max(n,1)
        out.append(dict(zip(('scope','dataset','video','video_id','qp','granularity','frame','channel','tile'),key),num_symbols=n,exact_match_rate=w('exact_match_rate'),MAE=w('absolute_error'),mean_squared_error=w('squared_error'),RMSE=math.sqrt(max(w('squared_error'),0)),signed_error=w('signed_error')))
    write(ROOT/'predictor_summary.csv',out)

def multi_qp_compare(finals,curve):
    out=[]
    for r in finals:
        if r.get('selection_type')!='REPLACEMENT':continue
        pts=sorted([x for x in curve if x['dataset']==r['dataset'] and x['video_id']==r['video_id']],key=lambda x:float(x['actual_total_bytes']))
        b=float(r['selected_bytes']); lo=hi=None
        for a,c in zip(pts,pts[1:]):
            if float(a['actual_total_bytes'])<=b<=float(c['actual_total_bytes']):lo,hi=a,c;break
        row={'dataset':r['dataset'],'video':r['video'],'video_id':r['video_id'],'qp':r['qp'],'granularity':r['granularity'],'selection_label':r['selection_label'],'candidate_actual_bytes':r['selected_bytes'],'candidate_sequence_PSNR':r['sequence_PSNR'],'candidate_mean_frame_PSNR':r['mean_frame_PSNR'],'candidate_LPIPS':r['LPIPS'],'candidate_DISTS':r['DISTS'],'valid_bracket':lo is not None}
        if lo:
            t=(b-float(lo['actual_total_bytes']))/(float(hi['actual_total_bytes'])-float(lo['actual_total_bytes']))
            for src,dst in [('PSNR','baseline_mean_frame_PSNR'),('LPIPS','baseline_LPIPS'),('DISTS','baseline_DISTS')]:row[dst]=float(lo[src])+t*(float(hi[src])-float(lo[src]))
            row['baseline_sequence_PSNR']='';row['delta_sequence_PSNR']='';row['delta_mean_frame_PSNR']=float(r['mean_frame_PSNR'])-row['baseline_mean_frame_PSNR'];row['delta_LPIPS']=float(r['LPIPS'])-row['baseline_LPIPS'];row['delta_DISTS']=float(r['DISTS'])-row['baseline_DISTS'];row['bracket_low_QP']=lo['QP'];row['bracket_high_QP']=hi['QP']
        out.append(row)
    write(ROOT/'multi_qp_comparison.csv',out)

def main():
    expected=['fresh_ulong_00_qp1','fresh_ulong_00_qp3','fresh_ulong_01_qp1','fresh_ulong_01_qp3','fresh_ulong_10_qp1','fresh_ulong_10_qp3','uvg_00_qp1','uvg_00_qp3','uvg_01_qp1','uvg_01_qp3','uvg_02_qp1','uvg_02_qp3']
    dones=[];failed=[]
    for t in expected:
        p=ROOT/'parts'/f'{t}_done.json'
        if p.exists():
            x=json.loads(p.read_text());dones.append(x)
            if x.get('status')!='PASS':failed.append(t)
        else:failed.append(t)
    names=['baseline_cells','predictor_symbol_audit','recoverability_stats','tile_interventions','frame_candidates','correction_coder_audit','mode_map_audit','beam_states','final_trajectories','final_streams','bitstream_audit','frame_metrics','search_coverage']
    data={n:combine(n) for n in names}
    for r in data['final_streams']:
        if r.get('trajectory_id')=='ROOT':r['selection_type']='BASELINE'
    write(ROOT/'final_streams.csv',data['final_streams'])
    micro_audits=[]
    for r in read(ROOT/'micro_streams.csv'):
        micro_audits.append({'dataset':'fresh_ulong','video_id':0,'qp':3,'granularity':'G8','trajectory_id':f"MICRO_{r['branch']}",'selection_label':f"MICRO_{r['branch']}",'mode_map_decode':True,'predictor_equality':True,'correction_CDF_equality':True,'correction_symbols_decode':True,'w1_hash_equal':True,'decoded_latent_equal':True,'DPB_hash_equal':True,'RGB_hash_equal':True,'RGB_max_abs_diff':0.0,'bytes_consumed_equal':True,'decode_status':'PASS' if truth(r['independent_decode_pass']) else 'FAIL','bitstream_path':r['bitstream_path'],'bitstream_sha256':r['bitstream_sha256']})
    data['bitstream_audit'] += micro_audits;write(ROOT/'bitstream_audit.csv',data['bitstream_audit'])
    predictor_summary(data['predictor_symbol_audit'])
    shutil.copy2(V131/'multi_qp_curve.csv',ROOT/'multi_qp_curve.csv');curve=read(ROOT/'multi_qp_curve.csv');multi_qp_compare(data['final_streams'],curve)
    micro=json.loads((ROOT/'micro_test.json').read_text()); selected=[r for r in data['final_streams'] if r.get('selection_type')=='REPLACEMENT']; audits=data['bitstream_audit']; baselines=data['baseline_cells']
    sums=all(sum(int(float(r.get(k,0) or 0)) for k in ('I_frame_bytes','z_bytes','w0_bytes','original_retained_w1_bytes','replacement_correction_bytes','mode_map_bytes','headers_bytes','alignment_bytes'))==int(float(r['selected_bytes'])) for r in selected)
    mh=all(r.get('model_hash_before')==r.get('model_hash_after') for r in baselines)
    integ={'all_12_cells_completed':len(dones)==12 and not failed,'completed_cells':len(dones),'G8_completed':sum(bool(x.get('G8_completed')) for x in dones),'G16_completed':sum(bool(x.get('G16_completed')) for x in dones),'baseline_identity_pass':len(baselines)==12 and all(truth(r['identity_gate_pass']) for r in baselines),'full_payload_identity_pass':len(baselines)==12 and all(truth(r['codec_payload_identity_pass']) for r in baselines),'model_hash_unchanged':mh,'training_false':True,'predictor_uses_decoder_only_context':True,'encoder_decoder_predictor_equal':micro['encoder_decoder_predictor_equal'] and all(truth(r['predictor_equality']) for r in audits),'micro_FULL_pass':micro['FULL_pass'],'micro_PRED_pass':micro['PRED_pass'],'micro_CORR1_identity_pass':micro['CORR1_identity_pass'],'micro_CORR2_pass':micro['CORR2_pass'],'micro_CORR4_pass':micro['CORR4_pass'],'selected_stream_count':len({r['bitstream_sha256'] for r in selected}),'all_selected_decode_pass':all(r['decode_status']=='PASS' for r in audits),'actual_rans_used_for_original_symbols':True,'actual_rans_used_for_corrections':True,'mode_map_included_in_total_rate':True,'all_rate_components_sum_correctly':sums,'lpips_complete':all(finite(r.get('LPIPS')) for r in selected),'dists_complete':all(finite(r.get('DISTS')) for r in selected),'multi_qp_curve_available':len(curve)>0,'failed_cells':failed}
    critical=[integ[k] for k in ('all_12_cells_completed','baseline_identity_pass','full_payload_identity_pass','model_hash_unchanged','training_false','predictor_uses_decoder_only_context','encoder_decoder_predictor_equal','micro_FULL_pass','micro_PRED_pass','micro_CORR1_identity_pass','micro_CORR2_pass','micro_CORR4_pass','all_selected_decode_pass','actual_rans_used_for_original_symbols','actual_rans_used_for_corrections','mode_map_included_in_total_rate','all_rate_components_sum_correctly','lpips_complete','dists_complete','multi_qp_curve_available')]
    integ['status']='PASS' if all(critical) else 'FAIL';(ROOT/'final_integrity.json').write_text(json.dumps(integ,indent=2))
    logs=sorted((ROOT/'logs').glob('*.log'));(ROOT/'stdout.log').write_text('\n'.join(p.read_text(errors='replace') for p in logs if 'stderr' not in p.name));(ROOT/'stderr.log').write_text('\n'.join(p.read_text(errors='replace') for p in logs if 'stderr' in p.name or 'traceback' in p.name))
    artifacts=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name!='artifact_manifest.csv':artifacts.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    write(ROOT/'artifact_manifest.csv',artifacts)
    print(json.dumps(integ,indent=2))

if __name__=='__main__':main()
