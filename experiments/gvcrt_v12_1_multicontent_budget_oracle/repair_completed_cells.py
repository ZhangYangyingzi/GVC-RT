#!/usr/bin/env python3
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
for done in sorted((ROOT/'parts').glob('*_done.json')):
    tag=done.name[:-10];base=read(ROOT/'parts'/f'{tag}_baseline.csv')[0];cand=read(ROOT/'parts'/f'{tag}_candidates.csv');sel=read(ROOT/'parts'/f'{tag}_selected.csv') if (ROOT/'parts'/f'{tag}_selected.csv').exists() else []
    for branch in ('compression_aware','generator_aware'):
        if not any(r.get('branch')==branch and r.get('lambda_or_beta')=='baseline_init' for r in cand):
            row={'scope':'cell','dataset':base['dataset'],'video':base['video'],'video_id':base['video_id'],'qp':base['qp'],'branch':branch,'restart':0,'lambda_or_beta':'baseline_init','iteration':0,'frame':'','surrogate_rate':'','actual_bytes':base['actual_bytes'],'actual_bits':base['actual_bits'],'actual_rate_ratio_to_baseline':1.0,'optimization_distortion':'','PSNR':base['PSNR'],'MS_SSIM_or_SSIM':base['MS_SSIM'],'LPIPS':base['LPIPS'],'DISTS':base['DISTS'],'quantized_symbol_hash':'','bitstream_hash':'','decode_status':'PASS','finite_status':'PASS','num_changed_symbols_vs_baseline':0,'L1_symbol_change':0,'L2_latent_change':0,'interpolated_only':False,'is_budget_feasible':True,'is_selected':False,'is_baseline':False,'bitstream_path':str(ROOT/'bitstreams'/f'{tag}_baseline.bin')};cand.append(row)
        feasible=[r for r in cand if r.get('scope')=='cell' and r.get('branch')==branch and str(r.get('is_budget_feasible')).lower()=='true' and r.get('decode_status')=='PASS']
        if not any(r.get('branch')==branch and r.get('selection')=='branch_objective' for r in sel):
            objective=[r for r in feasible if r.get('optimization_distortion') not in ('',None)];best=min(objective,key=lambda r:float(r['optimization_distortion'])) if objective else feasible[0];sel.append(dict(best,selection='branch_objective'))
    write(ROOT/'parts'/f'{tag}_candidates.csv',cand);write(ROOT/'parts'/f'{tag}_selected.csv',sel)
    sync=read(ROOT/'parts'/f'{tag}_sync.csv');payload=json.loads(done.read_text());payload['status']='PASS' if all(r['status']=='PASS' for r in sync) and len(sel)>=2 else 'FAIL';payload['selected_rows']=len(sel);payload['baseline_init_candidate_added']=True;done.write_text(json.dumps(payload,indent=2)+'\n')
