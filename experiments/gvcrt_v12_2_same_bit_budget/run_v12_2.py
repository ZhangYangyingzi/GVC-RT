#!/usr/bin/env python3
import argparse,csv,json,math,os,shutil,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];V121=REPO/'experiments/gvcrt_v12_1_multicontent_budget_oracle'
DBG=REPO/'expericent_generation_input/expericent_generator_aware_encoder_latent_rd_oracle_v12_debug';V11=REPO/'expericent_generation_input/expericent_generator_aware_latent_distortion_v11';V9=REPO/'expericent_generation_input/expericent_interface_causal_controls_v9'
for p in (REPO,V121,V9/'src',V11/'b2_latent_direction_magnitude_v11_7/src',DBG/'src'):sys.path.insert(0,str(p))
from common import sha256
from run_v12_1 import read_frames,init_metric_models,run_pass

CFG=json.loads((ROOT/'config.json').read_text());MAN=json.loads((ROOT/'manifest.json').read_text())
def read_csv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def tag(v,qp):return f"{v['dataset']}_{int(v['video_id']):02d}_qp{qp}"
def marker_name(t,b,beta):return ROOT/'parts'/f"{t}_{b}_beta{beta:.12g}.json"

def old_points(v,qp,branch):
    t=tag(v,qp);base=read_csv(V121/'parts'/f'{t}_baseline.csv')[0];rows=read_csv(V121/'parts'/f'{t}_candidates.csv');out=[]
    for r in rows:
        if r.get('scope')!='cell' or r.get('branch')!=branch or r.get('lambda_or_beta')=='baseline_init':continue
        try:beta=float(r['lambda_or_beta'])
        except ValueError:continue
        out.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'branch':branch,'beta':beta,'search_order':len(out),'search_phase':'REUSED_V12_1','baseline_bytes':int(base['actual_bytes']),'actual_bytes':int(r['actual_bytes']),'actual_bits':int(r['actual_bits']),'rate_ratio':float(r['actual_rate_ratio_to_baseline']),'rate_gap_bytes':int(r['actual_bytes'])-int(base['actual_bytes']),'rate_gap_percent':100*(float(r['actual_bytes'])/float(base['actual_bytes'])-1),'over_or_under_budget':'OVER' if int(r['actual_bytes'])>int(base['actual_bytes']) else 'UNDER','bracket_low_beta':'','bracket_high_beta':'','optimization_distortion':r['optimization_distortion'],'PSNR':r['PSNR'],'SSIM':r['MS_SSIM_or_SSIM'],'LPIPS':r['LPIPS'],'DISTS':r['DISTS'],'bitstream_path':r['bitstream_path'],'bitstream_sha256':r['bitstream_hash'],'decode_status':r['decode_status'],'state_sync_status':'PASS','finite_status':r['finite_status'],'reused':True,'is_optimized':True})
    return base,out

def successful(points,budget):return [p for p in points if .995<=p['actual_bytes']/budget<=1]
def bracket(points,budget):
    over=[p for p in points if p['actual_bytes']>budget];under=[p for p in points if p['actual_bytes']<=budget]
    return (min(over,key=lambda p:p['actual_bytes']-budget) if over else None,max(under,key=lambda p:p['actual_bytes']) if under else None)

def next_log_midpoint(points,budget,cache):
    """Return an unevaluated log midpoint from observed over/under pairs.

    Actual RANS bytes need not be monotone in beta.  Ranking endpoint pairs by
    their measured distance to the budget retains the requested closest-rate
    behavior, while trying the next pair prevents a cached midpoint from
    terminating the search prematurely.
    """
    over=[p for p in points if p['actual_bytes']>budget and float(p['beta'])>0]
    under=[p for p in points if p['actual_bytes']<=budget and float(p['beta'])>0]
    choices=[]
    for o in over:
        for u in under:
            a,b=sorted((float(o['beta']),float(u['beta'])))
            if a==b:continue
            mid=math.sqrt(a*b)
            if any(abs(math.log(max(x,1e-12))-math.log(mid))<1e-10 for x in cache):continue
            score=(abs(o['actual_bytes']-budget)+abs(u['actual_bytes']-budget),abs(math.log(b/a)))
            choices.append((score,mid,o,u))
    return min(choices,key=lambda x:x[0]) if choices else None

def evaluate(v,qp,branch,beta,order,phase,budget,frames,device,models,low,high):
    t=tag(v,qp);mark=marker_name(t,branch,beta)
    if mark.exists():
        m=json.loads(mark.read_text())
        if m.get('status')=='PASS' and Path(m['bitstream_path']).exists() and sha256(m['bitstream_path'])==m['bitstream_sha256']:return m['row']
    bs,debug_rows,frame_rows,sync,unchanged,obj=run_pass(v,frames,qp,branch,beta,device,models)
    path=ROOT/'bitstreams'/f'{t}_{branch}_beta{beta:.12g}.bin';path.write_bytes(bs);actual=len(bs);p={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'branch':branch,'beta':beta,'search_order':order,'search_phase':phase,'baseline_bytes':budget,'actual_bytes':actual,'actual_bits':actual*8,'rate_ratio':actual/budget,'rate_gap_bytes':actual-budget,'rate_gap_percent':100*(actual/budget-1),'over_or_under_budget':'OVER' if actual>budget else 'UNDER','bracket_low_beta':'' if low is None else low['beta'],'bracket_high_beta':'' if high is None else high['beta'],'optimization_distortion':obj,'PSNR':sum(float(x['PSNR']) for x in frame_rows)/64,'SSIM':sum(float(x['SSIM']) for x in frame_rows)/64,'LPIPS':sum(float(x['LPIPS']) for x in frame_rows)/64 if frame_rows[0]['LPIPS']!='' else '','DISTS':sum(float(x['DISTS']) for x in frame_rows)/64 if frame_rows[0]['DISTS']!='' else '','bitstream_path':str(path),'bitstream_sha256':sha256(path),'decode_status':'PASS' if all(x['status']=='PASS' for x in sync) else 'FAIL','state_sync_status':'PASS' if all(x['status']=='PASS' for x in sync) else 'FAIL','finite_status':'PASS','reused':False,'is_optimized':True}
    write_csv(ROOT/'parts'/f'{t}_{branch}_beta{beta:.12g}_frame_candidates.csv',debug_rows);write_csv(ROOT/'parts'/f'{t}_{branch}_beta{beta:.12g}_frame_metrics.csv',frame_rows);write_csv(ROOT/'parts'/f'{t}_{branch}_beta{beta:.12g}_state_sync.csv',sync)
    mark.write_text(json.dumps({'status':'PASS' if unchanged and p['decode_status']=='PASS' else 'FAIL','bitstream_path':str(path),'bitstream_sha256':p['bitstream_sha256'],'row':p,'model_unchanged':unchanged},indent=2)+'\n')
    if not unchanged or p['decode_status']!='PASS':raise RuntimeError(f'failed beta {beta}')
    return p

def search(v,qp,branch,device):
    t=tag(v,qp);done=ROOT/'parts'/f'{t}_{branch}_search_done.json'
    prior=None
    if done.exists():
        prior=json.loads(done.read_text())
        if prior.get('status')=='PASS' and (prior.get('same_bit_status')=='REACHED' or prior.get('new_beta_evaluations',0)>=CFG['maximum_new_beta_evaluations_per_cell_branch'] or prior.get('bracket_status')=='NO_UNDER_BUDGET_POINT'):
            print('reuse search',t,branch,flush=True);return
    base,old=old_points(v,qp,branch);points=(prior.get('points',old) if prior and prior.get('status')=='PASS' else old);budget=int(base['actual_bytes']);cache={float(p['beta']):p for p in points};new_count=(prior.get('new_beta_evaluations',0) if prior and prior.get('status')=='PASS' else 0);order=len(points)
    frames=None;models=None
    def run(beta,phase,low,high):
        nonlocal frames,models,new_count,order
        if frames is None:frames=read_frames(v,device);models=init_metric_models(device)
        p=evaluate(v,qp,branch,beta,order,phase,budget,frames,device,models,low,high);cache[beta]=p;points.append(p);new_count+=1;order+=1
        return p
    if not successful(points,budget):
        over,under=bracket(points,budget)
        if under is None:
            for beta in CFG['expansion_betas']:
                if beta in cache:continue
                run(beta,'EXPANSION',over,under);over,under=bracket(points,budget)
                if under is not None or new_count>=CFG['maximum_new_beta_evaluations_per_cell_branch']:break
        while under is not None and over is not None and not successful(points,budget) and new_count<CFG['maximum_new_beta_evaluations_per_cell_branch']:
            choice=next_log_midpoint(points,budget,cache)
            if choice is None:break
            _,mid,pair_over,pair_under=choice
            run(mid,'ADAPTIVE_LOG_MIDPOINT',pair_over,pair_under);over,under=bracket(points,budget)
    over,under=bracket(points,budget);optimized=[p for p in points if p['is_optimized'] and p['actual_bytes']<=budget and p['decode_status']=='PASS' and p['state_sync_status']=='PASS' and p['finite_status']=='PASS']
    exact=[p for p in optimized if p['actual_bytes']==budget];strict=[p for p in optimized if .995<=p['actual_bytes']/budget<1];relaxed=[p for p in optimized if .99<=p['actual_bytes']/budget<.995]
    if exact:selected=min(exact,key=lambda p:float(p['optimization_distortion']));level='EXACT';status='REACHED'
    elif strict:selected=min(strict,key=lambda p:float(p['optimization_distortion']));level='WITHIN_0.5_PERCENT';status='REACHED'
    elif relaxed:selected=min(relaxed,key=lambda p:float(p['optimization_distortion']));level='WITHIN_1_PERCENT';status='REACHED'
    elif optimized:selected=max(optimized,key=lambda p:p['actual_bytes']);level='CLOSEST_UNDER_BUDGET';status='NOT_REACHED'
    else:selected=None;level='NONE';status='NOT_REACHED'
    result={'status':'PASS','dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'branch':branch,'baseline':base,'points':points,'new_beta_evaluations':new_count,'bracket_status':'FOUND' if under is not None and over is not None else ('NO_UNDER_BUDGET_POINT' if under is None else 'NO_OVER_BUDGET_POINT'),'same_bit_level':level,'same_bit_status':status,'selected':selected}
    done.write_text(json.dumps(result,indent=2)+'\n');write_csv(ROOT/'parts'/f'{t}_{branch}_beta_search_trace.csv',points);print('done',t,branch,level,status,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--jobs',required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);device=torch.device('cuda:0');lookup={(v['dataset'],int(v['video_id'])):v for v in MAN['videos']}
    for spec in a.jobs.split(','):
        ds,vid,qp,branch=spec.split(':');search(lookup[(ds,int(vid))],int(qp),branch,device)
if __name__=='__main__':main()
