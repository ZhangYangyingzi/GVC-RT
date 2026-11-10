"""Pre-registered mechanical development selection. No final quality input."""
from common_s2 import *
from collections import defaultdict

def points(phase,model,step):
    base=ROOT/'evaluation'/phase/model/f'step_{step:04d}'/'points'
    return [load(p)['result'] for p in sorted(base.glob('*.json'))]

def complete(rows,count=96):
    assert len(rows)==count and len({(r['dataset'],r['sequence'],r['QP']) for r in rows})==count,('INCOMPLETE_EVALUATION',len(rows),count)
    assert all(r[k] is not None and math.isfinite(r[k]) for r in rows for k in ['DISTS','LPIPS_VGG','Sobel_MCResidual_PtoP']), 'MISSING_SELECTION_METRICS'

def means(rows):
    complete(rows);out={}
    for q in [0,5,9]:
        r=[x for x in rows if x['QP']==q];assert len(r)==32
        out[q]={k:sum(x[k] for x in r)/32 for k in ['Sobel_MCResidual_PtoP','DISTS','LPIPS_VGG']}
    return out

def decide(step):
    no=means(points('development','B300_noLoRA_step0',0));original=means(points('development','B300_original_reference',0));history={}
    for g in GROUPS:
        entries=[];best=1.0;patience=0
        for s in [50,100,200,300]:
            if s>step:continue
            raw=points('development',g,s);m=means(raw);S=sum(m[q]['Sobel_MCResidual_PtoP']/no[q]['Sobel_MCResidual_PtoP'] for q in no)/3;Sb=sum(m[q]['Sobel_MCResidual_PtoP']/original[q]['Sobel_MCResidual_PtoP'] for q in no)/3
            guards={str(q):{k:dict(value=m[q][k],noLoRA_limit=1.01*no[q][k],B300_limit=1.01*original[q][k],PASS=m[q][k]<=1.01*no[q][k] and m[q][k]<=1.01*original[q][k]) for k in ['DISTS','LPIPS_VGG']} for q in no}
            qualified=S<1 and Sb<1 and all(v['PASS'] for x in guards.values() for v in x.values())
            improved=qualified and best-S>=.0005
            if improved:best=S;patience=0
            else:patience+=1
            entries.append(dict(step=s,S=S,ratio_vs_B300=Sb,qualified=qualified,guards=guards,qualified_improvement=improved,patience=patience))
        assert entries;eligible=[r for r in entries if r['qualified']];selected=min(eligible or entries,key=lambda r:(r['S'],r['step']));cp=ROOT/g/f'training/checkpoints/step_{selected["step"]:04d}.pt'
        history[g]=dict(history=entries,patience=patience,patience_satisfied=patience>=2,step=selected['step'],S=selected['S'],qualified=bool(eligible),qualification='QUALIFIED' if eligible else 'NOT_QUALIFIED',checkpoint=str(cp),checkpoint_SHA256=sha(cp))
    stopping=all(x['patience_satisfied'] for x in history.values()) or step==300
    out=dict(status='FROZEN' if stopping else 'PROVISIONAL',last_common_step=step,groups=history,stop=stopping,stop_reason='both_groups_patience_two' if all(x['patience_satisfied'] for x in history.values()) else 'maximum_300' if step==300 else None,score='mean_QP(mean_video(PtoP Sobel)/mean_video(noLoRA0 PtoP Sobel))',min_delta=.0005,final_test_quality_used=False,time=time.time())
    frozen=ROOT/'selection.json'
    if frozen.exists() and load(frozen)['status']=='FROZEN':assert load(frozen)==out or load(frozen)['last_common_step']==step;return load(frozen)
    dump(frozen,out);dump(ROOT/f'audits/selection_step_{step:04d}.json',out);return out
if __name__=='__main__':import sys;decide(int(sys.argv[1]))
