"""Frozen numerical screen of complete96-point dev sets at50 and100."""
from common_s2 import *
def points(model,step,phase='development'):
 return [load(p)['result'] for p in sorted((ROOT/'evaluation'/phase/model/f'step_{step:04d}/points').glob('*.json'))]
def complete(rows,n=96):assert len(rows)==n and len({(r['sequence'],r['QP']) for r in rows})==n

def select():
 baseline=points('B300',0);complete(baseline);look={(r['sequence'],r['QP']):r for r in baseline};groups={};gate=load(ROOT/'audits/H2_gate.json')
 for g in GROUPS:
  if g==GROUPS[2] and gate['status']!='PASS':groups[g]=dict(status='MECHANICAL_CHECK_FAILED',qualified=False,selected=None,updates=0,reason='H2 16px control error not separated from8px under unchanged padding');continue
  entries=[]
  for step in [50,100]:
   rows=points(g,step);complete(rows);qp={}
   for q in [0,5,9]:
    items=[r for r in rows if r['QP']==q];assert len(items)==32
    ratios={}
    for metric in ['DISTS','LPIPS_Alex','LPIPS_VGG','FloLPIPS','Sobel_MCResidual_PtoP']:
     assert all(r[metric] is not None for r in items);ratios[metric]=(sum(r[metric] for r in items)/32)/(sum(look[(r['sequence'],q)][metric] for r in items)/32)
    qp[str(q)]=ratios
   S=sum(qp[str(q)]['Sobel_MCResidual_PtoP'] for q in [0,5,9])/3;qualified=S<1 and all(v[k]<=1.005 for v in qp.values() for k in ['DISTS','LPIPS_Alex','LPIPS_VGG','FloLPIPS']);cp=ROOT/g/f'training/checkpoints/step_{step:04d}.pt';entries.append(dict(step=step,S=S,qualified=qualified,per_QP_ratios=qp,checkpoint=str(cp),checkpoint_SHA256=sha(cp)))
  eligible=[x for x in entries if x['qualified']];choice=None
  for e in eligible:
   if choice is None or choice['S']-e['S']>=.0005:choice=e
  groups[g]=dict(status='NUMERICAL_SCREEN_PASS' if choice else 'NOT_QUALIFIED',qualified=bool(choice),selected=choice,history=entries,updates=100)
 out=dict(status='FROZEN',groups=groups,any_qualified=any(g['qualified'] for g in groups.values()),test_quality_used=False,no_step0_substitution=True,time=time.time())
 if (ROOT/'selection.json').exists():assert load(ROOT/'selection.json')['groups']==groups;return load(ROOT/'selection.json')
 dump(ROOT/'selection.json',out);return out
if __name__=='__main__':select()
