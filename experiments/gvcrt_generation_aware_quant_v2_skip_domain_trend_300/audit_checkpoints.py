"""CPU-only verification of stored update counters, initialization and finite states."""
import os,sys,json,hashlib,gc
from pathlib import Path
P=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def named_hash(items):
    h=hashlib.sha256()
    for name,value in items:
        h.update(name.encode());h.update(value.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    torch.set_num_threads(2)
    pre=json.loads((P/'preflight.json').read_text());initial=json.loads((P/'initialization_audit.json').read_text())
    out={'status':'PASS','device':'CPU','torch':torch.__version__,'compiled_cuda_version':torch.version.cuda,'checkpoints':[]}
    for b in ['H10','H01','H11']:
        for step in [0,100,300]:
            path=P/'server_artifacts'/b/f'step_{step}.pt';s=torch.load(path,map_location='cpu',weights_only=False)
            assert s['step']==s['cursor']==step and s['branch']==b
            assert s['plan_hash']==json.loads((P/'train_plan.json').read_text())['canonical_plan_sha256']
            assert s['config_sha256']==pre['config_sha256']
            for name in ['runtime.py','train.py']:assert s['code_sha256'][name]==sha(P/name)
            counters={int(v['step']) for v in s['optimizer']['state'].values()}
            assert counters==({step} if step else set())
            r={'branch':b,'step':step,'checkpoint_sha256':sha(path),'optimizer_steps':sorted(counters),'G_hash':named_hash(s['G'].items()),'H_hash':named_hash(s['H'].items()),'all_weights_finite':True}
            for mod in ['G','H']:
                assert all(torch.isfinite(v).all() for v in s[mod].values())
                if step==0:assert r[mod+'_hash']==initial[b][mod]
            out['checkpoints'].append(r)
            del s;gc.collect()
    (P/'checkpoint_content_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print('CHECKPOINT_CONTENT_AUDIT_PASS',len(out['checkpoints']))

if __name__=='__main__':main()
