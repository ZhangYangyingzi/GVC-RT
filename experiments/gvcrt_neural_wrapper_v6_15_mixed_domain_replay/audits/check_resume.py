"""Check persisted optimizer steps and exact future sampling from a saved RNG state."""
import sys,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mixed_io import *
def main():
    import torch
    torch.set_num_threads(2)
    s=torch.load(cp_path(250),map_location='cpu',weights_only=True)
    rows=[json.loads(x) for x in (ROOT/'training_logs/mixed.jsonl').read_text().splitlines()]
    assert len(rows)>=255 and s['adaptation_step']==s['optimizer_update_count']==250
    rng=random.Random();rng.setstate(s['sampling_rng_state']);dr=random.Random();dr.setstate(s['domain_rng_state'])
    random.setstate(s['python_rng_state']);torch.set_rng_state(s['torch_rng_state'])
    sample=module('resume_sample',ROOT/'data.py').sample;checks=[]
    for row in rows[250:255]:
        d='ulong' if dr.random()<.5 else 'uvg';frames,p=sample(d,rng,torch.device('cpu'))
        assert row['domain']==d
        keys=('video','source_frame_indices','crop_x','crop_y','crop_size','cropped_RGB_sha256')
        assert all(row[k]==p[k] for k in keys),(row['adaptation_step'],p)
        checks.append(dict(step=row['adaptation_step'],domain=d,video=p['video'],exact_crop_and_pixels=True));del frames
    assert s['optimizer']['state']
    assert {int(v['step']) for v in s['optimizer']['state'].values()}=={250}
    expected=load(ROOT/'config.json')['optimizer']
    assert all(g['lr']==expected[g['name']+'_lr'] and g['weight_decay']==expected['weight_decay'] for g in s['optimizer']['param_groups'])
    assert all(torch.isfinite(v[k]).all() for v in s['optimizer']['state'].values() for k in ('exp_avg','exp_avg_sq'))
    assert all(s['code_hashes'][n]==sha(ROOT/n) for n in ('train.py','data.py','mixed_io.py','v614_io.py'))
    dump(ROOT/'audits/resume_check.json',dict(status='PASS',checkpoint=str(cp_path(250)),checkpoint_sha256=sha(cp_path(250)),optimizer_entries=len(s['optimizer']['state']),optimizer_step=250,optimizer_moments_finite=True,python_and_torch_rng_loaded=True,domain_and_sampling_rng_exact=True,next_five_sample_checks=checks,cuda_rng_state_present=bool(s['cuda_rng_state'].numel())))
    print('RESUME STATE AND SAMPLING PASS',flush=True)
if __name__=='__main__':main()
