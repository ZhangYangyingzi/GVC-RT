"""Matched 1001--1500 continuation, exact optimizer/RNG restoration, atomic checkpoints."""
import argparse,fcntl,random,math,traceback
from v68_io import *
from model_runtime import *
from objective import Objective
def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=BRANCHES,required=True);p.add_argument('--gpu',choices=(4,5,6,7),type=int,required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);cfg=frozen();br=ROOT/'branches'/a.branch;lock=(br/'parts/train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert load(ROOT/'audits/interface_preservation_feasibility.json')['status']=='PASS';cal=load(ROOT/'audits/interface_alignment_calibration.json');assert cal['status']=='PASS'
    weight=cal['lambda_align'] if a.branch==C1 else 0.;device=torch.device('cuda:0');v4,im,pm,models,opt,quality,state=setup(device)
    initial={k:v4.module_hash(m) for k,m in models.items()};teacher=Teacher(v4,device) if a.branch==C1 else None;restore_rng(state)
    init=dict(status='PASS',source_sha256=sha(SOURCE),module_hashes=initial,optimizer=optimizer_summary(opt.state_dict()),compression_hash=v4.compression_hash(im,pm),rng_hashes={k:state_hash(state[k]) for k in ('sample_rng_state','torch_rng_state','cuda_rng_state')},architecture={k:[(n,list(p.shape)) for n,p in m.named_parameters()] for k,m in models.items()},plans_sha256=sha(ROOT/'audits/continuation_training_plans.json'))
    dump(br/'initialization_audit.json',init)
    assert init['module_hashes']==load(ROOT/'audits/source_B1000_audit.json')['module_hashes'] and init['optimizer']==load(ROOT/'audits/source_B1000_audit.json')['optimizer']
    plans=load(ROOT/'audits/continuation_training_plans.json')['plans'];rng=random.Random();rng.setstate(state['sample_rng_state']);start=1000
    hashpath=br/'checkpoint_hashes.json';hashes=load(hashpath) if hashpath.exists() else {'1000':dict(path=str(SOURCE),sha256=sha(SOURCE),source_reference=True,step=1000,module_hashes=initial,compression_hash=cfg['compression_hash'])}
    code={p.name:sha(p) for p in ROOT.glob('*.py')}
    if max(map(int,hashes))>1000:
        start=max(map(int,hashes));path=checkpoint(a.branch,start);assert sha(path)==hashes[str(start)]['sha256'];s=torch.load(path,map_location='cpu',weights_only=True)
        assert s['code_hashes']==code and s['config_sha256']==sha(ROOT/'config.json') and s['lambda_align']==weight
        for k,m in models.items():m.load_state_dict(s[k],strict=True)
        opt.load_state_dict(s['optimizer']);rng.setstate(s['sample_rng_state']);restore_rng(s)
    else:dump(hashpath,hashes)
    fn=Objective(v4,im,pm,models,quality,cfg,teacher,weight);params=[p for m in models.values() for p in m.parameters()];begin=time.time()
    logs=ROOT/'training_logs'/f'{a.branch}.jsonl';rows=[json.loads(x) for x in logs.read_text().splitlines()] if logs.exists() else []
    if any(r['absolute_step']>start for r in rows):
        dump(br/'logs'/f'pre_resume_{time.time_ns()}.json',rows);rows=[r for r in rows if r['absolute_step']<=start];logs.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert [r['absolute_step'] for r in rows]==list(range(1001,start+1))
    csvpath=ROOT/'training_logs'/f'{a.branch}.csv'
    if rows:write(csvpath,rows)
    def save(step):
        assert v4.compression_hash(im,pm)==cfg['compression_hash'];path=checkpoint(a.branch,step);assert not path.exists()
        payload=dict(step=step,branch=a.branch,**{k:m.state_dict() for k,m in models.items()},optimizer=opt.state_dict(),sample_rng_state=rng.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),compression_hash=cfg['compression_hash'],lambda_struct=cfg['lambda_struct'],lambda_align=weight,source_B1000_sha256=sha(SOURCE),training_plan_sha256=init['plans_sha256'],config_sha256=sha(ROOT/'config.json'),code_hashes=code)
        tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
        hashes[str(step)]=dict(path=str(path),sha256=sha(path),step=step,branch=a.branch,module_hashes={k:v4.module_hash(m) for k,m in models.items()},optimizer=optimizer_summary(opt.state_dict()),**{k:payload[k] for k in ('compression_hash','lambda_struct','lambda_align','source_B1000_sha256','training_plan_sha256','config_sha256','code_hashes')});dump(hashpath,hashes)
    with logs.open('a',buffering=1) as log:
        for step in range(start+1,1501):
            expected=plans[step-1001];frames,plan=sample_clip(rng,device,v4);assert all(plan[k]==expected[k] for k in ('sample_id','start','crop_x','crop_y','frame_sha256','temporal_indices'))
            opt.zero_grad(set_to_none=True);s,actual=fn(frames,expected['external_qp']);assert actual==expected['actual_qps']
            norms={k:v4.grad_norm(m.parameters()) for k,m in models.items()};assert all(math.isfinite(x) and x>0 for x in norms.values())
            grad=float(torch.nn.utils.clip_grad_norm_(params,cfg['optimizer']['grad_clip']));assert math.isfinite(grad);opt.step()
            pn={k:norm(list(m.parameters())) for k,m in models.items()};assert all(math.isfinite(x) for x in pn.values())
            row=dict(branch=a.branch,**expected,**s,total_loss=s['loss'],gradient_norm=grad,**{k+'_gradient_norm':x for k,x in norms.items()},**{k+'_parameter_norm':x for k,x in pn.items()})
            log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
            with csvpath.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row))
                if step==1001:w.writeheader()
                w.writerow(row)
            if step in (1250,1500):log.flush();os.fsync(log.fileno());save(step)
            if step%5==0 or step==start+1:
                dump(br/'training_status.json',dict(status='RUNNING',absolute_step=step,continuation_step=step-1000,target=1500,gpu=a.gpu,PID=os.getpid(),seconds_per_update=(time.time()-begin)/(step-start),updated_unix=time.time()));print('TRAIN',a.branch,step,'/1500',flush=True)
    fn.close();after={k:v4.module_hash(m) for k,m in models.items()};changed={k:after[k]!=initial[k] for k in initial};assert all(changed.values())
    assert v4.compression_hash(im,pm)==cfg['compression_hash'];teacher_changed=False
    if teacher:teacher.assert_frozen();teacher_changed=teacher.hash()!=teacher.hashes;assert not teacher_changed
    dump(ROOT/'audits'/f'parameter_update_audit_{a.branch[:2]}.json',dict(status='PASS',before=initial,after=after,changed=dict(changed,compression_core=False,teacher=teacher_changed),updates=500))
    dump(br/'training_status.json',dict(status='PASS',absolute_step=1500,continuation_step=500,gpu=a.gpu,PID=os.getpid(),updated_unix=time.time()));print('TRAIN PASS',a.branch,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs/failures'/f'train_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
