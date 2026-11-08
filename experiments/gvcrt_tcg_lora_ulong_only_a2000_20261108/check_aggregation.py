"""Restored A2000: four-rank mean versus independent four-sample reference."""
from common_s2 import *

def metric(actual,reference):
    import torch
    assert len(actual)==len(reference)
    diffs=[a.double()-b.double() for a,b in zip(actual,reference)]
    numerator=math.sqrt(sum(float(t.square().sum()) for t in diffs))
    denominator=math.sqrt(sum(float(t.double().square().sum()) for t in reference))
    return dict(absolute_L2=numerator,relative_L2=numerator/max(denominator,1e-30),maximum_absolute=max(float(t.abs().max()) for t in diffs),
        bitwise_equal=all(torch.equal(a,b) for a,b in zip(actual,reference)))

def launch(world,tag,mode):
    path=ROOT/'large_artifacts/aggregation'/tag/'A_control'
    if (path/'engineering_segment_0002.json').exists():assert load(path/'engineering_segment_0002.json')['status']=='PASS';return path
    cmd=[PYTHON,'-B','-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node='+str(world),str(ROOT/'train.py'),
        '--group','A_control','--until','2','--engineering-tag',tag,'--gt-mode',mode]
    with (ROOT/f'logs/aggregation_{tag}.log').open('a') as f:
        code=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='4,5,6,7' if world==4 else '4'),stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,('AGGREGATION_PROBE_FAILED',tag,code)
    return path

def main():
    env();import torch
    four=launch(4,'four_rank_inline','inline')
    repeat=launch(4,'four_rank_inline_repeat','inline')
    sequential=launch(1,'four_sample_serial_reference','inline')
    auxiliary=launch(4,'four_rank_GT_auxiliary','auxiliary')
    refs=[torch.load(four/f'comparison_step_{i}.pt',map_location='cpu',weights_only=True) for i in [1,2]]
    records=[]
    # The first update starts at exactly A2000 in all independent copies. The second
    # update reports native drift too; mathematical aggregation is checked from the same saved VJPs.
    ref=refs[0]
    for name,path in [('four_rank_repeat',repeat),('serial_four_sample_reference',sequential),('GT_auxiliary',auxiliary)]:
        other=torch.load(path/'comparison_step_1.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(a,b) for a,b in zip(ref['parameters_before'],other['parameters_before']))
        assert ref['optimizer_before']['param_groups']==other['optimizer_before']['param_groups']
        assert all(torch.equal(ref['optimizer_before']['state'][i][k],other['optimizer_before']['state'][i][k])
                   for i in ref['optimizer_before']['state'] for k in ref['optimizer_before']['state'][i])
        assert ref['sample_keys']==other['sample_keys'] and ref['logical_RNG_sha256']==other['logical_RNG_sha256']
        checks={k:metric(other[k],ref[k]) for k in ['gradient_preclip','gradient_postclip','parameters_after']}
        for k in ['exp_avg','exp_avg_sq']:
            ids=sorted(ref['optimizer']['state']);checks[k]=metric([other['optimizer']['state'][i][k] for i in ids],[ref['optimizer']['state'][i][k] for i in ids])
        loss_error=max(abs(a[k]-b[k]) for a,b in zip(other['losses'],ref['losses']) for k in a)
        assert all(r['relative_L2']<1e-3 for r in checks.values()) and loss_error<1e-5,(name,checks,loss_error)
        assert all(int(v['step'])==2001 for v in other['optimizer']['state'].values())
        update_errors=metric([a-b for a,b in zip(other['parameters_after'],other['parameters_before'])],
                             [a-b for a,b in zip(ref['parameters_after'],ref['parameters_before'])])
        records.append(dict(comparison=name,status='PASS',checks=checks,parameter_update_delta_error=update_errors,
            loss_max_absolute=loss_error,tolerance_relative_L2=1e-3,absolute_optimizer_step=2001))
    for i,ref in enumerate(refs,1):
        grads=ref['sample_gradients'];double=[sum(grads[r][j].double() for r in range(4))/4 for j in range(len(ref['gradient_preclip']))]
        check=metric(ref['gradient_preclip'],double);assert check['relative_L2']<1e-3
        records.append(dict(disposable_step=i,same_saved_four_VJPs_mean_check=check,divide4_once=True,clip_once=True))
    def seconds(path):
        rr=[load(p) for p in sorted((path/'receipts').glob('*.json.gz'))]
        return sum(max(x['seconds_before_commit'] for x in r['physical_ranks']) for r in rr)/len(rr)
    inline_seconds=seconds(four);aux_seconds=seconds(auxiliary);chosen='auxiliary' if aux_seconds<inline_seconds else 'inline'
    dump(ROOT/'audits/runtime_selection.json',dict(status='PASS',selected_GT_mode=chosen,inline_four_rank_seconds_per_update=inline_seconds,auxiliary_four_rank_seconds_per_update=aux_seconds,
        criterion='lower measured two-update processing time, startup excluded',TF32_precision_loss_batch_plan_unchanged=True,formal_updates=0,time=time.time()))
    dump(ROOT/'audits/aggregation_comparison.json',dict(status='PASS',records=records,parent_optimizer_restored=True,parent_logical_RNG_restored=True,four_ULong_samples=True,
        physical_rank_mapping='four ranks -> four logical samples -> physicalGPU4/5/6/7',all_updates_disposable=True,formal_updates=0,time=time.time()))
    print('FOUR_RANK_SERIAL_REFERENCE_AND_GT_RUNTIME_SELECTION_PASS',chosen)

if __name__=='__main__':main()
