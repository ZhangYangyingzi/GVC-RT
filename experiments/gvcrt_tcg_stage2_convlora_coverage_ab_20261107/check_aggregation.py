"""Actual four-rank reference versus two-rank logical-sample accumulation."""
from common_s2 import *

def metric(actual,reference):
    import torch
    assert len(actual)==len(reference)
    diffs=[a.double()-b.double() for a,b in zip(actual,reference)]
    numerator=math.sqrt(sum(float(t.square().sum()) for t in diffs))
    denominator=math.sqrt(sum(float(t.double().square().sum()) for t in reference))
    return dict(absolute_L2=numerator,relative_L2=numerator/max(denominator,1e-30),maximum_absolute=max(float(t.abs().max()) for t in diffs),
        bitwise_equal=all(torch.equal(a,b) for a,b in zip(actual,reference)))

def launch(group,world,tag,anchor=None):
    gpu='4,5,6,7' if world==4 else ('4,5' if group=='A_control' else '6,7')
    path=ROOT/'large_artifacts/aggregation'/tag/group
    if (path/'engineering_segment_0002.json').exists():
        assert load(path/'engineering_segment_0002.json')['status']=='PASS';return path
    cmd=[PYTHON,'-B','-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node='+str(world),str(ROOT/'train.py'),
        '--group',group,'--until','2','--engineering-tag',tag]
    if anchor is not None:cmd+=['--reference-anchor',str(anchor/'checkpoints')]
    with (ROOT/f'logs/aggregation_{group}_{tag}.log').open('a') as f:
        result=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID'),stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0,('ENGINEERING_AGGREGATION_SUBPROCESS_FAILED',group,world,result.returncode)
    return path

def main():
    env()
    import torch
    records=[];passed=True
    for group in GROUPS:
        reference=launch(group,4,'reference4_anchored_v2')
        repeated=launch(group,4,'repeat4_anchored_v2',reference)
        accumulated=launch(group,2,'accumulated2_anchored_v2',reference)
        for step in [1,2]:
            ref=torch.load(reference/f'comparison_step_{step}.pt',map_location='cpu',weights_only=True)
            rep=torch.load(repeated/f'comparison_step_{step}.pt',map_location='cpu',weights_only=True)
            two=torch.load(accumulated/f'comparison_step_{step}.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(a,b) for a,b in zip(ref['parameters_before'],two['parameters_before'])),'COMPARISON_PARAMETERS_NOT_IDENTICAL'
            assert all(torch.equal(a,b) for a,b in zip(ref['parameters_before'],rep['parameters_before']))
            assert ref['sample_keys']==rep['sample_keys']==two['sample_keys']
            assert ref['logical_RNG_sha256']==rep['logical_RNG_sha256']==two['logical_RNG_sha256']
            checks={key:metric(two[key],ref[key]) for key in ['gradient_preclip','gradient_postclip','parameters_after']}
            ref_repeat={key:metric(rep[key],ref[key]) for key in ['gradient_preclip','gradient_postclip','parameters_after']}
            update=lambda x:[a-b for a,b in zip(x['parameters_after'],x['parameters_before'])]
            updates=metric(update(two),update(ref));repeat_updates=metric(update(rep),update(ref))
            opt_checks={}
            for name in ['exp_avg','exp_avg_sq']:
                order=sorted(ref['optimizer']['state'])
                opt_checks[name]=metric([two['optimizer']['state'][i][name] for i in order],[ref['optimizer']['state'][i][name] for i in order])
            assert all(int(v['step'])==step for v in two['optimizer']['state'].values())
            loss_max=max(abs(a[k]-b[k]) for a,b in zip(two['losses'],ref['losses']) for k in a)
            # Exact same measured local sample VJPs replay both aggregation orders,
            # separating SUM/4 implementation from repeat-level GPU VJP variability.
            gs=ref['sample_gradients'];count=len(ref['gradient_preclip'])
            double_mean=[sum(gs[r][j].double() for r in range(4))/4 for j in range(count)]
            two_order=[((gs[0][j]+gs[2][j])+(gs[1][j]+gs[3][j]))/4 for j in range(count)]
            cached=metric(two_order,double_mean)
            four_reference_mean=metric(ref['gradient_preclip'],double_mean)
            assert four_reference_mean['relative_L2']<1e-3,'FOUR_RANK_REFERENCE_AGGREGATION_FAILED'
            this=all(v['relative_L2']<1e-3 for v in [*checks.values(),*opt_checks.values(),cached]) and loss_max<1e-5
            passed &= this
            records.append(dict(group=group,disposable_step=step,status='PASS' if this else 'FAILED_TOLERANCE',
                tolerance_relative_L2=1e-3,loss_maximum_absolute_difference=loss_max,checks=checks,optimizer_state_checks=opt_checks,
                parameter_update_delta_errors=updates,four_rank_repeat_errors=ref_repeat,four_rank_repeat_update_delta_errors=repeat_updates,
                cached_same_VJP_two_order_mean_vs_double_reference=cached,four_rank_mean_vs_double_reference=four_reference_mean,logical_sample_keys_equal=True,logical_RNG_streams_equal=True,
                identical_pre_update_checkpoint=True,parameter_update_delta_reported_separately=True,bitwise_equivalence_claim=False))
    mode='two_card_parallel' if passed else 'four_card_alternating_segments'
    dump(ROOT/'audits/aggregation_comparison.json',dict(status='PASS' if passed else 'TWO_CARD_FAILED_FOUR_CARD_FALLBACK',
        selected_mode=mode,groups=records,effective_batch=4,normalization='sum four logical VJPs, divide4 once, clip once, update once',
        all_engineering_updates_discarded=True,formal_updates=0,time=time.time()))
    print('AGGREGATION_CHECK_FINISHED',mode)

if __name__=='__main__':main()
