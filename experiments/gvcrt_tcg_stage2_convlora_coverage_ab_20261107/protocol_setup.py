"""Freeze all experimental settings; runtime scheduling is chosen only by correctness checks."""
from common_s2 import *

def main():
    ref=load(ROOT/'vendor/reference/protocol.json')
    arch=load(ROOT/'audits/initialization_and_dispatch.json')
    protocol=dict(task='convlora_coverage_ab_20261107',revision='coverage_AB_v1',seed=20261106,
        base_commit='b1f763f87e7239f5a4be2df7fe56d98f33616292',branch='codex/tcg-stage2-convlora-coverage-ab-20261107',
        pilot_sha256=PILOT_SHA,baseline_sha256=BASELINE_SHA,TCG_cumulative_step_frozen_at=6500,
        baseline_composition=ref['baseline_composition'],baseline_and_freezing=ref['model'],
        groups=dict(A_control=dict(targets=['stage3.blocks.0.dc.2','stage3.blocks.1.dc.2','stage3.blocks.2.dc.2','stage3.blocks.3.dc.2','head'],parameters=59392),
            B_expanded=dict(additional_target='upsample.conv1',extra_target=arch['extra_target'],parameters=arch['B_parameters'])),
        architecture_order='stage1 -> stage2 -> tcg_low -> ada3 -> upsample -> tcg_mid -> stage3 -> ada4 -> stage4 -> tcg_out -> ada_final -> head',
        extra_position='upsample.conv1 output before depth_to_space; after tcg_low and ada3, before tcg_mid',
        common_initialization=dict(path='common_initialization/untrained_common.pt',sha256=arch['common_state_sha256'],
            fresh=True,shared_tensors_exact=True,extra_branch_RNG='independent fork with seed20261106; restores caller RNG; no data RNG consumed'),
        actual_TCG_feature_shapes=arch['actual_TCG_feature_shapes'],three_distinct_TCG_resolutions_claimed=False,
        data=ref['data'],training=ref['training'],state=ref['state'],optimizer=ref['optimizer'],
        logical_samples=dict(per_update=4,domains=['2UVG','2ULong'],same_plan_sample_keys_A_B=True,logical_RNG_streams=4,
            two_card_mapping={'0':[0,2],'1':[1,3]},mean_gradient='SUM of four per-sample15frame-mean gradients, divide4 exactly once, clip once, optimizer once'),
        scheduling=dict(preferred='two_card_parallel',A_GPUs=[4,5],B_GPUs=[6,7],fallback='four_card_alternating_segments',
            fallback_GPUs=[4,5,6,7],decision_source='audits/aggregation_comparison.json before formal training'),
        stopping=dict(max_updates=3000,validation_interval=250,min_delta=.0005,patience=4,mode='min',anchor_initial=1.,
            score=ref['stopping']['score'],anchor_update='S < anchor-0.0005; else bad_check_count+=1',
            invalid_or_initial_checks_count=False,raw_best='lowest full trained-checkpoint score, ties earlier',
            independent_groups=True,baseline_retained='best trained score >=1; min_delta status reported separately',
            matched_budget='all common completed planned validation steps; no selection by quality gap'),
        evaluation=dict(**ref['evaluation'],final_test_reused_from_prior_experiments=True,
            final_states=['common baseline','A_control best-trained','B_expanded best-trained'],
            final_selection_after_both_groups_stop=True,historical_baseline_reuse='Only after source/protocol/weight/stream/frame/transition audits; no new decode claim',
            candidate_quality_reuse=False),
        decoding_timing=dict(sequence=sorted([Path(r['video']).stem for r in load(ROOT/'vendor/sealed/manifests/evaluation_rows.json') if r['split']=='development'])[0],
            QP=5,frames=64,physical_GPU=4,warmup_full_sequences=1,repetitions=3,
            timing_scope='RANS+GPU decoder with CUDA synchronization; excludes file I/O, model load and metric networks',
            shared_GPU_interference='Record nvidia process/utilization snapshots; operational timing, no exclusive real-time claim'),
        publication=dict(repo='ZhangYangyingzi/GVC-RT',branch='codex/tcg-stage2-convlora-coverage-ab-20261107',allowlist_only=True,no_weights_media_cache=True,force_push=False),
        resource_limits=ref['resource_limits'])
    # The A/B experiment uses the same optimizer, with normalization adapted to two physical ranks.
    protocol['optimizer']=dict(ref['optimizer'],reduction='four logical samples summed across physical ranks; divide4 once')
    dump(ROOT/'protocol.json',protocol)
    status('engineering_checks','IMPLEMENTING',A_updates=0,B_updates=0)
    print('AB_FIXED_PROTOCOL_WRITTEN',sha(ROOT/'protocol.json'))

if __name__=='__main__':main()
