"""Finalize the user-fixed protocol before new quality scoring or formal updates."""
from common_s2 import *

def main():
    old=load(ROOT/'vendor/previous_protocol.json');request=load(ROOT/'requested_protocol.json')
    assert request['pilot_sha256']==PILOT_SHA and sha(PILOT)==PILOT_SHA and sha(BASELINE)==BASELINE_SHA
    assert sha(ROOT/'loss.py')==old['training']['loss_source_sha256']
    protocol=dict(task=request['task'],revision='generator_convlora_v1_corrected_SHA',seed=SEED,
        base_commit=request['base_commit'],branch=request['branch'],pilot_sha256=PILOT_SHA,baseline_sha256=BASELINE_SHA,
        baseline_cumulative_TCG_step=6500,baseline_composition='Original I/P codec plus strict pilot Bridge/generator/persistent buffers plus exact FP32 B+1500 TCG overlay; not original GVC-RT weights',
        SHA_correction_audit='audits/sha_correction_20261106/correction.json',
        model=dict(Router_instantiated=False,TCG_frozen=True,original_generator_frozen=True,Bridge_frozen=True,codec_frozen=True,quality_networks_frozen=True,
            targets=request['targets'] if 'targets' in request else ['stage3.blocks.0.dc.2','stage3.blocks.1.dc.2','stage3.blocks.2.dc.2','stage3.blocks.3.dc.2','head'],
            stage3='Four parallel low-rank updates alongside existing depthwise3x3; not dense-convolution weight factorization',
            stage3_branch=[320,4,320],head_branch=[320,4,192],head_geometry='Original head produces192 channels before PixelShuffle8',
            bias=False,rank=4,alpha_div_rank=1,parameters=59392,initial_down='Kaiming uniform, fixed seed20261106',initial_up='exact zeros',
            LoRA_internal='FP32, FP32 addition then cast once to incoming dtype',TCG='FP32 parameters and computation; exact B1500 state remains frozen',
            deployment='Original nonTCG FP16; matching baseline/candidate stage3/head safe PyTorch dispatch; codec dispatch unchanged',
            custom_CUDA='No new fused kernel; both explicit CUDA and PyTorch generator entry routes execute LoRA; runtime backend availability recorded',
            additional_modules_or_losses=False),
        data=old['data'],training=old['training'],state=dict(TCG_cache='detached pre-residual current',codec_reference='unchanged feature DPB',previous_reconstruction_detach=True,BPTT=False),
        optimizer=dict(type='AdamW',new=True,groups=[dict(name='LoRA',lr=5e-5)],betas=[.9,.999],eps=1e-8,weight_decay=1e-4,global_clip_norm=1.,reduction='SUM4/4 before one clipping and optimizer update',schedule=None),
        stopping=dict(max_updates=3000,validation_interval=250,patience=4,min_delta=.001,mode='min',anchor_initial=1.,
            score='mean_q(mean32video FloLPIPS(candidate,q)/mean32video FloLPIPS(B1500 matched-FP32 safe-dispatch baseline,q))',
            raw_best='trained checkpoints >=250, ties earlier',initialization_validation_counts=False,invalid_validation_counts=False,
            anchor_update='score < anchor-0.001; otherwise bad_check_count+=1',final_selection='lower of baseline1 and best trained score; tie baseline',
            keep_and_test_best_trained_even_if_baseline_retained=True,selection_before_final_test=True),
        evaluation=dict(development_videos=32,development_QPs=[0,5,9],development_points=96,final_videos=45,final_QPs=[0,5,9],final_points=135,frames=64,
            baseline='B+1500 matched-FP32 TCG with same safe dispatch, no LoRA or optimizer',candidate='LoRA-only selected checkpoint',
            bitstreams='Same pre-encoded verified B streams for both baseline and every candidate; new independent decoder per point; no claimed fresh encoding',
            bpp_and_bitstream_SHA='Exact per-point equality required before quality scoring; any mismatch stops evaluation',
            independent_fresh_RANS_engineering_probe=True,RGB='matched_rgb_fp32_v2: float before clamp/range conversion/native crop',
            metrics=old['evaluation']['metrics'],datasets=old['evaluation']['datasets'],inherited_PWC_sampler=old['evaluation']['inherited_PWC_sampler'],
            weighting='Equal videos within dataset/split/QP. Normalized dev score averages the three per-QP ratios.',historical_quality_metrics_reused=False,BD_rate=False),
        physical_GPUs=[4,5,6,7],world_size=4,effective_batch=4,resource_limits=old['resource_limits'],
        publication=dict(repo='ZhangYangyingzi/GVC-RT',branch=request['branch'],allowlist_only=True,no_weights_media_streams_caches=True,max_file_bytes=50*2**20,force_push=False))
    dump(ROOT/'protocol.json',protocol)
    (ROOT/'README.md').write_text('# Generator Conv-LoRA\n\nOne LoRA-only line from strictly loaded pilot+B1500. Five rank4 branches,59392 parameters; Router absent and all original parameters frozen.\n\nRun/resume (the agent executes this entry):\n\n```bash\nCUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 '+PYTHON+' -B -u '+str(ROOT/'runner.py')+'\n```\n\nState: status.json. Results: REPORT.md. Publication status is separate. Historical SHA blocker is preserved in audits/sha_correction_20261106/.\n')
    status('implementation_and_engineering','IMPLEMENTING',committed_optimizer_updates=0,blocker=None)
    print('FIXED_PROTOCOL_WRITTEN',sha(ROOT/'protocol.json'))

if __name__=='__main__':main()
