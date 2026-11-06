"""Write the prespecified joint protocol before reading any new quality scores."""
from common_s2 import *
def main():
    env();assert not (ROOT/'protocol.json').exists(),'PROTOCOL_ALREADY_EXISTS; inspect and resume rather than overwrite'
    previous=load(PREVIOUS/'protocol.json')
    p=dict(task='tcg_stage2_history_gate_joint_20261106',revision='internal_history_gate_joint_v1',seed=SEED,
      baseline_checkpoint=str(BASELINE),baseline_sha256=BASELINE_SHA,baseline_cumulative_tcg_step=6500,
      receiver_initialization=dict(pilot=str(PILOT),sha256=PILOT_SHA,loading='strict original I/P, strict pilot Bridge/generator/buffers, exact FP32 B+1500 TCG overlay, new Router'),
      previous_Router_checkpoint_loaded=False,TCG_fusion_rezeroed_after_load=False,
      trainable_modules=['Router','tcg_low','tcg_mid','tcg_out'],parameter_counts=dict(Router=33889,TCG=137760,total=171649),
      frozen=['I_model','P_codec_and_entropy','Bridge','non_TCG_generator','LPIPS','DISTS','RAFT'],
      gate=dict(position='after_previous_adapter_before_concat',formula='fusion(cat(previous_adapter(previous.float())*(1-resize(M.float())),current_adapter(current.float())))',
          residual='(current.float()+delta).to(incoming_dtype)',mask_type='single_channel_spatial_shared_across_three_TCG_layers',
          resize='bilinear align_corners=False',old_outer_residual_gate=False,current_branch_kept=True,
          M1='zeros directly projected previous branch; current branch and residual may remain; NOT a TCG-off mode',
          M0='unattenuated history at the CURRENT weights; trained-joint M0 is not original B+1500'),
      router=dict(decoded_C=256,input='Concat(pm.dec current, detached previous decoded feature, abs difference)',input_channels=768,
          architecture='Conv1x1(768,32),SiLU,Conv3x3(32,32,pad1),SiLU,Conv1x1(32,1),Sigmoid',
          spatial_pooling=False,initial_M=.2,last_weight=0,last_bias=math.log(.2/.8),first_two_layers_seed=SEED,
          additional_inputs=[],additional_heads=False),
      precision=dict(codec='historical FP16',training_Bridge_and_nonTCG_generator='historical FP32',evaluation_Bridge_and_nonTCG_generator='historical FP16',
          Router_and_three_TCG_parameters='FP32; promoted before B state loading, no FP16 round trip',
          Router_and_TCG_forward='autocast disabled; projection, mask interpolation, 1-mask, history product, fusion and current+delta FP32',
          TCG_output='cast back to incoming dtype after FP32 addition; final rounding still possible',
          parent_half_or_to='dtype islands preserve original FP32 values and device movement',backend_TF32='record inherited actual backend flags; no whole-model precision conversion',
          scoring='matched_rgb_fp32_v2: x.float(), clamp(-1,1), (x+1)/2, native crop'),
      state=dict(TCG_cache='detached pre-residual current in incoming dtype',Router_history='detached decoded codec feature; read previous then commit current',
          codec_DPB='original decoded feature and existing codec reference rules; no gated or generator features',
          update_buffer_false='neither Router nor TCG history advances',internal_reference='use_tcg=False/update_buffer=False bypasses both',
          reset='sequence/I/GOP, compatible dimension boundary',BPTT=False),
      data=dict(B_ULong_pool=8470,quarantined_source_groups=236,plan_index_sha256=sha(ROOT/'plans/index.json'),
          source_cursor_start=1500,source_cursor_at_cap=4500,immutable_B_rows='6000:18000',plan_order='unchanged',effective_batch=4,
          samples_per_update=dict(uvg=2,ulong=2),new_independent_samples_claim=False,split_and_source_hashes='inherited sealed audited manifests',
          Vimeo=False,excluded_identities_reintroduced=False),
      training=dict(frames=64,crop=[256,256],I0='original I',warmup=[1,48],supervised=[49,63],frame_mean_denominator=15,
          external_QPs=list(range(10)),internal_QP='inherited frame_qp/shift_qp',previous_reconstruction_detach=True,codec_reference_detach=True,
          loss='LPIPS+0.5*DISTS+0.1383589248954108*RGB_L1+0.01*RAFT_motion+beta(q)*Rate+0.01*proxy',
          loss_source_sha256=sha(ROOT/'loss.py'),beta='1/(0.08+q*(0.9-0.08)/9)',proxy='Identity -> zero',rate_gradient='none when codec frozen',
          VJP='historical fused/15 then reconstruction*(4096/15), unscale gradient exactly once by4096, multiply0.1383589248954108, combine',
          RAFT=dict(local_weight_sha256=RAFT_SHA,download=False,parameters_frozen=True,prediction_input_differentiable=True,transforms='weights enum transforms'),
          paper_identical_loss_claim=False,new_losses=[],smoke='disposable engineering states; formal first2 included in budget'),
      optimizer=dict(type='AdamW',fresh_state=True,historical_optimizer_loaded=False,groups=[dict(name='Router',lr=1e-4),dict(name='TCG',lr=1e-5)],
          betas=[.9,.999],eps=1e-8,weight_decay=1e-4,clip_combined_norm=1.,reduction='SUM4/4 then one global clip and one update per synchronized replica',
          schedule=None,TCG_learning_rate_status='prespecified engineering default, not proven optimal or selected from test'),
      physical_GPUs=[4,5,6,7],world_size=4,
      stopping=dict(max_updates=3000,cap_not_mandatory=True,validation_interval=250,patience=4,min_delta=.001,anchor_initial=1.,
          score='mean_q(equal32video mean FloLPIPS(candidate,q)/equal32video mean FloLPIPS(B1500_matched_FP32,q))',
          mode='min',initialization_counts_for_patience=False,valid_complete_checks_only=True,
          raw_best='trained validation checkpoints >=250 only; strict lower score, ties earlier',
          anchor_update='S < anchor-0.001 else bad_check_count+1',
          final_selection='argmin of independently matched baseline score1 and best trained score; exact tie retains baseline',
          raw_and_meaningful_flags_separate=True,meaningful_improvement_threshold_met='best_trained_score < 1.0-0.001',
          final_choice_not_a_scientific_improvement_claim=True,selection_frozen_before_final_test=True),
      evaluation=dict(development_videos=32,development_QPs=[0,5,9],development_points=96,
          matched_baseline='independent frozen B+1500 instance, FP32 TCG parameters and internal operations; no Router, no optimizer or updates',
          historical_baseline='historical deployment FP16 TCG metrics preserved separately, never used for normalization',
          matched_baseline_streams='verified B streams may be independently decoded, input_stream_reused=True/fresh_encode=False; source/bytes/hash/window audited',
          joint_streams='fresh RANS encode and new independent decoder per point',final_videos=45,final_QPs=[0,5,9],final_frames=64,
          final_points_per_new_candidate=135,final_states=['GVC_RT_original','B1500_historical','B1500_matched_FP32','trained_joint'],
          datasets=dict(ULong_new_holdout=32,ULong_historical=6,HEVC_B=5,UVG_historical_HoneyBee_Jockey=2),
          metrics=['bpp','kbps','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError_GTFlow_PWC','WarpingError_ReconFlow_PWC'],
          inherited_PWC_sampler=previous['evaluation']['PWC_sampler'],PWC_metric_revision=False,pure_flicker_measure_claim=False,
          plotting='separate four split/domain means with equal video weights and45 per-video figures; actual3QP points only',BD_rate=False,
          historical_and_matched_precision_differences_saved=True),
      diagnostics=dict(previous['diagnostics'],statistics=previous['diagnostics']['statistics']+['projected_history_norm_before_after','current_projection_norm','residual_norm','actual_dtypes','keep_weight_eq1_ratio','Router_TCG_gradient_norms','relative_parameter_changes'],
          mask_semantics='continuous projected-history suppression, no motion/occlusion supervision guarantee',fixed_M_checks='0 and1 engineering checks only, no fixed-mask quality scan'),
      resource_limits=previous['resource_limits'],
      publication=dict(repo='ZhangYangyingzi/GVC-RT',branch='codex/tcg-stage2-history-gate-joint-20261106',allowlist_only=True,
          no_weights_videos_streams_cache_PDF_credentials=True,max_file_bytes=50*2**20,force_push=False))
    dump(ROOT/'protocol.json',p)
    dump(ROOT/'audits/protocol_frozen_before_new_quality.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),time=time.time(),new_quality_points=0,formal_joint_updates=0))
    (ROOT/'EXECUTION_PROTOCOL.md').write_text('# Internal history gate + Router/TCG joint execution\n\n'+
      'This protocol implements the single joint line prescribed on20261106. The exact settings are in protocol.json. The source manifest freezes executable code before formal updates.\n\n'+
      'B+1500 is cumulative TCG6500. Load original I/P, pilot receiver/buffers and exact B TCG; create a new seed20261106 Router at M0=.2. No historical Router or optimizer is loaded; trained fusion heads are retained.\n\n'+
      'Gate only previous_adapter(previous), before concat with current_adapter(current). Keep one spatial mask, shared across3layers; remove outer residual gating. M1 zeros direct history, not the whole residual. Inputs remain decoded codec features. All Router/TCG computations and parameters are FP32 with disabled autocast; output returns to incoming dtype. Parent half() preserves FP32 values. Other modules retain historical precision.\n\n'+
      'Only Router33889 and TCG137760 parameters update, total171649. Fresh AdamW groups1e-4/1e-5, fixed settings, combined clip1.0, SUM4/4. Preserve historical64frame causal detach and exact loss/VJP4096 ordering; no added groups/modules/losses/BPTT.\n\n'+
      'Use audited pool8470/quarantine236 and identical fixed splits/windows, source cursor1500, batch4=2UVG+2ULong. Revisited data are recorded, not claimed independent. Baseline normalization uses new matched-FP32 B inference, not historical FP16 scores.\n\n'+
      'Validate96points every250 updates, cap3000, patience4, min_delta.001. Trained raw best excludes initialization; score ties use earlier checkpoint. Final choice is raw dev-score argmin against independent matched baseline; threshold flag is separately recorded and selection does not assert effective improvement. Freeze selection before new final-test quality scoring. Always test and plot the best trained joint checkpoint.\n\n'+
      'Final45videos×QP0/5/9 use original RANS/native64frames and matching RGB FP32-first scores. Fresh joint encodes and independent decoders; matched baseline may decode audited old B streams. Preserve separate historical vs matched precision results. Four separate dataset plots and45 per-video plots use actual measured points, no BD-rate/extrapolation. Fixed prior diagnostic sequences/frames use0–1 mask colors.\n\n'+
      'All new writes are within the project. Preserve historical experiments and current dirty work. Atomic checkpoints include both Router andTCG, optimizer,4RNG states, cursor, receipts and validation state. No duplicate committed updates. Publish only allowlisted small task artifacts, no weights/media/cache/PDF/secrets, to the authorized new branch, with verified remote HEAD/core blobs.\n')
    print('PROTOCOL_FROZEN',sha(ROOT/'protocol.json'))
if __name__=='__main__':main()
