"""Apply the user's explicit paper-faithful proxy scope before observing results."""
from audit_io import *


def main():
    protocol=load(ROOT/'protocol.json')
    protocol.update(audit_scope='paper-faithful proxy audit',exact_original_teacher_verified=False,
        scope_authorization='User explicitly authorized paper_compatible_teacher_proxy after exact original training source was found unavailable.',
        claim_limit='All fidelity claims are relative to this fixed proxy; no exact original alignment-loss reproduction.')
    protocol['target'].update(primary_version='continuous_pre_LFQ',
        source='original GT x through fixed pretrained Open-MAGVIT2 2^18 / 18-channel EMA encoder',
        exact_official_training_preprocessing_required=False,
        preprocessing='FP32 RGB [0,1] -> replicate bottom 8 pixels to 1088x1920 -> 2*x-1; no resize; shared by all methods',
        normalized_version='derived diagnostic F.normalize(dim=1); NOT asserted to be original training target')
    protocol['fidelity'].update(primary='proxy channelwise cosine distance: mean(1-cos(z_gen_actual,l_t),dim=channel,eps=1e-8)',
        paper_distribution_loss='0.2 L_cos + 0.02 L_margin (user-provided paper description)',
        unreproduced='L_margin exact formula unavailable; neither guessed nor labeled reproduced; image-domain L1+LPIPS is supervision, not a latent metric',
        official_primary_metric=False, continuous_and_quantized_reported_separately=True)
    protocol['inversion']['latent_metrics']=['cosine','L1','normalized_L2']
    protocol['launch_gate']=['proxy teacher semantics/config/weights verified','full 4-frame sanity PASS']
    protocol['inversion']['noise_floor_selection']='10 points selected by SHA256(seed=20261014,dataset,video,method,QP) before quality; two fresh repeats per point'
    protocol['full_tensor_samples']='fixed random seed 20261014; 4 points; frames 1,32; no quality-based selection'
    protocol['decision_rule']=dict(primary_method='B1',secondary_method='B_covstats',
        reliable_counterexample='P proxy cosine worse beyond repeat tolerance OR same within tolerance, and >=2 of LPIPS/DISTS/FloLPIPS improved',
        Strong='>=4 reliable B1 primary pairs across >=2 datasets, >=4 videos and >=2 distinct QP pairs',
        Partial='at least one reliable B1 primary counterexample below Strong coverage',
        Not_supported='primary B1 matched pairs exist but no reliable counterexample',
        Not_assessable='no primary B1 matched pair',
        quantized_and_other_latent_metrics='separate diagnostic inversion counts; do not replace the primary cosine criterion')
    protocol['figures']['D']='up to4 strict P cosine/LPIPS primary inversions, ascending gap then descending delta_cosine*(-delta_LPIPS); lexical final ties; fixed frame32'
    dump('protocol.json',protocol)
    idx=load(V29/'branches/B_covstats/checkpoint_index.json')['1000']
    inv=load(ROOT/'checkpoint_inventory.json')
    assert idx['inference']['sha256']==inv['B_covstats']['sha256']
    assert idx['module_hashes']==inv['B_covstats']['module_hashes_fp32']
    inv['B_covstats']['step_source']=dict(path=str(V29/'branches/B_covstats/checkpoint_index.json'),key='1000',record=idx)
    dump('checkpoint_inventory.json',inv)
    tasks=[]
    for d,vs in load(ROOT/'dataset_manifest.json')['datasets'].items():
        gpu=next(int(g) for g,ds in protocol['gpu_assignment'].items() if d in ds)
        for v in vs:
            for m in protocol['methods']:
                for q in range(10):tasks.append(dict(dataset=d,video=v['video_index'],method=m,qp=q,gpu=gpu))
    def digest(t):return hashlib.sha256(('20261014'+json.dumps(t,sort_keys=True)).encode()).hexdigest()
    selected=sorted(tasks,key=digest)
    dump('task_manifest.json',dict(tasks=tasks,noise_points=selected[:10],tensor_sample_points=selected[10:14]))
    evidence=dict(status='PASS_PAPER_COMPATIBLE_PROXY_ONLY', exact_original_teacher_checkpoint=False,
        teacher_checkpoint=load(V13/'audits/pretrain_teacher_checkpoint_source.json')['checkpoint'],EMA=True,
        target_cont='official Open-MAGVIT2 Encoder.forward output immediately before LFQ.forward',
        target_quant='torch.where(l_t>0,+1,-1), matching official LFQ.forward codebook_value=1.0',
        target_normalized='derived unit-channel-norm version, diagnostic only',
        no_projection=True,no_extra_normalization_in_official_LFQ=True,
        codebook_size=262144,channels=18,expected_shape=[1,18,68,120],
        generator_input='P DMC.decompress -> entropy decoded y_hat -> dec(y_hat,causal_ctx,q_decoder) -> pixel_unshuffle(feature,2) -> GroupNorm/DepthConv/GroupNorm/SiLU/DepthConv -> codeword -> decoder',
        actual_generator_input_is_continuous=True,LFQ_quantization_not_in_receiver=True,
        quant_step='separate decoder argument q_scale_recon[actual_qp]; not a scalar normalization of codeword',
        dtype=dict(generator_input='float16',teacher='float32',metrics='float64'),
        coordinate_evidence=str(V13/'audits/teacher_gate.json'),
        coordinate_limit='historical unchanged-axis compatibility is empirical evidence; it does not identify original teacher weights or original training target',
        training_target_in_released_codec='not present; user-authorized continuous pre-quantization paper proxy',
        forbidden_spaces=['128-channel compression y_hat','128-channel hyper z_hat','256-channel bridge input feature','18-channel quantized LFQ codeword confused with continuous output'],
        primary_metric='explicit proxy cosine distance, not official primary metric',
        inference_source_locations=['src/models/video_model_gvcrt.py:69','src/models/video_model_gvcrt.py:300','src/models/video_model_gvcrt.py:395','src/models/improved_model_gvcrt.py:214'])
    dump('tensor_space_audit.json',evidence)
    (ROOT/'tensor_space_audit.md').write_text('''# Paper-faithful proxy tensor-space audit

This uses a **paper_compatible_teacher_proxy**, NOT a verified GVC-RT original training teacher checkpoint. The user authorized this scope because official training code and exact teacher weights are not public.

The frozen official Open-MAGVIT2 262144-codebook, 18-channel EMA encoder returns continuous `l_t` immediately before `LFQ.forward`. In the archived official implementation, `VQModel.encode` calls `encoder(x)` then `quantize(h)`. LFQ applies `where(h > 0, +1, -1)` without extra normalization or a learned projection. Continuous and hard-sign versions are recorded separately. A normalized version is explicitly derived and is not claimed to be a training target.

Released P-frame inference is fully traceable: `DMC.decompress` entropy-decodes hyperlatent and compression latent, reconstructs causal context from the DPB, calls `dec(y_hat, ctx, q_decoder)`, then `PretrainedReconWrapper.forward`. Its feature is pixel-unshuffled by 2, processed by GroupNorm / DepthConv / GroupNorm / SiLU / DepthConv, and the resulting continuous 18-channel `codeword` goes directly into `recon_generation_net.decoder(codeword, q_recon)`. There is no LFQ hard sign at this interface. `q_recon` is a separate decoder conditioning argument and is saved separately. Compression latent, bridge input, codeword and teacher hard-sign code are never treated as identical tensors.

Both teacher and actual generator input have shape `[1,18,68,120]` for the inherited 1088x1920 padded canvas. Teacher processing uses original source RGB in FP32, replicate padding at the bottom, then `2*x-1`, without P(x), resizing, axis transforms or learned projection. Released coding uses the inherited half -> replicate pad -> scale sequence. Shape agreement alone does not prove original training provenance. Historical fixed non-test compatibility evidence supports unchanged axes for this proxy; all conclusions remain conditional on the proxy.

The paper distribution supervision is described as `0.2 L_cos + 0.02 L_margin` with image-domain `L1+LPIPS`. We do not know the exact margin definition or complete unpublished training implementation and do not reproduce or invent it. The primary audit distance is explicitly defined as channelwise cosine distance averaged over pixels (eps=1e-8), with L1, RMSE, target-norm L2, channel-normalized RMSE, Pearson, means and covariance as diagnostics. These are proxy fidelity metrics, never “official primary metric.” Distances to continuous and quantized targets are separately named. Actual receiver input is continuous, so no transmitted exact-token-match claim is made. No fidelity to teacher is interpreted causally.
''')
    s=load(ROOT/'status.json');s.update(status='READY_FOR_PROXY_SANITY',blockers=[],audit_scope='paper-faithful proxy audit')
    dump('status.json',s)
    integrity=load(ROOT/'integrity.json')
    integrity.pop('exact_target_space',None)
    integrity.update(status='PREPARED_NOT_COMPLETE',exact_original_teacher_verified=False,proxy_tensor_semantics='PASS',
                     full_sanity=load(ROOT/'sanity/proxy_sanity.json')['status'] if (ROOT/'sanity/proxy_sanity.json').exists() else 'NOT_RUN')
    dump('integrity.json',integrity)
    watched=load(ROOT/'source_hashes.json')
    for p in [V13/'teacher_sources/configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_262144.yaml', V29/'branches/B_covstats/checkpoint_index.json']:
        watched[str(p)]=sha(p)
    # Freeze imported historical engine, metrics and extension dependencies, including their model files.
    dep=V29/'audits/dependencies.json'
    if dep.exists():
        for p,h in load(dep).items():
            if Path(p).is_file(): watched[p]=sha(p)
    dump('source_hashes.json',watched)
    for p in [Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')]:watched[str(p)]=sha(p)
    dump('source_hashes.json',watched)
    dump('implementation_hashes.json',{p.name:sha(p) for p in ROOT.glob('*.py')})
    print('PROXY_PROTOCOL_READY',len(tasks),'points',flush=True)


if __name__=='__main__':main()
