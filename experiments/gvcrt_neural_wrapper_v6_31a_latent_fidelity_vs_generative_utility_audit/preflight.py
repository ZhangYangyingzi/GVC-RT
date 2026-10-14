"""Freeze protocol and audit evidence. Never substitutes an unofficial target/loss."""
import ast
import platform
import subprocess
import sys
from audit_io import *


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def main():
    import torch
    (ROOT / 'logs').mkdir(exist_ok=True)
    for name in ('parts', 'figures/selected_counterexamples', 'sanity'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    # Copy exact historical manifests; never re-sample or re-select the source videos.
    manifest = {}
    origin = {}
    for dataset in ('ulong', 'uvg_holdout', 'uvg_validation', 'hevc_b', 'mcljcv'):
        p = V29 / 'manifests' / f'{dataset}.json'
        manifest[dataset] = load(p)['videos']
        origin[dataset] = dict(path=str(p), sha256=sha(p))
        for v in manifest[dataset]:
            assert v['frames'] == 64 and len(v['source_frame_indices']) == 64
            assert len(v['frame_rgb_sha256']) == 64
    v31 = load(V31 / 'eval_manifest.json')
    assert all(v31[d] == manifest[d] for d in v31)
    # Exact inherited finalize.py protocol: exclude the overlapping source 17.
    mcl29 = [v for v in manifest['mcljcv'] if v['name'] != 'videoSRC17_1920x1080_24']
    assert len(mcl29) == 29
    dump('dataset_manifest.json', dict(datasets=manifest, origins=origin,
         v631_manifest=dict(path=str(V31/'eval_manifest.json'), sha256=sha(V31/'eval_manifest.json'), identical_subsets=True),
         mcl29=dict(kind='secondary_subset_of_mcljcv', names=[v['name'] for v in mcl29],
                    excluded='videoSRC17_1920x1080_24', protocol_source=str(V29/'finalize.py'),
                    protocol_source_sha256=sha(V29/'finalize.py'), no_duplicate_evaluation=True)))
    # Locate C strictly in the existing evaluation manifest, then verify bytes and state.
    cfg = load(V29 / 'evaluation/config.json')
    c = cfg['checkpoints']['B_covstats_1000']
    assert sha(c['path']) == c['sha256']
    records = {}
    for method, path in [('Original_I', REPO/'checkpoints/GVC-RT_I.pt'),
                         ('Original_P', REPO/'checkpoints/GVC-RT_P.pt'),
                         ('B1', B1), ('B_covstats', Path(c['path']))]:
        state = torch.load(path, map_location='cpu', weights_only=True)
        rec = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                   top_level_keys=list(state), selection='fixed_before_quality')
        if method in ('B1', 'B_covstats'):
            if method == 'B1':
                assert state['adaptation_step'] == 1000
                rec['step'] = state['adaptation_step']
            else:
                rec['step'] = 1000
                rec['step_source'] = 'existing evaluation/config.json checkpoints.B_covstats_1000 (export contains only module states)'
            rec['module_hashes_fp32'] = {k:tensor_hash(state[k]) for k in ('wrapper','bridge','generator')}
            rec['module_hashes_deployment'] = {k:tensor_hash({n:t.half() for n,t in state[k].items()})
                                                 for k in ('bridge','generator')}
            key = 'B1_initial' if method == 'B1' else 'B_covstats_1000'
            assert rec['module_hashes_fp32'] == cfg['checkpoints'][key]['module_hashes']
            assert rec['module_hashes_deployment'] == cfg['deployment_receiver_hashes'][key]
            rec['compression_hash_recorded_in_checkpoint'] = state.get('compression_hash')
        else:
            state = state.get('student_ema',state.get('student',state.get('state_dict',state)))
            rec['state_dict_hash'] = tensor_hash(state)
            if method == 'Original_P':
                rec['module_hashes_deployment'] = {k:tensor_hash({n[len(prefix):]:t.half() for n,t in state.items() if n.startswith(prefix)})
                     for k,prefix in [('bridge','recon_generation_net.mlp.'),('generator','recon_generation_net.decoder.')]}
                assert rec['module_hashes_deployment'] == cfg['original_receiver_hashes']
        records[method] = rec
        del state
    records['secondary_checkpoint_locator'] = dict(path=str(V29/'evaluation/config.json'), sha256=sha(V29/'evaluation/config.json'), key='checkpoints.B_covstats_1000')
    records['B1_equivalence_to_historical_export'] = dict(
        user_requested_path=str(B1), historical_export=cfg['checkpoints']['B1_initial'],
        tensor_weights_identical=True, file_hash_identity=records['B1']['sha256']==cfg['checkpoints']['B1_initial']['sha256'])
    dump('checkpoint_inventory.json', records)
    teacher = load(V13/'audits/teacher_gate.json')
    candidates = {}
    for name in ('teacher_checkpoint_source.json', 'pretrain_teacher_checkpoint_source.json'):
        rec = load(V13/'audits'/name)
        assert sha(rec['checkpoint']['path']) == rec['checkpoint']['sha256']
        candidates[name] = rec
    dump('tokenizer_inventory.json', dict(candidates=candidates, historical_gate=teacher,
         exact_original_training_checkpoint_binding_verified=False,
         candidate_compatibility_is_not_exact_training_provenance=True))
    source_paths = list((REPO/'src').rglob('*.py')) + [REPO/'README.md']
    for folder in (V13,V29,V31, EXPS/'gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control'):
        source_paths += list(folder.glob('*.py'))
    source_paths += [V13/'audits/teacher_gate.json', V29/'evaluation/config.json',
        V13/'teacher_sources/src/Open_MAGVIT2/modules/vqvae/lookup_free_quantize.py',
        V13/'teacher_sources/src/Open_MAGVIT2/models/lfqgan.py',
        V13/'teacher_sources/src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py']
    frozen = {str(p):sha(p) for p in sorted(set(source_paths))}
    frozen.update({rec['path']:rec['sha256'] for rec in records.values() if isinstance(rec,dict) and 'bytes' in rec})
    frozen.update({rec['checkpoint']['path']:rec['checkpoint']['sha256'] for rec in candidates.values()})
    dump('source_hashes.json', frozen)
    hits = []
    for p in (REPO/'src').rglob('*.py'):
        for i,line in enumerate(p.read_text().splitlines(),1):
            if any(term in line.lower() for term in ('cosine_similarity','cosine_embedding','margin_loss','alignment_loss','lfq')):
                hits.append(dict(path=str(p),line=i,text=line))
    dump('official_objective_search.json', dict(scope='all released src/**/*.py',hits=hits,
         training_release_todo='README.md: Training code release is unchecked',
         prior_audit=str(EXPS/'gvcrt_neural_wrapper_v6_21b_gvcrt_latent_interface_audit/audits/gvcrt_objective_source.json'),
         exact_alignment_loss=None, exact_target_position=None, exact_teacher_preprocessing=None))
    dump('protocol.json', dict(schema='v6.31a-preregistered', no_training=True,no_checkpoint_updates=True,
         hypothesis='At matched actual bitrate, better tokenizer latent fidelity is not necessarily better generative reconstruction.',
         source_commit=git('rev-parse','HEAD'), branch=git('branch','--show-current'),
         methods=['Original','B1','B_covstats'], qps=list(range(10)), frames=64, IP=-1,
         populations={d:len(v) for d,v in manifest.items()},unique_expected_points=sum(map(len,manifest.values()))*30,
         sampling='exact dataset_manifest.json source_frame_indices; no 20-fps resampling',
         primary_frames='P (frames 1..63)', secondary_frames=['I0','all'],
         coding=dict(real_RANS=True,persist_bitstream=True,fresh_independent_decode=True,
                     causal_DPB_required=True,qp_phase='INDEX_MAP[i%8] with official shift_qp',
                     iframe_bypass_wrapper=True,rate='8*actual_file_bytes/(64*H*W)',
                     kbps_20fps='actual_bpp*H*W*20/1000', headers_included=True),
         target=dict(source='original GT x only', exact_official_training_preprocessing_required=True,
                     versions=['continuous_pre_LFQ','quantized_codeword','normalized_only_if_officially_defined'],
                     primary_version='UNRESOLVED: requires original training target source',
                     dimensions_do_not_prove_same_space=True, latent_interpolation=False),
         fidelity=dict(primary='UNRESOLVED: exact original alignment objective required',
                     diagnostics=['L1','RMSE','normalized_L2','cosine_distance','per_channel_normalized_RMSE',
                                  'per_channel_Pearson','mean_difference','covariance_difference'],
                     discrete_metrics='only after verifying discrete semantics; no hard-signing actual continuous inputs as if transmitted tokens',
                     normalized_L2='||z-t||_2 / ||t||_2; zero denominator reported undefined',
                     channel_NRMSE='mean_c(RMSE_c/std(target_c)); zero-variance channels separately reported',
                     distances_lower_is_better=True, correlation_higher_is_better=True),
         utility=dict(primary=['LPIPS','DISTS','FloLPIPS'], secondary=['PSNR','MS_SSIM','FID'],
                      FID='dataset/method/QP pooled distribution; no framewise pairing',
                      metric_reference=str(V29/'evaluation/config.json')),
         matching=dict(unit='dataset/video/method vs Original; all 10x10 actual-rate QP pairs',
                       relative_gap='abs(Rm-Ro)/((Rm+Ro)/2)', primary_max_gap=.01,secondary_max_gap=.02,
                       keep_all=True,primary_selection=['smallest_rate_gap','lowest_mean_rate','lowest_original_QP','lowest_method_QP'],
                       quality_blind=True, primary_excludes_2pct=True,rate_interpolation_core_evidence=False),
         inversion=dict(delta_latent='method-original; >noise_floor = worse',delta_quality='method-original; <0 = better',
                        latent_metrics=['official','L1','normalized_L2','cosine_distance'],quality_metrics=['LPIPS','DISTS','FloLPIPS'],
                        same_latent='abs(delta)<=deterministic_repeat_noise_floor',
                        tolerance='max observed absolute metric difference over >=10 points independently decoded twice; no posthoc enlarged tolerance',
                        summaries=['dataset','external_QP_pair','pooled_actual_rate_tertiles'],
                        primary_population='one primary <=1% pair per video/method; all eligible pairs secondary'),
         correlation=dict(analyses=['within-video Spearman','<=2% matched neighborhood delta correlation',
                                   'latent and quality residual correlations after separate linear fits against log(actual_bpp) with video/method fixed effects'],
                          statistics=['Pearson','Spearman','n','bootstrap_95pct_CI'],
                          bootstrap='2000 video-cluster resamples; seed=20261014; preserve all points from sampled video',
                          strata='dataset, method, target_version, frame_scope'),
         figures=dict(A='latent vs quality, marker=method, size=actual_bpp',B='matched delta latent vs delta quality with quadrants',
                      C='inversion fraction by dataset',D='up to 4 primary inversion pairs sorted by rate gap then descending inversion magnitude; frame32 fixed',
                      quality_metrics=['LPIPS','DISTS','FloLPIPS']),
         wrapper_diagnostics=['RGB residual','frequency energy','existing entropy estimates if available','actual generator-input fidelity'],
         sensitivity_probe='optional supporting evidence only, after primary completion',
         decision='Strong / Partial / Not supported based on reliable repeated matched-rate evidence; missing data is NOT CASE 3',
         gpu_assignment={'4':['ulong'],'5':['uvg_holdout','uvg_validation'],'6':['hevc_b'],'7':['mcljcv']},
         writes='per-worker parts only, single aggregation process',
         launch_gate=['exact official target/loss/preprocessing provenance','same-space semantics confirmed','full sanity PASS'],
         publish_gate='complete evaluation, report and integrity PASS before add/commit/push; only this experiment',
         commit_message='Add v6.31a latent fidelity vs generative utility audit'))
    dump('environment.json', dict(python=sys.version, executable=sys.executable,platform=platform.platform(),
         torch=torch.__version__,cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
         visible_device_count=torch.cuda.device_count(), packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()))
    dump('status.json', dict(status='BLOCKED_PRE_SANITY',full_experiment_launched=False,
         completed_evaluation_points=0, gpu_workers=[], background_pid=None,
         blockers=['Official GVC-RT alignment loss and exact training target/preprocessing not released in available source',
                   'Historical teacher gate explicitly does not establish exact original training checkpoint identity'],
         gpu_access='host access verified; sandbox has no device nodes', no_training=True))
    dump('integrity.json', dict(status='PREPARED_NOT_COMPLETE', no_training=True,
         checkpoint_hash_verification='PASS', source_snapshot='source_hashes.json',
         exact_target_space='BLOCKED', full_sanity='NOT_RUN', publication='NOT_ATTEMPTED'))
    print('PREFLIGHT_RECORDED: official alignment provenance blocks full sanity/launch',flush=True)


if __name__ == '__main__':
    main()
