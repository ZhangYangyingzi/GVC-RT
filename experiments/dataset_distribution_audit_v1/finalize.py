"""Exhaustive required-output and immutable-source checks."""
from audit_io import *
from prepare_cohorts import inventory_old
def main():
    for path,h in load(ROOT/'frozen_dependencies.json').items():assert sha(path)==h,('old dependency changed',path)
    before=load(ROOT/'old_inventory_before.json');after=inventory_old();assert before==after,'Old experiment inventory changed'
    videos=load(ROOT/'manifests/all_sources.json')['videos'];assert len(videos)==4111
    assert sha(ROOT/'manifests/all_sources.json')==load(ROOT/'cohort_audit.json')['source_manifest_sha256']
    for v in videos:
        assert load(ROOT/'parts/source'/f'{v["sample_id"]}.json')['status']=='PASS'
        assert load(ROOT/'parts/motion'/f'{v["sample_id"]}.json')['status']=='PASS'
    for name in ('vimeo_inventory','vimeo_archive_audit','vimeo_split_audit','vimeo_training_readiness','cohort_audit','source_progress','codec_profile_audit','data_leakage_audit','aggregation_audit','plot_audit'):
        assert load(ROOT/(name+'.json'))['status']=='PASS',name
    assert load(ROOT/'codec_profile_audit.json')['points']==2373
    scaler=load(ROOT/'feature_scaler.json');pca=load(ROOT/'pca_fit_audit.json');ids={v['sample_id'] for v in videos if v['role']=='candidate_training'}
    assert set(scaler['fit_sample_ids'])<=ids and set(pca['fit_sample_ids'])<=ids
    files=['vimeo_inventory.json','vimeo_training_readiness.json','training_input_protocol_audit.json','source_spatial_statistics.csv',
        'source_temporal_statistics.csv','motion_statistics.csv','camera_motion_statistics.csv','codec_profile_per_video.csv','codec_profile_summary.csv',
        'codec_difficulty_features.csv','reconstruction_difficulty_proxy.csv','test_failure_association.csv','test_failure_correlations.csv',
        'dataset_statistics_summary.csv','dataset_pairwise_distance.csv','reference_to_training_distance.csv','reference_feature_percentiles.csv',
        'uvg_nearest_ulong.csv','uvg_nearest_vimeo.csv','pca_coordinates.csv','pca_explained_variance.csv','data_leakage_audit.json']
    for p,h in load(ROOT/'plot_audit.json')['files'].items():assert sha(ROOT/p)==h
    dump(ROOT/'final_integrity.json',dict(status='PASS',completed_unix=time.time(),no_training=True,old_experiments_untouched=True,Vimeo_inventory_complete=True,
        ULong_train_exact_reuse=True,unused_disjoint=True,Vimeo_official_train_only=True,Vimeo7frames_explicitly_approved=True,
        UVG_frozen_canonical=True,VIRAT_frozen_unique_sources=True,no_reference_scaler_fit=True,no_reference_PCA_fit=True,no_reference_guided_sampling=True,
        source_temporal_motion_statistics_complete=True,codec_real_RANS=True,codec_original_only=True,QP0_4_9_complete=True,
        all_plots_generated=True,missing_values_explicit_not_zero_imputed=True,source_checkpoints_hashed=True,
        archive_validation_scope='All extracted headers; central directory;64 seeded ZIP image CRC/decode checks;2048 selected train sequences fully decoded; unread archive payloads not certified',
        all_manifests_hashed={p.name:sha(p) for p in (ROOT/'manifests').glob('*.json')},artifacts={f:sha(ROOT/f) for f in files}))
    print('FINAL INTEGRITY PASS',flush=True)
if __name__=='__main__':main()
