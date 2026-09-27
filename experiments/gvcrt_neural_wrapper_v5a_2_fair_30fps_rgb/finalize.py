"""Fail-closed completion audit for all 450 freshly encoded evaluation points."""
import ast
import math
from audit_utils import *
from canonical_loader import rgb_hash
from prepare_protocol import source_inventory
from report import collect

ACTIVE_SCRIPTS=('audit_utils.py','canonical_loader.py','prepare_protocol.py','evaluate.py','source_statistics.py',
                'report.py','plot_results.py','rd_analysis.py','stream_audit.py','run_pipeline.py','finalize.py','print_status.py')

def main():
    config=load('config.json');checks={}
    checks['evaluation_only']=config['evaluation_only'] and config['training_allowed'] is False
    prohibited=[]
    for name in ACTIVE_SCRIPTS:
        tree=ast.parse((ROOT/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                function=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                if function in ('train','backward','optimizer_step'):prohibited.append([name,function])
    checks['no_training']=not prohibited and not list(ROOT.glob('checkpoints/*'))
    checks['checkpoints_unchanged']=all(not c['path'] or sha(c['path'])==c['sha256'] for c in config['checkpoints'].values())
    checks['source_and_model_implementations_unchanged']=all(sha(p)==h for p,h in {**config['source_hashes'],**config['additional_source_hashes']}.items())
    checks['metric_implementations_unchanged']=all(sha(p)==h for p,h in load('metric_implementation_audit.json')['source_and_weight_sha256'].items())
    before=load('source_inventory_before.json');after=source_inventory()
    changed=[p for p,stat in before.items() if after.get(p)!=stat];added=[p for p in after if p not in before]
    dump('source_experiments_unchanged_audit.json',dict(status='PASS' if not changed and not added else 'FAIL',
         existing_file_count=len(before),changed_or_missing=changed,added=added,
         method='file size and nanosecond modification time inventory, plus SHA256 of selected frozen inputs/reports/checkpoints'))
    checks['source_experiments_unchanged']=not changed and not added
    audit=load('canonical_source_audit.json');sources=audit['records']
    checks['canonical_source_audit_pass']=audit['status']=='PASS'
    checks['canonical_resolution_pass']=all((r['canonical_width'],r['canonical_height'])==(1920,1080) for r in sources)
    checks['canonical_fps_pass']=all(r['canonical_fps']==30.0 for r in sources)
    checks['canonical_num_frames_pass']=all(r['canonical_num_frames']==64 for r in sources)
    checks['ulong_video_count_pass']=len([r for r in sources if r['dataset']=='ulong'])==8
    checks['uvg_sequence_count_pass']=len([r for r in sources if r['dataset']=='uvg'])==7
    checks['uvg_stride4_pass']=all(r['selected_original_frame_indices']==list(range(0,253,4)) for r in sources if r['dataset']=='uvg')
    checks['ulong_contiguous_frames_pass']=all(r['selected_original_frame_indices']==list(range(64)) for r in sources if r['dataset']=='ulong')
    for source in sources:
        directory=Path(source['canonical_dir'])
        assert sha(source['original_path'])==source['original_sha256']
        assert [sha(directory/f'frame_{i:06d}.png') for i in range(64)]==source['canonical_frame_sha256']
        assert rgb_hash(directory)==source['whole_sequence_rgb_hash']
    checks['canonical_images_unchanged']=True
    loader=load('canonical_loader_audit.json')
    checks['same_lossless_rgb_loader']=loader['status']=='PASS' and loader['same_loader_for_ulong_and_uvg'] and sha(loader['loader_path'])==loader['loader_sha256']
    checks['same_model_input_range']=all(r['range']==[0,1] and r['dtype']=='float32' and r['RGB_order'] and r['roundtrip_bitexact'] for r in loader['records'])
    records=[];summaries=[];counts={};rates=[];invalid=[]
    for dataset in ('ulong','uvg'):
        own=collect(dataset);records.extend(own);counts[dataset]=len(own)
        summary=read(f'{dataset}_all_qp_summary.csv');assert len(summary)==30
        expected=512 if dataset=='ulong' else 448
        for method in METHODS:
            assert sorted(int(r['external_qp']) for r in summary if r['method']==method)==list(range(10))
        assert all(int(r['FID_num_samples'])==int(r['num_frames'])==expected for r in summary)
        summaries+=summary
        rate=read(f'parts/{dataset}_rate_sanity.csv');assert len(rate)==30;rates+=rate
        for kind in ('equal_rate_summary','bd_rate'):
            table=read(f'{dataset}_{kind}.csv');assert len(table)==12
            for row in table:
                assert row['status'] in ('valid','invalid')
                if row['status']=='invalid':assert row['reason'];invalid.append(dict(row,kind=kind))
                else:
                    value=row['mean_equal_rate_delta'] if kind=='equal_rate_summary' else row['BD_rate_percent']
                    assert math.isfinite(float(value))
        assert load(f'parts/{dataset}_report_done.json')['summary_sha256']==sha(ROOT/f'{dataset}_all_qp_summary.csv')
    write('rate_sanity_audit.csv',rates)
    checks['all_450_points_complete']=counts=={'ulong':240,'uvg':210}
    checks['fresh_real_RANS']=all(r['fresh_real_RANS'] and r['protocol']=='V5-A.2-30fps-RGB' and Path(r['bitstream_path']).resolve().is_relative_to(ROOT/'bitstreams') for r in records)
    checks['independent_decode_pass']=all(r['independent_decode_pass'] and r['metric_decode_pass'] and r['state_sync_pass'] for r in records)
    checks['rate_sanity_pass']=all(truth(r['pass']) and math.isclose(float(r['kbps']),float(r['bits_per_frame'])*30/1000,rel_tol=1e-12,abs_tol=1e-10) for r in rates)
    checks['GPU_0_3_unused']=all(r['physical_gpu'] in (4,5,6,7) for r in records)
    checks['all_values_finite']=all(math.isfinite(float(r[m])) for r in summaries for m in (*METRICS,'mean_real_kbps','mean_bits_per_frame','mean_bpp','PSNR','SSIM','MS_SSIM'))
    for metric in METRICS:checks[metric+'_complete']=all(math.isfinite(float(r[metric])) for r in summaries)
    for dataset in ('ulong','uvg'):
        for i in range(len(videos(dataset))):
            per_method=[load(f'parts/source_{dataset}_{m}_{i}.json') for m in METHODS]
            assert all(r['processing_canvas']==[1920,1088] and r['metric_crop']==[1920,1080] for r in per_method)
            assert len({r['source_hash'] for r in per_method})==len({r['tensor_hash'] for r in per_method})==1
    checks['same_padding']=checks['same_metric_crop']=checks['source_frames_identical_across_methods']=True
    checks['RD_curves_complete']=all((ROOT/'rd_curves'/d/f'Rate_{m}.{ext}').is_file() and (ROOT/'rd_curves'/d/f'Rate_{m}.{ext}').stat().st_size>100 for d in ('ulong','uvg') for m in METRICS for ext in ('png','pdf'))
    checks['source_statistics_complete']=load('source_statistics_audit.json')['status']=='PASS' and len(read('source_domain_statistics_per_video.csv'))==15 and len(read('source_domain_statistics_dataset_summary.csv'))==2
    for row in read('source_domain_statistics_per_video.csv'):
        assert all(math.isfinite(float(v)) for k,v in row.items() if k not in ('dataset','name'))
    checks['uvg_per_sequence_complete']=len(read('uvg_per_sequence_summary.csv'))==210
    checks['FID_pooling_pass']=all(int(r['FID_num_samples']) in (512,448) for r in summaries)
    failed=[k for k,v in checks.items() if not v]
    dump('final_integrity.json',dict(checks,status='FAIL' if failed else 'PASS',failed_checks=failed,
        canonical_resolution=[1920,1080],canonical_fps=30.0,canonical_num_frames=64,ulong_video_count=8,uvg_sequence_count=7,
        uvg_stride=4,uvg_selected_indices=list(range(0,253,4)),external_qps=list(range(10)),point_counts=counts,
        FID_samples_per_method_QP={'ulong':512,'uvg':448},invalid_RD_results=invalid,
        active_script_sha256={n:sha(ROOT/n) for n in ACTIVE_SCRIPTS}))
    assert not failed,failed
    print('FINAL INTEGRITY PASS',counts,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump('final_integrity.json',dict(status='FAIL',error=repr(exc),evaluation_only=True))
        raise

