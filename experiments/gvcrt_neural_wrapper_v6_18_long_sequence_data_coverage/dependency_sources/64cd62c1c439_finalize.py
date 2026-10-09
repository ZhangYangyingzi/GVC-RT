import argparse,ast,math
from parallel_utils import *
from report import collect

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=('A','B'),required=True);args=p.parse_args()
    root=exp(args.experiment);config=load(root/'config.json');checks={}
    try:
        rows=collect(root,config);verify_frozen(config)
        checks.update(evaluation_only=True,no_training=True,checkpoints_unchanged=True,real_RANS_complete=True,independent_decode_pass=True,
            all_values_finite=True,rate_sanity_pass=True,GPU_assignment_pass=all(r['physical_gpu'] in config['gpus'] for r in rows))
        for path in ROOT.glob('*.py'):
            tree=ast.parse(path.read_text())
            calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            assert not set(calls)&{'backward','train','optimizer_step'}
        assert load(root/'report_integrity.json')['status']=='PASS'
        summary=read(root/('virat_480p20_all_qp_summary.csv' if args.experiment=='A' else 'module_attribution_summary.csv'))
        for m in METRICS:
            checks[m+'_complete']=all(math.isfinite(float(r[m])) for r in summary)
        if args.experiment=='A':
            for v in config['videos']:
                assert sha(v['path'])==v['source_sha256']
                h=hashlib.sha256();count=0
                for a in raw_mp4(v):h.update(a.tobytes());count+=1
                assert h.hexdigest()==v['rgb_sha256'] and count==v['frames']
            checks.update(source_resolution=[854,480],source_count=8,random_count=5,specified_count=3,
                source_paths_are_existing_visualization_originals=True,no_source_regeneration=True,no_temporal_resampling=True,no_frame_drop=True,
                rate_accounting_fps=20.0,qps=config['qps'],original_complete=sum(r['method']=='original' for r in rows)==80,
                clip8_complete=sum(r['method']=='clip8' for r in rows)==80)
            assert len(summary)==20 and all(int(r['FID_num_samples'])==579 for r in summary)
            gates=read(root/'strict_per_qp_gate.csv');assert len(gates)==10
            for row in gates:
                q=int(row['external_qp']);a=next(s for s in summary if s['method']=='original' and int(s['external_qp'])==q);c=next(s for s in summary if s['method']=='clip8' and int(s['external_qp'])==q)
                values=[float(c['kbps'])<float(a['kbps'])]+[float(c[m])<=float(a[m]) for m in METRICS]
                assert (row['strict_all_pass']=='True')==all(values)
            assert all((root/'rd_curves'/f'Rate_{m}.{e}').is_file() for m in METRICS for e in ('png','pdf'))
            assert len(read(root/'equal_rate_summary.csv'))==len(read(root/'bd_rate.csv'))==4
            for name,key in [('equal_rate_summary.csv','mean_equal_rate_delta'),('bd_rate.csv','BD_rate_percent')]:
                for r in read(root/name):
                    assert r['status'] in ('valid','invalid')
                    if r['status']=='invalid':assert r['reason']
                    else:assert math.isfinite(float(r[key]))
            visual=load(root/'visualization_manifest.json');assert visual['status']=='PASS' and len(visual['videos'])==32
            assert all(sha(v['path'])==v['sha256'] for v in visual['videos'])
            checks['visualizations_complete']=True
        else:
            audit=load(root/'source_integrity.json');source=load(V52/'canonical_source_audit.json')
            assert sha(V52/'canonical_source_audit.json')==audit['canonical_audit_sha256']
            loader=module('final_source_loader',V52/'canonical_loader.py')
            for s in source['records']:
                directory=Path(s['canonical_dir'])
                assert [sha(directory/f'frame_{i:06d}.png') for i in range(64)]==s['canonical_frame_sha256']
                assert loader.rgb_hash(directory)==s['whole_sequence_rgb_hash']
            checks.update(canonical_sources_reused=True,source_hashes_match_v5a2=True,original_weights_unchanged=True,
                v41_weights_unchanged=True,clip8_weights_unchanged=True,identity_P_exact=all(r['identity_P_exact'] for r in rows if r['P']=='identity'),
                all_module_combinations_complete=len(rows)==420,qps=config['qps'])
            assert len(summary)==56 and all(int(r['FID_num_samples'])==(512 if r['dataset']=='ulong' else 448) for r in summary)
            stats=read(root/'proxy_domain_statistics.csv');assert len(stats)==34
            for r in stats:
                for k in ('proxy_source_L1','proxy_source_MSE','sobel_difference','laplacian_variance_difference','temporal_delta','residual_std','residual_p95_abs','residual_sobel_abs'):
                    assert math.isfinite(float(r[k]))
            checks['proxy_domain_statistics_complete']=True
            checks['full_methods_v5a2_regression_pass']=all(r.get('v5a2_regression_pass') for r in rows if r['method'] in ('ORIGINAL','V41_FULL','CLIP8_FULL'))
        failed=[k for k,v in checks.items() if isinstance(v,bool) and not v]
        dump(root/'final_integrity.json',dict(checks,status='FAIL' if failed else 'PASS',failed_checks=failed,num_points=len(rows),
             script_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},performance_gates_are_not_integrity_gates=True))
        assert not failed,failed
        print('FINAL INTEGRITY PASS',args.experiment,flush=True)
    except Exception as exc:
        dump(root/'final_integrity.json',dict(status='FAIL',error=repr(exc),completed_checks=checks));raise
if __name__=='__main__':main()
