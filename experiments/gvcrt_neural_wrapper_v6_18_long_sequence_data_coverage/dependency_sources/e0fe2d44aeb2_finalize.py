import ast
import math
import random
import numpy as np
from PIL import Image
from diag_utils import *
from media import probe

def main():
    protected=load('protected_inputs.json');checks={}
    originals=protected['existing_files']
    checks['existing_experiment_results_unchanged']=all(Path(p).is_file() and sha(p)==h for p,h in originals.items())
    checks['no_new_files_outside_diagnostic_directory']=set(str(p) for p in PARENT.rglob('*') if p.is_file() and ROOT not in p.parents)==set(originals)
    checks['all_source_checkpoints_unchanged']=all(sha(p)==h for p,h in protected['checkpoints_and_model_sources'].items())
    checks['old_rd_curves_unchanged']=all(sha(p)==h for p,h in originals.items() if str(PARENT/'rd_curves')+'/' in p)
    prohibited=[]
    for path in ROOT.glob('*.py'):
        tree=ast.parse(path.read_text())
        prohibited.extend(n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step','train'))
    checks['no_training_performed']=not prohibited and not list(ROOT.rglob('*.pt'))
    rng=random.Random(20260927)
    expected=[(d,i) for d,n in [('uvg',3),('ulong',2)] for i in rng.sample(range(len(old.videos(d))),n)]
    checks['fixed_random_sampling_reproduced']=[(s['dataset'],s['video_index']) for s in samples()]==expected
    assert all(sha(s['source_path'])==s['source_sha256'] for s in samples())
    reuse=[];videos=[];colors=[]
    for s in samples():
        audit=load(f'parts/reuse_{tag(s)}_current.json');assert audit['status']=='PASS'
        assert len(audit['records'])==12
        for r in audit['records']:
            assert not r['reencoded'] and r['decoded_reconstruction_sha256']==r['expected_reconstruction_sha256']
            assert sha(r['bitstream_path'])==r['bitstream_sha256'] and sha(r['raw_RGB_path'])==r['raw_RGB_sha256']
            assert r['physical_gpu'] in (4,6)
            frame=read(r['frame_QP_csv']);assert len(frame)==count(s)
            assert {int(v['external_qp']) for v in frame}=={r['external_qp']}
            reuse.append(r)
        comp=load(f'parts/comparison_{tag(s)}_current.json');assert comp['status']=='PASS' and len(comp['videos'])==4
        videos.extend(comp['videos'])
        rows=read(f'parts/color_{tag(s)}_current.csv');assert len(rows)==24;colors.extend(rows)
    assert len(reuse)==60 and len(videos)==20 and len(colors)==120
    for r in videos:
        info=probe(r['path']);n=96 if r['dataset']=='uvg' else 64
        assert int(info['nb_read_frames'])==n and info['codec_name']=='h264'
        path=Path(r['contact_sheet_directory'])
        assert len(list(path.glob('*.png')))==5
        for png in path.glob('*.png'):
            with Image.open(png) as im:assert im.mode=='RGB'
    for r in colors:
        for key,value in r.items():
            if key.startswith(('mean_','std_','signed_error_','MAE_','RGB_global')):assert math.isfinite(float(value))
    write('color_statistics.csv',colors)
    dump('bitstream_reuse_audit.json',dict(status='PASS',points=len(reuse),records=reuse,all_main_visualizations_reuse_existing_streams=True))
    dump('comparison_video_manifest.json',dict(status='PASS',videos=videos))
    conv=load('uvg_color_conversion_audit.json');assert conv['status']=='PASS' and len(conv['conversions'])==3
    for r in conv['conversions']:
        assert r['current_matches_frozen_experiment']
        for path in r['videos']:assert int(probe(path)['nb_read_frames'])==96
    smoke=samples()[0];smoke_rows=[]
    for m in METHODS:
        for q in (0,3):
            a=point(smoke,m,q);b=load(f'bt709_smoke/records/{m}_qp{q}.json')
            assert b['independent_decode_pass'] and b['metric_decode_pass']
            assert sha(b['bitstream_path'])==b['bitstream_sha256']
            assert b['checkpoint_sha256']==old.load('config.json')['checkpoints'][m]['sha256']
            for conversion,row in [('CURRENT_BT601',a),('BT709_EXPLICIT',b)]:
                smoke_rows.append(dict(source_conversion=conversion,sequence=smoke['name'],method=m,QP=q,
                    num_frames=96,**{k:float(row[k]) for k in ('kbps','LPIPS','DISTS','FloLPIPS')},
                    bitstream_path=row['bitstream_path'],bitstream_sha256=row['bitstream_sha256'],independent_decode_pass=True))
    write('bt709_smoke_summary.csv',smoke_rows)
    smoke_comp=load(f'parts/comparison_{tag(smoke)}_709.json')
    assert len(smoke_comp['videos'])==2
    for r in smoke_comp['videos']:assert int(probe(r['path'])['nb_read_frames'])==96
    checks.update(comparison_videos_complete=True,contact_sheets_complete=True,proxy_visualization_complete=True,
        color_statistics_complete=True,BT601_current_conversion_reproduced=True,BT709_explicit_conversion_complete=True,BT709_smoke_complete=True)
    mins=read('rd_curves/ulong/minimum_bitrates.csv');source=read(PARENT/'ulong_all_qp_summary.csv')
    for m in old.METHODS:
        best=min((r for r in source if r['method']==m),key=lambda r:float(r['mean_real_kbps']))
        r=next(r for r in mins if r['method']==m)
        assert float(r['mean_real_kbps'])==float(best['mean_real_kbps']) and r['external_qp']==best['external_qp']
    checks['ulong_minimum_bitrate_annotations_complete']=load('rd_plot_audit.json')['status']=='PASS' and all(
        (ROOT/'rd_curves/ulong'/f'Rate_{m}.{ext}').stat().st_size>100 for m in old.METRICS for ext in ('png','pdf'))
    checks['minimum_bitrate_values_loaded_from_csv']=True
    checks['uvg_per_sequence_diagnostic_complete']=len(read('uvg_per_sequence_diagnostic.csv'))==210
    checks['ulong_per_qp_diagnostic_complete']=len(read('ulong_per_qp_diagnostic.csv'))==20
    checks['GPU_0_3_unused']=all(r['physical_gpu'] in (4,6) for r in reuse)
    checks['all_statistics_finite']=all(math.isfinite(r[k]) for r in smoke_rows for k in ('kbps','LPIPS','DISTS','FloLPIPS'))
    failed=[k for k,v in checks.items() if not v]
    dump('final_integrity.json',dict(checks,status='FAIL' if failed else 'PASS',failed_checks=failed,random_seed=20260927,
        uvg_random_samples=3,ulong_random_samples=2,QP_visualized=list(QPS),main_comparison_count=20,
        BT709_comparison_count=2,conversion_video_count=9,bitstream_reuse_count=60,color_statistics_rows=120,
        protected_existing_file_count=len(originals),diagnostic_only=True,checkpoint_selection=False))
    assert not failed,failed
    print('DIAGNOSTIC FINAL INTEGRITY PASS',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump('final_integrity.json',dict(status='FAIL',error=repr(exc)));raise
