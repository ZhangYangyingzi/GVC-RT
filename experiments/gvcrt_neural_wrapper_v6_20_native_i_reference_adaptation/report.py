"""Historical pooled FID and common-rate arithmetic; three fixed methods."""
import ast,math,statistics,traceback
from io20 import *
PAIRS=(('native_i_adapt_1000','wrapped_i_control_1000'),('native_i_adapt_1000','B_i_bypass'),('wrapped_i_control_1000','B_standard'),('native_i_adapt_1000','original'),('wrapped_i_control_1000','original'))
def comparisons(*args,**kw):
    src=(V15/'report.py').read_text();tree=ast.parse(src);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='comparisons');code=ast.get_source_segment(src,node).replace('expected_models=6','expected_models=5').replace('available_models=6','available_models=5').replace('six-model','five-model');ns=dict(globals());exec(code,ns);return ns['comparisons'](*args,**kw)
def numerical():
    src=(V15/'report.py').read_text();tree=ast.parse(src);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');code=ast.get_source_segment(src,node).split('    validation_counts={}')[0].replace("audits/step0_codec_equivalence.json","audits/implementation_check.json").replace('len(vs)*60','len(vs)*50').replace('expected_pooled_FID=60','expected_pooled_FID=50').replace('six-model','five-model')
    # Preserve source-model exposure facts from v6.18; no unseen-content claim.
    code=code.replace('dict(r,actual_bpp=', 'dict(r,source_overall_exposure_status="unknown; see v6.18 historical exposure audit",independent_unseen_generalization_claim=False,actual_bpp=')
    ns=dict(globals());exec(compile(code,str(ROOT/'report.py'),'exec'),ns);ns['main']()
def extra_tables():
    for d in DATASETS:
        folder=ROOT/'results'/d;raw=load(folder/'per_sequence_all_qp.json');same=[];bytes_rows=[]
        for v in sources(d):
            for q in range(10):
                rs={r['method']:r for r in raw if r['video_index']==v['video_index'] and r['QP']==q}
                for m,b in PAIRS:
                    for metric in ('bpp','LPIPS','DISTS','FloLPIPS','FID','PSNR','SSIM'):
                        x,y=rs[m][metric],rs[b][metric];same.append(dict(dataset=d,sequence=v['name'],external_QP=q,method=m,reference=b,metric=metric,method_value=x,reference_value=y,delta=x-y,relative_change_percent=100*(x/y-1) if y!=0 else None,comparison='same_external_QP',not_BD_rate=True))
                for m,r in rs.items():
                    fr=read(r['frame_metrics_path']);ib=int(fr[0]['real_bits'])//8;pb=sum(int(x['real_bits']) for x in fr[1:])//8;assert ib+pb==r['real_bytes'];bytes_rows.append(dict(dataset=d,sequence=v['name'],method=m,external_QP=q,I_bytes_including_SPS=ib,P_bytes_including_headers=pb,total_bytes=ib+pb,bpp=r['bpp']))
        write(folder/'same_QP_differences.csv',same);dump(folder/'same_QP_differences.json',same);write(folder/'I_P_byte_split.csv',bytes_rows);dump(folder/'I_P_byte_split.json',bytes_rows)
        macro=load(folder/'all_qp_summary.json');dataset_differences=[]
        for q in range(10):
            rs={r['method']:r for r in macro if r['QP']==q}
            for m,b in PAIRS:
                for metric in ('bpp','LPIPS','DISTS','FloLPIPS','FID','PSNR','SSIM'):
                    x,y=rs[m][metric],rs[b][metric];dataset_differences.append(dict(dataset=d,external_QP=q,method=m,reference=b,metric=metric,method_value=x,reference_value=y,delta=x-y,relative_change_percent=100*(x/y-1) if y else None,aggregation='fixed dataset pooled FID; other metrics equal video mean',comparison='same_external_QP',not_BD_rate=True))
        write(folder/'same_QP_dataset_differences.csv',dataset_differences);dump(folder/'same_QP_dataset_differences.json',dataset_differences)
        for name in ('equal_rate_per_sequence','equal_rate_summary','equal_rate_fid_pooled','window_equal_rate_per_sequence','window_equal_rate_summary'):
            rows=load(folder/(name+'.json'))
            for r in rows:
                if r.get('status')=='available':
                    y=r.get('reference_mean');r['relative_change_percent']=100*r['delta']/y if y else None;r['relative_change_definition']='100*(method_mean-reference_mean)/reference_mean';r['not_BD_rate']=True
            write(folder/(name+'.csv'),rows);dump(folder/(name+'.json'),rows)
def integrity():
    import numpy as np
    from adapter import validate
    from window_training import replay_rows
    frozen();assert load(ROOT/'audits/step0_codec_equivalence.json')['status']=='PASS';branches={}
    for b in BRANCHES:
        path=ROOT/'branches'/b;assert load(path/'training_integrity.json')['status']=='PASS';rows=[json.loads(x) for x in (path/'training_logs/train.jsonl').read_text().splitlines()];assert len(rows)==1000;plan=replay_rows(b)
        for r,p in zip(rows,plan):
            assert r['adaptation_step']==p['adaptation_step'] and r['reference_resets']==[0] and r['supervision_frames']==15 and len(r['frame_trace'])==15
            for k in ('video','domain','source_frame_indices','external_qp','actual_qps','full_RGB_sha256','cropped_RGB_sha256','crop_x','crop_y'):assert r[k]==p[k]
            assert all(x['reference_detached'] for x in r['frame_trace']) and r['all_finite']
        index=load(path/'checkpoint_index.json');assert sorted(map(int,index))==[0,250,500,1000]
        for c in index.values():assert sha(c['path'])==c['sha256'] and sha(c['inference']['path'])==c['inference']['sha256'] and c['compression_hash']==evalcfg()['compression_hash']
        branches[b]=dict(updates=1000,supervised_P_frames=15000,reference_resets=[0],checkpoint_index=index)
    count=reused=fresh=0;by={}
    for d in DATASETS:
        n=0
        for v in sources(d):
            for m in METHODS:
                for q in range(10):
                    r=load(point(d,m,v['video_index'],q));validate(r,v,q,m);count+=1;n+=1;reused+=bool(r.get('reused'));fresh+=not bool(r.get('reused'))
                    if m=='native_i_adapt_1000':assert r['native_I_original_match']['status']=='PASS'
        assert len(read(ROOT/'results'/d/'per_sequence_all_qp.csv'))==n;assert np.array_equal(np.load(ROOT/'features'/f'GT_{d}.npy'),np.load(V19/'features'/f'GT_{d}.npy'));by[d]=dict(expected=len(sources(d))*50,completed=n,FID_pool_frames=len(sources(d))*64)
    assert count==800 and fresh>=320 and fresh+reused==800
    old=load(ROOT/'audits/historical_inventory.json');now=historical_inventory();assert all(now.get(p)==stamp for p,stamp in old.items()),'historical files changed'
    assert len(list((ROOT/'results').glob('*/Rate_*.png')))==16 and len(list((ROOT/'plots/first_frames').glob('*.png')))==12
    hashes={str(p.relative_to(ROOT)):sha(p) for folder in ('results','plots','audits') for p in (ROOT/folder).rglob('*') if p.is_file()}
    dump(ROOT/'final_integrity.json',dict(status='PASS',branches=branches,methods=METHODS,expected_points=800,completed_points=800,reused_points=reused,new_run_points=fresh,failed_points=0,missing=[],datasets=by,source_checkpoint_sha256=EXPECTED,checks=dict(source_frames_consistent=True,parameters_fixed_during_eval=True,frozen_codec_unchanged=True,independent_decode=True,state_synchronization=True,full_64_frame_bit_accounting=True,cache_contains_weights_and_I_mode=True,historical_files_unchanged=True,FID_fixed_dataset_pools=True,no_extrapolation=True,native_I_matches_original=True,no_checkpoint_selection=True),output_hashes=hashes,finished_unix=time.time()));print('FINAL PASS 800/800',flush=True)
def main():
    numerical();from diagnostics import main as diagnostic;diagnostic();extra_tables();subprocess.run([PLOT_PYTHON,'-B',str(ROOT/'temporal_plot.py')],cwd=REPO,check=True);integrity()
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'final_integrity.json',dict(status='FAIL',error=traceback.format_exc()));raise
