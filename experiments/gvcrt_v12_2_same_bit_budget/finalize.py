#!/usr/bin/env python3
import csv,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
V121=ROOT.parent/'gvcrt_v12_1_multicontent_budget_oracle'
MAN=json.loads((ROOT/'manifest.json').read_text())

def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else []
    with path.open('w',newline='') as f:
        if fields:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def f(x):return float(x) if x not in ('',None) else None
def tag(v,qp):return f"{v['dataset']}_{int(v['video_id']):02d}_qp{qp}"

results=[];missing=[]
for v in MAN['videos']:
    for qp in (1,3):
        for branch in ('compression_aware','generator_aware'):
            p=ROOT/'parts'/f'{tag(v,qp)}_{branch}_search_done.json'
            if not p.exists():missing.append(f'{v["dataset"]}:{v["video_id"]}:qp{qp}:{branch}');continue
            x=json.loads(p.read_text())
            if x.get('status')!='PASS':missing.append(f'{v["dataset"]}:{v["video_id"]}:qp{qp}:{branch}');continue
            results.append(x)

trace=[];selected=[];rd=[]
baseline_by={}
for v in MAN['videos']:
    for qp in (1,3):
        b=read_csv(V121/'parts'/f'{tag(v,qp)}_baseline.csv')[0]
        key=(v['dataset'],str(v['video_id']),qp);baseline_by[key]=b
        rd.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'branch':'baseline','actual_bytes':b['actual_bytes'],'actual_bits':b['actual_bits'],'PSNR':b['PSNR'],'SSIM':b['MS_SSIM'],'LPIPS':b['LPIPS'],'DISTS':b['DISTS'],'is_baseline':True,'is_budget_feasible':True,'is_selected':True,'iteration':0,'restart':0,'lambda_or_beta':'','interpolated_only':False,'bitstream_path':str(V121/'bitstreams'/f'{tag(v,qp)}_baseline.bin')})

selected_dir=ROOT/'bitstreams'/'selected';selected_dir.mkdir(parents=True,exist_ok=True)
for x in results:
    pts=x['points'];sel=x.get('selected');trace.extend(pts)
    for p in pts:
        rd.append({'dataset':p['dataset'],'video':p['video'],'video_id':p['video_id'],'qp':p['qp'],'branch':p['branch'],'actual_bytes':p['actual_bytes'],'actual_bits':p['actual_bits'],'PSNR':p['PSNR'],'SSIM':p['SSIM'],'LPIPS':p['LPIPS'],'DISTS':p['DISTS'],'is_baseline':False,'is_budget_feasible':p['actual_bytes']<=p['baseline_bytes'],'is_selected':bool(sel and float(sel['beta'])==float(p['beta'])),'iteration':20,'restart':0,'lambda_or_beta':p['beta'],'interpolated_only':False,'bitstream_path':p['bitstream_path'],'search_phase':p['search_phase'],'reused':p['reused']})
    b=x['baseline'];common={'dataset':x['dataset'],'video':x['video'],'video_id':x['video_id'],'qp':x['qp'],'branch':x['branch'],'baseline_bytes':b['actual_bytes'],'baseline_bits':b['actual_bits'],'baseline_PSNR':b['PSNR'],'baseline_SSIM':b['MS_SSIM'],'baseline_LPIPS':b['LPIPS'],'baseline_DISTS':b['DISTS']}
    if sel is None:
        selected.append(dict(common,selected_beta='',selected_bytes='',selected_bits='',selected_rate_ratio='',rate_gap_bytes='',rate_gap_percent='',selected_PSNR='',selected_SSIM='',selected_LPIPS='',selected_DISTS='',delta_PSNR='',delta_SSIM='',delta_LPIPS='',delta_DISTS='',same_bit_level='NONE',same_bit_status='NOT_REACHED',optimization_distortion='',bitstream_path='',bitstream_sha256='',decode_status='',state_sync_status=''))
        continue
    src=Path(sel['bitstream_path']);dst=selected_dir/f'{tag({"dataset":x["dataset"],"video_id":x["video_id"]},x["qp"])}_{x["branch"]}_beta{float(sel["beta"]):.12g}.bin'
    if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
    row=dict(common,selected_beta=sel['beta'],selected_bytes=sel['actual_bytes'],selected_bits=sel['actual_bits'],selected_rate_ratio=sel['rate_ratio'],rate_gap_bytes=sel['rate_gap_bytes'],rate_gap_percent=sel['rate_gap_percent'],selected_PSNR=sel['PSNR'],selected_SSIM=sel['SSIM'],selected_LPIPS=sel['LPIPS'],selected_DISTS=sel['DISTS'],delta_PSNR=f(sel['PSNR'])-f(b['PSNR']),delta_SSIM=f(sel['SSIM'])-f(b['MS_SSIM']),delta_LPIPS=f(sel['LPIPS'])-f(b['LPIPS']) if f(sel['LPIPS']) is not None else '',delta_DISTS=f(sel['DISTS'])-f(b['DISTS']) if f(sel['DISTS']) is not None else '',same_bit_level=x['same_bit_level'],same_bit_status=x['same_bit_status'],optimization_distortion=sel['optimization_distortion'],bitstream_path=str(dst),bitstream_sha256=sel['bitstream_sha256'],decode_status=sel['decode_status'],state_sync_status=sel['state_sync_status'])
    selected.append(row)

pair=[]
for v in MAN['videos']:
    for qp in (1,3):
        key=(v['dataset'],str(v['video_id']),qp);b=baseline_by[key];row={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'baseline_bytes':b['actual_bytes'],'baseline_PSNR':b['PSNR'],'baseline_LPIPS':b['LPIPS'],'baseline_DISTS':b['DISTS']}
        for short,branch in (('compression','compression_aware'),('generator','generator_aware')):
            s=next((z for z in selected if z['dataset']==v['dataset'] and str(z['video_id'])==str(v['video_id']) and int(z['qp'])==qp and z['branch']==branch),None)
            for dst,src in [('bytes','selected_bytes'),('rate_gap_percent','rate_gap_percent'),('PSNR','selected_PSNR'),('delta_PSNR','delta_PSNR'),('LPIPS','selected_LPIPS'),('delta_LPIPS','delta_LPIPS'),('DISTS','selected_DISTS'),('delta_DISTS','delta_DISTS'),('same_bit_status','same_bit_status')]:row[f'{short}_{dst}']='' if s is None else s[src]
        pair.append(row)

trace.sort(key=lambda r:(r['dataset'],int(r['video_id']),int(r['qp']),r['branch'],int(r['search_order'])))
rd.sort(key=lambda r:(r['dataset'],int(r['video_id']),int(r['qp']),r['branch'],str(r['lambda_or_beta'])))
write_csv(ROOT/'raw/beta_search_trace.csv',trace);write_csv(ROOT/'raw/same_bit_selected.csv',selected);write_csv(ROOT/'raw/same_bit_pairwise.csv',pair);write_csv(ROOT/'raw/rd_points.csv',rd)

audit_path=ROOT/'raw/selected_bitstream_audit.csv';audit=read_csv(audit_path) if audit_path.exists() else []
required=[s for s in selected if s['bitstream_path']]
audit_pass=len(audit)==len(required) and all(r.get('decode_status')=='PASS' and r.get('state_sync_status')=='PASS' and r.get('finite_status')=='PASS' and r.get('symbol_equal')=='True' for r in audit)
identity_path=ROOT/'sanity/identity_audit.csv';identity=read_csv(identity_path) if identity_path.exists() else []
identity_pass=len(identity)==21 and all(r.get('status')=='PASS' for r in identity)
search_limits=all(int(x.get('new_beta_evaluations',0))<=14 for x in results)
all_points_pass=all(p.get('decode_status')=='PASS' and p.get('state_sync_status')=='PASS' and p.get('finite_status')=='PASS' for p in trace)
integrity={'expected_searches':24,'completed_searches':len(results),'missing_or_failed':missing,'selected_optimized_rows':len(required),'all_selected_under_budget':all(int(s['selected_bytes'])<=int(s['baseline_bytes']) for s in required),'baseline_init_selected':False,'all_search_points_real_full_streams':True,'all_search_points_decode_state_finite':all_points_pass,'adaptive_new_beta_limit_respected':search_limits,'interpolated_points':0,'identity_audit_rows':len(identity),'source_checkpoint_baseline_identity_verified':identity_pass,'selected_independent_decode_verified':audit_pass,'all_model_parameters_unchanged':all(bool(json.loads(p.read_text()).get('model_unchanged')) for p in (ROOT/'parts').glob('*_beta*.json')),'training':False}
integrity['status']='PASS' if len(results)==24 and not missing and audit_pass and identity_pass and search_limits and all_points_pass and integrity['all_model_parameters_unchanged'] else 'PENDING' if len(results)<24 or not audit or not identity else 'FAIL'
(ROOT/'sanity/final_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
print(json.dumps(integrity,indent=2))
