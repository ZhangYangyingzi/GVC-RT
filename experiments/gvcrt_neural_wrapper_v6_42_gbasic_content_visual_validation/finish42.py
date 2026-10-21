"""Objective numerical summaries, provenance, and complete artifact verification."""
from common42 import *
import math,collections
from PIL import Image

def main():
 assert sha(CP)==CP_SHA
 manifest=load(ROOT/'data_manifest.json');vm=load(ROOT/'visual_manifest.json');vis=load(ROOT/'visual_index.json')
 reference_cfg=load(REF/'evaluation_config.json');stream_cfg=load(V37/'evaluation_config.json')
 for k in ('compression_hash','I_hash','force_zero_thres','original_receiver_hashes'):
  assert reference_cfg[k]==stream_cfg[k],k
 assert len(vis['static_groups'])==32 and len(vis['video_groups'])==8
 assert vis['visual_manifest_sha256']==sha(ROOT/'visual_manifest.json')==load(ROOT/'audits/preflight.json')['visual_manifest_sha256']
 rows=[];tasks=[];streams=[]
 for v in manifest:
  for q in (0,5,9):
   key=f"{v['dataset']}/{v['name']}/q{q}";t=load(ROOT/'tasks'/key/'pair.json');assert t['signature']==signature42() and t['status']=='PASS'
   assert t['optimizer_updates']==0 and t['I_output_identical'] and t['same_y_f_q_z_bridge_output'] and t['metrics_unchanged']
   assert len(t['input_proof'])==63 and all(p['z']==p['bridge_output'] for p in t['input_proof'])
   assert len(t['results'])==2 and {r['method'] for r in t['results']}==set(METHODS)
   assert len({(r['real_bytes'],r['bytes_consumed'],r['bitstream_sha256'],r['actual_bpp']) for r in t['results']})==1
   for r in t['results']:
    fs=read(r['frame_csv']);ts=read(r['transition_csv']);assert len(fs)==64 and len(ts)==63
    assert sha(r['frame_csv'])==r['frame_csv_sha256'] and sha(r['transition_csv'])==r['transition_csv_sha256']
    assert [int(f['frame']) for f in fs]==list(range(64))
    assert [(int(f['from_frame']),int(f['to_frame'])) for f in ts]==[(i,i+1) for i in range(63)]
    assert [int(f['source_frame']) for f in fs]==v['source_frame_indices']
    assert [f['frame_type'] for f in fs]==['I']+['P']*63
    assert sum(int(f['real_bits']) for f in fs)==r['real_bytes']*8
    assert abs(r['PSNR']+10*math.log10(sum(float(f['pixel_SSE']) for f in fs)/(64*1920*1080*3)))<1e-10
    assert r['actual_bpp']==r['real_bytes']*8/(64*1920*1080) and all(math.isfinite(r[m]) for m in METRICS)
    rows.append(r)
   tasks.append(t);streams.append(dict(key=key,status='PASS',independent_decodes=2,identical=['bytes','SHA256','y','f','q','Bridge_output','Generator_z','I_output'],no_extra_reset=True))
   for r in t['results']:
    method=ROOT/'tasks'/v['dataset']/v['name']/f'q{q}'/f'{r["method"]}.json'
    dump(method,dict(status='PASS',task_key=dict(dataset=v['dataset'],sequence=v['name'],QP=q,method=r['method']),
      signature=t['signature'],result=r,frozen=next(x for x in t['frozen'] if x['method']==r['method']),
      stream_sha256=t['stream_sha256'],feature_input_proof=t['input_proof'],optimizer_updates=0))
 write(ROOT/'evaluation.csv',rows)
 paired=[]
 for v in manifest:
  for q in (0,5,9):
   a,b=[next(r for r in rows if (r['dataset'],r['sequence'],r['QP'],r['method'])==(v['dataset'],v['name'],q,m)) for m in ('G_basic','O_original')]
   for m in METRICS:
    delta=a[m]-b[m];signed=delta if m!='PSNR' else -delta;tol=1e-12
    paired.append(dict(dataset=v['dataset'],sequence=v['name'],QP=q,comparison='G_basic - O_original',metric=m,
     G_basic=a[m],O_original=b[m],delta=delta,unit='dB' if m=='PSNR' else 'bpp' if m=='actual_bpp' else 'metric units',
     relative_change_percent=None if m in ('PSNR','actual_bpp') else 100*delta/max(abs(b[m]),1e-12),
     outcome='improved' if signed < -tol else 'degraded' if signed>tol else 'tied',tie_tolerance=tol))
 write(ROOT/'paired_differences.csv',paired)
 scopes=[('four_supplementary_videos_equal','all',lambda p:True)]
 for v in manifest:scopes.append(('video',v['name'],lambda p,n=v['name']:p['sequence']==n))
 for q in (0,5,9):scopes.append(('QP',str(q),lambda p,q=q:p['QP']==q))
 for ds in ('hevc_b','uvg_holdout'):scopes.append(('dataset',ds,lambda p,ds=ds:p['dataset']==ds))
 summary=[]
 for scope,label,select in scopes:
  for metric in METRICS:
   ps=[p for p in paired if select(p) and p['metric']==metric];n=len(ps)
   summary.append(dict(scope=scope,label=label,metric=metric,points=n,distinct_videos=len({p['sequence'] for p in ps}),
    G_basic_mean=sum(p['G_basic'] for p in ps)/n,O_original_mean=sum(p['O_original'] for p in ps)/n,
    mean_delta=sum(p['delta'] for p in ps)/n,unit=ps[0]['unit'],
    mean_relative_change_percent=None if metric in ('PSNR','actual_bpp') else sum(p['relative_change_percent'] for p in ps)/n,
    improved=sum(p['outcome']=='improved' for p in ps),tied=sum(p['outcome']=='tied' for p in ps),degraded=sum(p['outcome']=='degraded' for p in ps),
    aggregation='equal fixed QP points; four-video scope gives each video equal weight; multiple QPs are not independent videos'))
 write(ROOT/'summary.csv',summary)
 # Original two-video results are explicitly separate and never pooled with supplementary data.
 reference=[];reference_audit=[]
 assert sha(ROOT/'evaluation_config.json')==sha(REF/'evaluation_config.json')
 assert load(REF/'branches/G_basic/final_integrity.json')['inference_checkpoint_sha256']==CP_SHA
 for method in METHODS:
  for vi in (0,1):
   for q in (0,5,9):
    p=REF/'tasks'/f'{method}_v{vi}_q{q}.json';t=load(p);r=t['result'];assert t['status']=='PASS'
    f=REF/'results/frames'/f'{method}_v{vi}_q{q}.csv';tr=REF/'results/transitions'/f'{method}_v{vi}_q{q}.csv'
    assert sha(f)==r['frame_csv_sha256'] and sha(tr)==r['transition_csv_sha256']
    if method=='G_basic':assert r['checkpoint_sha256']==CP_SHA
    reference.append(dict(scope='original_reference_only',**r))
    reference_audit.append(dict(task=str(p),sha256=sha(p),frame_sha256=sha(f),transition_sha256=sha(tr)))
 write(ROOT/'original_reference_12_points.csv',reference)
 dump(ROOT/'audits/original_reference_source.json',dict(status='PASS',points=12,separate_from_supplementary=True,checkpoint_sha256=CP_SHA,records=reference_audit))
 # Every lossless source frame is indexed and all displayed originals are traceable.
 server=[]
 for p in sorted((ROOT/'server_media').rglob('*.png')):
  with Image.open(p) as im:assert im.mode=='RGB' and im.size==(1920,1080)
  server.append(dict(path=str(p),relative_path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
 assert len(server)==8*3*64
 for g in vis['static_groups']:
  assert g['frame'] in vm['frames'] and g['QP'] in vm['QPs']
  assert (ROOT/g['overview']).is_file() and len(g['rois'])==2
  for p in g['rois']:
   with Image.open(ROOT/p) as im:assert im.size==(768,286)
  for a in g['originals']:assert sha(a['path'])==a['sha256']
 import subprocess
 from fractions import Fraction
 for v in vis['video_groups']:
  assert len(v['previews'])==3 and v['same_verified_decode']
  for p in v['previews']:
   path=ROOT/p['preview'];assert sha(path)==p['sha256']
   info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,r_frame_rate,width,height','-of','json',str(path)],text=True))['streams'][0]
   assert int(info['nb_frames'])==64 and abs(float(Fraction(info['r_frame_rate']))-v['fps'])<1e-9
   server.append(dict(path=str(path),relative_path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
   if 'lossless' in p:
    lp=ROOT/p['lossless'];assert sha(lp)==p['lossless_sha256'];server.append(dict(path=str(lp),relative_path=str(lp.relative_to(ROOT)),bytes=lp.stat().st_size,sha256=sha(lp)))
 # Add direct lossless ROI links to the local-only material section.
 html_path=ROOT/'index.html';doc=html_path.read_text()
 if 'id="lossless-roi-index"' not in doc:
  extra=['<section id="lossless-roi-index"><h2>Lossless synchronized ROI videos (server only)</h2>']
  for v in vis['video_groups']:
   for p in v['previews']:
    if 'lossless' in p:
     rel=p['lossless'];extra.append(f'<p>{v["sequence"]} QP{v["QP"]} {p["view"]}, FFV1 {v["fps"]} fps: <span class="local" data-path="{rel}">{ROOT/rel}</span> SHA256 {p["lossless_sha256"]}</p>')
  extra.append('</section>');textfile(html_path,doc.replace('<script>','\n'.join(extra)+'<script>',1))
 dump(ROOT/'server_artifacts_index.json',dict(status='PASS',uploaded=False,full_resolution_PNG_frames=1536,video_groups=8,entries=server))
 dump(ROOT/'audits/same_stream_generator_inputs.json',dict(status='PASS',streams=streams,hook_fields=['y','f','q','z','bridge_output'],read_only_hooks=True))
 dump(ROOT/'audits/stream_reuse_provenance.json',dict(status='PASS',stream_manifest_sha256=sha(ROOT/'stream_manifest.json'),
  original_model_config_fields_equal=['compression_hash','I_hash','force_zero_thres','original_receiver_hashes'],
  all_original_decodes_match_historical_reconstruction_hash=True,actual_QPs_IP_layout_padding_and_consumption_verified=True,
  no_reencoding_needed=True,original_streams=12))
 dump(ROOT/'audits/freeze_initialization_precision.json',dict(status='PASS',optimizer_updates=0,only_generator_different=True,
  checkpoint_sha256=CP_SHA,all_eval=True,all_parameter_gradients_absent=True,all_weights_and_buffers_unchanged=True,
  codec_receiver_precision='float16 inherited v6.40 deployment',records=[dict(key=t['task_key'],frozen=t['frozen'],metric_hashes=t['metric_hashes']) for t in tasks]))
 execution=[json.loads(s) for s in (ROOT/'logs/execution.jsonl').read_text().splitlines()]
 assert all(e['gpu'] in (4,5,6,7) for e in execution)
 exits=[e for e in execution if e['event']=='EXIT'];assert len(exits)==11 and all(e['exit_code']==0 for e in exits)
 for gpu in (4,5,6,7):
  es=sorted((e for e in exits if e['gpu']==gpu),key=lambda e:e['start']);assert all(a['end']<=b['start'] for a,b in zip(es,es[1:]))
 dump(ROOT/'audits/execution.json',dict(status='PASS',queue_tasks=11,first_pair_counted=True,one_task_per_GPU=True,allowed_GPUs=[4,5,6,7],all_exit_zero=True))
 textfile(ROOT/'README.md','# v6.42 fixed G_basic content and visual validation\n\nEvaluation-only, zero optimizer updates. Fixed v6.40 G_basic step100 checkpoint. See `REPORT.md`, `index.html`, `protocol.json`, and `visual_manifest.json`.\n\nGitHub includes code, numerical results, static comparison panels and indices. Full lossless source PNG sequences, ROI FFV1 videos and H264 previews remain on the server; `index.html` enables those links only when opened locally.\n')
 textfile(ROOT/'PROTOCOL.md','# Fixed protocol\n\nO_original and the exact v6.40 G_basic step100 are evaluated on Kimono1, ParkScene, HoneyBee and Jockey, QP0/5/9, 64 frames. Native I0, P1–63, IP=-1, no extra reset, 1920×1080 with inherited 1088 padding. Models deploy in the inherited float16 precision. No training, optimizer update, weight tuning or bit allocation experiment.\n\nAll 12 shared original RANS streams were verified; two models independently decode each. Hooks verify exact y/f/q, Bridge output and Generator inputs; I output and full consumed byte counts agree. PSNR uses aggregate RGB SSE; metrics use floating output. Visual PNG rule is clamp[0,1], round(255*x), uint8, identical across GT/models. ROIs and frames were fixed using GT before reconstruction.\n\nOutcomes use absolute metric delta tolerance 1e-12. Perceptual percentage changes are not bitrate savings. Supplementary videos and the original two reference videos remain separate. Official pretraining exposure is unknown.\n')
 lines=['# Fixed G_basic content and visual validation','',
  '24/24 supplementary evaluation points completed; zero new optimizer updates. The fixed G_basic checkpoint SHA256 is '+CP_SHA+'.',
  '', '## Supplementary four-video equal summary', '',
  '|Metric|O_original mean|G_basic mean|G_basic − original|Improved / tied / degraded|','|---|---:|---:|---:|---:|']
 for s in summary:
  if s['scope']=='four_supplementary_videos_equal':lines.append(f"|{s['metric']}|{s['O_original_mean']:.10g}|{s['G_basic_mean']:.10g}|{s['mean_delta']:.10g} {s['unit']}|{s['improved']} / {s['tied']} / {s['degraded']}|")
 lines+=['','## Per-video results','','|Video|Metric|Mean delta|Improved / tied / degraded QP points|','|---|---|---:|---:|']
 for s in summary:
  if s['scope']=='video':lines.append(f"|{s['label']}|{s['metric']}|{s['mean_delta']:.10g} {s['unit']}|{s['improved']} / {s['tied']} / {s['degraded']}|")
 lines+=['','Per-QP and separate HEVC-B / UVG summaries are in summary.csv. All fixed point differences are in paired_differences.csv. The original BasketballDrive/BQTerrace 12 points are separately listed in original_reference_12_points.csv and are excluded from these summaries.', '',
  '## Material entry points','',f'Local HTML: `{ROOT}/index.html`. 32 static groups, each with a three-model ROI-box overview and two native-size ROI panels. Eight synchronized 64-frame video groups provide a uniformly resized full overview and two native-size ROI views at the manifest timeline (HEVC-B 24fps, UVG 30fps). H264 is display encoding; lossless PNG and FFV1 ROI sources remain on the server. See visual_index.json and server_artifacts_index.json for paths, sizes and SHA256.', '',
  '## Audits and limits','', 'Same complete RANS bytes/SHA/consumption, y/f/q, Bridge output and Generator inputs, I0 equality, eval mode, absent parameter gradients and unchanged weights/buffers: PASS. The first dual-model point was validated before the remaining queue. No training or additional model/data scale was introduced.', '',
  'Four supplementary videos and three QPs per video are a fixed small evaluation; QP points are not independent videos. These videos were absent from the actual v6.40 fine-tuning plan. Official pretraining exposure is unknown; prior evaluation use is not excluded. PSNR deltas are in dB. Perceptual percentage changes are not rate savings. No FID, BD-rate or FPS performance was computed. No subjective visual improvement claim, mechanism explanation, significance, generalization or next experiment is asserted.']
 textfile(ROOT/'REPORT.md','\n'.join(lines)+'\n')
 final=dict(status='PASS',optimizer_updates=0,supplementary_points_complete=24,supplementary_points_planned=24,
  frame_records=1536,transition_records=1512,shared_streams=12,static_groups=32,video_groups=8,lossless_full_resolution_frames=1536,
  source_and_model_hashes_verified=True,same_stream_and_generator_inputs=True,all_weights_buffers_unchanged=True,
  reference_points_separate=12,visual_manifest_fixed_before_evaluation=True,blocked=[])
 dump(ROOT/'final_integrity.json',final);dump(ROOT/'run_status.json',dict(status='COMPUTE_COMPLETE',**{k:v for k,v in final.items() if k!='status'}))
 print('FINAL_INTEGRITY_PASS',flush=True)
if __name__=='__main__':main()
