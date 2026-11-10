"""Coverage, immutable sources, matched stream and parameter checks; fail closed."""
from common_s2 import *
from collections import Counter

def main():
 env();import torch
 from results import records,all_tables
 from selection import points,complete
 from experiment_model import construct,restore,category
 verify_sources();sel=load(ROOT/'selection.json');assert sel['status']=='FROZEN';done=load(ROOT/'training_completion.json');assert done['status']=='PASS';checks={};rows=plan_rows();active=[]
 for group in GROUPS:
  if group==GROUPS[2] and load(ROOT/'audits/H2_gate.json')['status']!='PASS':
   assert sel['groups'][group]['status']=='MECHANICAL_CHECK_FAILED' and done['updates'][group]==0 and not list((ROOT/group/'training/receipts').glob('*.json'));continue
  active.append(group);assert done['updates'][group]==100
  assert load(ROOT/f'audits/engineering_{group}.json')['status']=='PASS' and load(ROOT/f'audits/aggregation_{group}.json')['status']=='PASS'
  for step in [0,25,50,75,100]:assert (ROOT/group/f'training/checkpoints/step_{step:04d}.pt').exists()
  a=torch.load(ROOT/group/'training/latest.pt',map_location='cpu',weights_only=True);assert a['committed_updates']==100 and a['next_cursor']==400 and all(int(v['step'])==100 for v in a['optimizer']['state'].values());assert a['B300_SHA256']==B300_SHA and a['s']==.5
  for i in range(1,101):
   receipt=load(ROOT/group/f'training/receipts/{i:04d}.json');assert receipt['global_optimizer_updates']==i and receipt['effective_batch']==4 and receipt['sample_keys']==[r['sample_key'] for r in rows[4*(i-1):4*i]];assert receipt['post_clip_norm']<=1.0001 and receipt['physical_GPU']==GROUPS.index(group) and math.isfinite(receipt['loss'])
  c=construct(group,False,0);restore(c,a,False);assert len([m for m in c['models']['generator'].modules() if m.__class__.__name__=='ParallelConvLoRA'])==5;assert all(m.inference_gain==.5 for m in c['models']['generator'].modules() if m.__class__.__name__=='ParallelConvLoRA');assert sum(p.numel() for p in c['params'])==998276
  base=torch.load(ROOT/'checkpoints/common_B300_step0.pt',map_location='cpu',weights_only=True)['recon_state']
  for n,t in c['pm'].recon_generation_net.state_dict().items():
   if 'recon_generation_net.'+n not in c['trainable_parameter_names']:assert torch.equal(t.cpu(),base[n].to(t.dtype)),('FROZEN_RECON_PARAMETER_OR_BUFFER_CHANGED',n)
  assert all(t.dtype==torch.float32 for t in a['trainable_state'].values());del a,c;torch.cuda.empty_cache()
  for s in [50,100]:complete(points(group,s))
 checks.update(full_B300_LoRA_retained_s05=True,full_Bridge_TCG_LoRA_exact998276=True,frozen_backbone_and_buffers=True,real100_updates_and_matched_data=True,all_checkpoints_optimizer_RNG_cursor_present=True,H2_mechanical_failure_explicit=True)
 init=load(ROOT/'initialization_audit.json');assert all(a['max_abs']==0 for a in init['equality']) and init['common_step0_SHA256']==sha(ROOT/'checkpoints/common_B300_step0.pt');checks['step0_matches_B300_in_both_matching_precision_paths']=True
 complete(points('B300',0));all_records=records('development')+records('final');identity={};observed=Counter()
 for a in all_records:
  assert a['status']=='PASS' and a['real_RANS'] and a['actual_I_frame_bits_included'] and a['input_feature_trace_checked'] and not a['reconstruction_uses_GT'] and not a['reconstruction_uses_Teacher'];r=a['result'];observed[(r['phase'],r['model'],r['step'])]+=1
  for f,h in a['record_SHA256'].items():assert sha(ROOT/f)==h
  key=(r['phase'],r['dataset'],r['sequence'],r['QP']);ids=tuple(r[k] for k in ['stream_SHA256','real_bytes','bpp','codec_trace_SHA256','source_SHA256','source_RGB_SHA256'])
  if key in identity:assert identity[key]==ids,('STREAM_OR_FEATURE_TRAJECTORY_DIFFERS',key,r['model'])
  identity[key]=ids
  framespath=next(f for f in a['record_SHA256'] if '/frames/' in f);transpath=next(f for f in a['record_SHA256'] if '/transitions/' in f);frames=list(csv.DictReader((ROOT/framespath).open()));trans=list(csv.DictReader((ROOT/transpath).open()));assert len(frames)==64 and len(trans)==63 and sum(int(f['real_bits']) for f in frames)==r['real_bytes']*8
  for metric in ['LPIPS_Alex','LPIPS_VGG','DISTS']:
   assert abs(r[metric]-sum(float(f[metric]) for f in frames)/64)<1e-12
  for metric in ['RGB_MCResidual','Sobel_MCResidual']:
   values=[float(t[metric]) for t in trans[1:] if t[metric]!=''];assert values and abs(r[metric+'_PtoP']-sum(values)/len(values))<1e-12
 checks['real_RANS_I_headers_sources_and_feature_trajectory_equal']=True;checks['metric_tables_recomputed_from_frame_transition_records']=True
 final=load(ROOT/'final_test_status.json')
 if not sel['any_qualified']:assert final['status']=='NOT_RUN_NO_QUALIFIED_CANDIDATE' and not records('final');checks['no_automatic_final_test_without_qualification']=True
 else:
  assert final['status']=='PASS' and final['selection_SHA256']==sha(ROOT/'selection.json')
  for model in final['models']:
   s=sel['groups'][model[:-9]]['selected']['step'] if model.endswith('_selected') else 0;assert observed[('final',model,s)]==225
  for ds,n in load(ROOT/'protocol.json')['final']['datasets'].items():
   for model in final['models']:assert sum(a['result']['dataset']==ds and a['result']['model']==model for a in records('final'))==n*3
  for a in records('final'):
   if a['result']['model']=='Original':assert a['original_baseline_source']['commit']==ORIGINAL_COMMIT and a['original_baseline_source']['no_v6_TCG_LoRA_B300_weights_in_Original_codec']
  checks['conditional_test_all_datasets_complete_independent_Original']=True
 data=all_tables();aggregate=list(csv.DictReader((ROOT/'rd_points_by_dataset.csv').open()))
 for a in aggregate:
  z=[r for r in data[a['phase']] if r['model']==a['model'] and r['dataset']==a['dataset'] and r['step']==int(a['step']) and r['QP']==int(a['QP'])];assert len(z)==int(a['videos'])
  for k in ['bpp','PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','Sobel_MCResidual_PtoP']:assert abs(float(a[k])-sum(r[k] for r in z)/len(z))<1e-12
 checks['equal_video_aggregates_recomputed']=True
 media=load(ROOT/'video_manifest.json');assert media['status']=='PASS' and len(media['native_clips'])==3*(4+len(active)) and len(media['ROI_comparisons'])==10
 for a in media['native_clips']+media['ROI_comparisons']:assert sha(a['path'])==a['SHA256']
 assert {a['sequence'] for a in media['native_clips']}=={'24747b35-04ca-416c-b1d0-0bc670fbac99','28a7c428-1a8d-4bd1-afbc-fed596ec6153','dce599d3-d196-4d87-b2b1-698acdbbb3d4'};checks['native_FFV1_and_unscaled_ROI_anonymous_materials_including_person_dog']=True
 plots=load(ROOT/'plots/manifest.json');assert plots['no_cross_dataset_curves'] and plots['actual_RANS_bpp'] and plots['no_BD_rate']
 for p in plots['files']:assert sha(p['path'])==p['SHA256']
 checks['per_dataset_plots_and_measured_three_QP_points']=True
 for path,h in load(ROOT/'source_inputs.json')['readonly'].items():assert sha(path)==h
 assert load(G_SOURCE/'status.json')['status']=='PAUSED';checks['old_experiments_unmodified_and_G_still_PAUSED']=True
 dump(ROOT/'checkpoint_manifest.json',dict(common_step0=dict(path=str(ROOT/'checkpoints/common_B300_step0.pt'),SHA256=sha(ROOT/'checkpoints/common_B300_step0.pt')),files=[dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size) for g in active for p in sorted((ROOT/g/'training/checkpoints').glob('*.pt'))],selected={g:sel['groups'][g].get('selected') for g in GROUPS}))
 dump(ROOT/'final_integrity.json',dict(status='PASS',complete_local=True,checks=checks,updates=done['updates'],development_points=sum(v for (phase,m,s),v in observed.items() if phase=='development'),final_test_status=final['status'],H2_status=load(ROOT/'audits/H2_gate.json')['status'],publication_complete=False,selection_SHA256=sha(ROOT/'selection.json'),time=time.time()))
if __name__=='__main__':main()
