from io32 import *
import subprocess

def main():
 cfg=load(ROOT/'evaluation_config.json');ev=load(ROOT/'eval_manifest.json');rows=read(ROOT/'results/reused_points.csv');records=[]
 for row in rows:
  p=Path(row['reuse_source_receipt']);assert sha(p)==row['reuse_source_receipt_sha256'],p
  if p.suffix=='.csv':
   rr=next(x for x in read(p) if int(x['video_index'])==int(row['video']) and int(x['QP'])==int(row['QP']) and x['method']=='original' and int(x['IP'])==-1)
   r={k:json.loads(v) if v.startswith(('[','{')) else v for k,v in rr.items()}
   for k in ('real_bytes','bytes_consumed'):r[k]=int(r[k])
   for k in ('real_RANS','independent_decode_pass','state_sync_pass'):r[k]=str(r[k]).lower()=='true'
   for k in ('LPIPS','DISTS','FloLPIPS'):r[k]=float(r[k])
  else:
   s=load(p);assert s['status']=='PASS';r=s['result']
  v=next(v for v in ev[row['dataset']] if v['video_index']==int(row['video']))
  assert r['source_rgb_sha256']==v['rgb_sha256'] and r['source_frame_indices']==v['source_frame_indices']
  assert r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][row['QP']][:64]
  assert r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass'] and r['bytes_consumed']==r['real_bytes']==int(row['real_bytes'])
  for kind in ('bitstream','frame_metrics','transitions'):assert r[kind+'_sha256']==row[kind+'_sha256'],(str(p),kind)
  # Official Inception aliases can replace the source CSV's legacy feature file;
  # all current cache files are independently re-extracted and compared by the replay audit.
  for metric in ('LPIPS','DISTS','FloLPIPS'):assert abs(float(row[metric])-r[metric])<=1e-12
  key={'O_original':'original','B_basicLoss_noP_500':'B_basicLoss_noP_500','F_fullLoss_noP_500':'N_oursLoss_noP_500'}[row['method']]
  expected=cfg['original_receiver_hashes'] if key=='original' else cfg['deployment_receiver_hashes'][key];assert r['loaded_receiver_module_hashes']==expected
  assert r['compression_hash_before']==r['compression_hash_after']==cfg['compression_hash']
  records.append(dict(dataset=row['dataset'],video=int(row['video']),QP=int(row['QP']),method=row['method'],receipt=str(p),sha256=sha(p),receiver_deployment_hashes=expected,all_input_model_metric_stream_fields_match=True))
 d=load(ROOT/'audits/reuse_pointwise_verification.json');d.update(deep_source_receipt_checks=records,all_original_receipt_SHA_fields_verified=True);dump(ROOT/'audits/reuse_pointwise_verification.json',d)
 source=load(ROOT/'source_manifest.json');paths=[str(Path(p).relative_to(REPO)) for p in source['files'] if p.endswith('.py')]
 raw=subprocess.check_output(['git','ls-tree','-r','origin/main','--',*paths],cwd=REPO,text=True);remote={line.split('\t',1)[1]:line.split('\t',1)[0].split()[-1] for line in raw.splitlines()};comparisons=[]
 for p in paths:
  v=source['files'][str(REPO/p)];comparisons.append(dict(path=p,executed_server_blob=v['git_blob'],remote_main_blob=remote.get(p),exact_remote_main_blob=remote.get(p)==v['git_blob'],server_sha256=v['sha256']))
 dump(ROOT/'audits/remote_dependency_comparison.json',dict(observed_main=subprocess.check_output(['git','rev-parse','origin/main'],cwd=REPO,text=True).strip(),runtime_HEAD=source['HEAD'],user_reference_main=source['user_reference_main'],comparisons=comparisons,server_only_or_different=[x for x in comparisons if not x['exact_remote_main_blob']],policy='Read-only comparison; executed server source SHA is authoritative and endpoint replay verifies numerical compatibility; no rollback or overwrite.'))
 print('REUSE DEEP AUDIT PASS',len(records),flush=True)
if __name__=='__main__':main()
