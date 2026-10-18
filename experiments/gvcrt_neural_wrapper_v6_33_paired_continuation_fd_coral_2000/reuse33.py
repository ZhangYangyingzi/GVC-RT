"""Read-only point reuse after exact historical signatures and artifact hashes."""
from io32 import *
from evaluate32 import task_id,receipt,eval_signature,valid,source,artifacts

def main():
 oldio=module('previous_io33',PREV/'io32.py');ns=dict(oldio.__dict__);exec(compile((PREV/'evaluate32.py').read_text().replace('from io32 import *',''),str(PREV/'evaluate32.py'),'exec'),ns)
 src_tasks=load(PREV/'task_manifest.json')['tasks'];fallback=[];mapping={(t['dataset'],t['video'],t['method'],t['qp']):t for t in src_tasks};records=[];cache={}
 for t in load(ROOT/'task_manifest.json')['tasks']:
  if t['stage']=='new':continue
  if t['stage']=='historical':
   ix=ROOT/'branches'/t['branch']/'checkpoint_index.json'
   if not ix.exists() or '1000' not in load(ix):continue
  try:
   oldt=mapping[(t['dataset'],t['video'],t['method'],t['qp'])];p=PREV/'tasks'/(ns['task_id'](oldt)+'.json');r=load(p)
   assert ns['valid'](oldt) and r['signature']==ns['eval_signature'](oldt)
   v=source(t);rr=r['result'];assert rr['source_rgb_sha256']==v['rgb_sha256'] and rr['source_frame_indices']==v['source_frame_indices']
   assert rr['real_RANS'] and rr['frames']==64 and rr['num_transitions']==63 and rr['state_sync_pass'] and rr['independent_decode_pass']
   assert rr['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(t['qp'])][:64]
   assert rr['compression_hash_before']==rr['compression_hash_after']==load(ROOT/'evaluation_config.json')['compression_hash']
   if t['stage']=='historical':
    ix=load(ROOT/'branches'/t['branch']/'checkpoint_index.json')['1000'];assert rr['loaded_receiver_module_hashes']==ix['deployment_receiver_hashes']
   assert rr['bytes_consumed']==rr['real_bytes']==Path(rr['bitstream_path']).stat().st_size and abs(rr['bpp']-rr['real_bytes']*8/(64*1920*1080))<1e-15
   for path,md in r['artifact_metadata'].items():
    if path not in cache:cache[path]=sha(path)
    assert cache[path]==md['sha256']
   assert len(read(rr['frame_metrics_path']))==64 and len(read(rr['transitions_path']))==63
   arts=dict(r['artifact_metadata']);arts[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
   dump(receipt(t),dict(r,signature=eval_signature(t),task=t,result=dict(rr,method=t['method']),historical_signature=r['signature'],read_only_reuse=True,fresh_encode=False,reuse_receipt=str(p),reuse_receipt_sha256=sha(p),artifact_metadata=arts,source_protocol_sha256=sha(PREV/'protocol.json'),new_protocol_sha256=sha(ROOT/'protocol.json')))
   records.append(dict(task=t,source=str(p),source_sha256=sha(p),source_signature=r['signature'],protocol_mapping='unchanged codec/metric/RGB; new experiment namespace and step labels only'))
  except Exception as exc:
   fallback.append(dict(task=t,reason=repr(exc),action='fresh real encoding and independent decode with the fixed source model'))
 # Initial migration snapshots must exist before a historical branch point has a new signature.
 if len(records)+len(fallback)!=2300:
  dump(ROOT/'audits/reuse_progress.json',dict(status='WAITING_FOR_MIGRATION',completed=len(records),expected=2300));return
 dump(ROOT/'reuse_manifest.json',dict(status='PASS',completed=len(records),expected=2300,fresh_fallback=fallback,records=records,artifact_files_verified=len(cache),metric_protocol='identical inherited source hashes/official features/native I/real RANS/decode'))
 print('HISTORICAL REUSE PASS',len(records),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  error(dict(stage='reuse33',error=traceback.format_exc()));raise
