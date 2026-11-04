"""New engineering audit of read-only P/E1 inputs; reuse is never new A/B evaluation."""
from ab_common import *
import csv,collections,zipfile
def parent_rows():return [r for f in sorted((P/'plans').glob('base_rows_*.json')) for r in load(f)]
def main():
 with lock('reuse.lock'):
  assert git('rev-parse','HEAD')==BASELINE
  task=load(ROOT/'task_identity.json');assert task['task']==TASK_ID
  assert sha(P/'checkpoints/step_5000.pt')==PARENT_SHA
  pins=load(P/'source_manifest.json');sources=[]
  for group in ('training_sources','evaluation_sources','runtime_acceleration','engineering_sources'):
   for name,h in pins.get(group,{}).items():
    path=P/name if '/' not in name else P.parents[1]/name
    assert sha(path)==h,('PARENT_SOURCE_CHANGED',name);sources.append(dict(file=name,sha256=h))
  for cp in load(P/'checkpoints_manifest.json')['checkpoints']:
   f=P/'checkpoints'/cp['filename'];assert f.stat().st_size==cp['size_bytes'] and sha(f)==cp['sha256']
  import torch
  st=torch.load(P/'checkpoints/step_5000.pt',map_location='cpu',weights_only=True)
  assert st['protocol_sha256']==sha(P/'protocol.json')
  assert sha(P/'data_plan.json')==load(P/'protocol.json')['data_plan_sha256']
  parent_plan=load(P/'data_plan.json');assert sha(P/'plans/schedule.json')==parent_plan['schedule_sha256']==load(P/'protocol.json')['schedule_sha256']
  for shard in parent_plan['shards']:assert sha(P/'plans'/shard['file'])==shard['sha256']
  assert st['committed_global_updates']==st['plan_cursor']==5000 and len(st['all_rank_RNG_states'])==4
  assert {int(s['step']) for s in st['optimizer']['state'].values()}=={5000}
  assert all(x is None for b in st['TCG_buffers'] for x in b.values())
  e1_index={r['file']:r for r in load(E1/'SHA256_MANIFEST.json')['files']};reused=[]
  def verify(name):
   f=E1/name;r=e1_index[name];assert f.stat().st_size==r['size_bytes'] and sha(f)==r['sha256'],('E1_INPUT_CHANGED',name)
   reused.append(dict(file=name,size_bytes=f.stat().st_size,sha256=r['sha256']));return f
  for name in ('inventory/qualified.json','inventory/entries.json','inventory/archives.json','manifests/reserved_union.local.json','manifests/historical_project_identities.json','audits/GTFlow_sources.json'):
   f=verify(name);dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
  history=load(ROOT/'manifests/historical_project_identities.json')
  historical_sources=[]
  for source in history['source_manifests']:
   f=ORIGINAL/source['file'];assert sha(f)==source['sha256'],('HISTORICAL_METADATA_CHANGED',source['file']);historical_sources.append(source)
  for source in load(ROOT/'manifests/reserved_union.local.json')['sources']:
   f=P/source['file'][7:] if source['file'].startswith('parent/') else ORIGINAL/source['file']
   assert sha(f)==source['sha256'],('RESERVED_METADATA_CHANGED',source['file'])
  inv=load(ROOT/'inventory/qualified.json');assert len(inv)==8859 and len({r['identity'] for r in inv})==8859
  archive_lookup={(r['archive'],r['entry']):r for r in inv};archives=[]
  for f in sorted(ARCHIVES.glob('*.zip')):
   with zipfile.ZipFile(f) as z:
    for i in z.infolist():
     if not i.filename.lower().endswith('.mp4'):continue
     r=archive_lookup[(f.name,i.filename)];assert r['size_bytes']==i.file_size and r['crc32']==f'{i.CRC:08x}' and r['compressed_bytes']==i.compress_size and r['CRC_verified']
   archives.append(dict(filename=f.name,size_bytes=f.stat().st_size))
  assert len(archives)==11
  points=[];keys=set();GT=collections.defaultdict(set);frame_count=transition_count=0
  for f in sorted((E1/'evaluation/parent_gtflow/points').glob('*.json')):
   relative=str(f.relative_to(E1));point=load(verify(relative));r=point['result'];key=(r['domain'],r['sequence'],r['QP'],r['step']);assert key not in keys;keys.add(key)
   assert point['status']=='PASS' and point['input_stream_reused'] and not point['fresh_encode'] and point['independent_decode'] and point['model_hashes_unchanged']
   name=f.stem;old=load(P/f'evaluation/points/{name}.json');assert point['reconstruction_sha256']==old['reconstruction_sha256']
   cp=P/f"checkpoints/step_{r['step']}.pt";assert sha(cp)==r['checkpoint_sha256']==old['result']['checkpoint_sha256']
   stream=P/'large_artifacts/evaluation/bitstreams'/r['bitstream_file'];assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==point['bytes_consumed']==r['real_bytes']
   for metric in ('bpp','kbps','PSNR','MS_SSIM','LPIPS','DISTS','FloLPIPS','WarpingError'):
    assert math.isfinite(r[metric]) and abs(r[metric]-old['result'][metric])<1e-6
   paths=[]
   for subdir,expected in [('frames',64),('transitions',63)]:
    name2=f'evaluation/parent_gtflow/{subdir}/{name}.csv';src=verify(name2);rows=list(csv.DictReader(src.open()));assert len(rows)==expected
    if subdir=='frames':
     frame_count+=len(rows);assert sum(int(x['real_bits']) for x in rows)==8*r['real_bytes']
     for metric in ('MS_SSIM','LPIPS','DISTS'):assert abs(sum(float(x[metric]) for x in rows)/64-r[metric])<1e-10
    else:
     transition_count+=len(rows)
     for x in rows:GT[(r['domain'],r['sequence'],int(x['to_frame']))].add(x['GT_flow_sha256'])
     for metric in ('FloLPIPS','WarpingError','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC'):
      assert math.isfinite(r[metric]) and abs(sum(float(x[metric]) for x in rows)/63-r[metric])<1e-10
    dest=ROOT/name2;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);paths.append(name2)
   dest=ROOT/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
   points.append(dict(point=name,source_receipt=relative,source_receipt_sha256=sha(f),checkpoint_sha256=r['checkpoint_sha256'],stream_sha256=r['bitstream_sha256'],reconstruction_sha256=point['reconstruction_sha256'],reused_from='E1',new_AB_model_evaluation=False,input_stream_reused=True,fresh_encode=False))
  expected={(v['dataset'],v['name'],q,s) for vs in load(P/'evaluation/manifest.local.json').values() for v in vs for q in (0,5,9) for s in (0,1000,2500,5000)}
  assert keys==expected and len(keys)==156 and frame_count==9984 and transition_count==9828
  assert len(GT)==13*63 and all(len(v)==1 for v in GT.values())
  from gtflow import write
  write(ROOT/'evaluation/parent_gtflow/metrics.csv',[load(f)['result'] for f in sorted((ROOT/'evaluation/parent_gtflow/points').glob('*.json'))])
  dump(ROOT/'evaluation/parent_gtflow/run_status.json',dict(status='PASS',completed_points=156,frames=9984,transitions=9828,reused_from_E1=True,fresh_encode=False,new_AB_model_evaluation=False))
  dump(ROOT/'evaluation/parent_gtflow/reuse_manifest.json',dict(status='PASS',source='E1/evaluation/parent_gtflow',source_E1_commit='c224ac23e9fcbaf4d51bc73808f589c08111761b',points=points,files=reused,common_GT_hashes_verified=True))
  dump(ROOT/'audits/source_and_parent.json',dict(status='PASS',baseline_commit=BASELINE,parent_checkpoint=dict(filename='step_5000.pt',sha256=PARENT_SHA,size_bytes=(P/'checkpoints/step_5000.pt').stat().st_size),optimizer_absolute_step=5000,rank_RNG_count=4,verified_sources=sources,P_and_E1_read_only=True))
  dump(ROOT/'audits/reuse.json',dict(status='PASS',inventory_entries=8859,archive_central_directories_checked=True,source_scope='E1 per-entry CRC/SHA/metadata/native-first64 only; not all native windows',archives=archives,reused_points=156,frames=9984,transitions=9828,files=reused,historical_manifest_hashes_verified=True))
  event('READ_ONLY_REUSE_VERIFIED',inventory=8859,parent_GTFlow_points=156,frames=frame_count,transitions=transition_count)
if __name__=='__main__':main()
