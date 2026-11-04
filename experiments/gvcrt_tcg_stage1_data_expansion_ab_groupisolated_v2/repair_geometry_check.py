"""Focused native-pixel and resume-authority checks; no optimizer update or resampling."""
from ab_common import *
from ab_data_reader import expected_native_shape,read_arrays
import concurrent.futures,collections

def check_window(row):
 arrays=read_arrays(row,verify=True)
 hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
 assert len(arrays)==64 and hashes==row['cropped_RGB_sha256']
 return dict(source_sha256=row['source_sha256'],row_sha256=digest(row),native_shape=list(expected_native_shape(row)),frames=64,full_frame_hash_checks=64,crop_hash_checks=64,crop_hash_list_sha256=digest(hashes))

def main():
 transition_path=ROOT/'repairs/native_geometry_v1/transition.json';t=load(transition_path)
 assert sha(transition_path)=='9b5c1529bbedeb4e73dc92e43d2cc6bd211c8c75e35481d56e758090ae3c9a0f'
 assert sha(ROOT/'protocol.json')==t['protocol_sha256']
 current=load(ROOT/'source_manifest.json');old=load(ROOT/t['predecessor_source_manifest_file'])
 assert sha(ROOT/t['predecessor_source_manifest_file'])==t['predecessor_source_manifest_sha256']
 assert {n for n in current['code'] if current['code'][n]!=old['code'][n]}==set(t['allowed_changed_original_sources'])
 for n,h in current['code'].items():assert sha(ROOT/n)==h
 parent_meta={r['sha256']:r for r in load(P/'plans/train_manifest_ulong.local.json')['videos']}
 pools={};metadata_rows=0
 for arm in ('A_clean_old_pool','B_expanded_ulong'):
  index_path=ROOT/f'plans/{arm}/index.json';assert sha(index_path)==t['plan_index_sha256_unchanged'][arm];index=load(index_path)
  for shard in index['shards']:
   assert sha(ROOT/shard['file'])==shard['sha256']
   for row in load(ROOT/shard['file']):
    shape=expected_native_shape(row)
    if row['domain']=='ulong' and row['source_sha256'] in parent_meta:
     reference=parent_meta[row['source_sha256']];assert shape==(reference['height'],reference['width'],3)
    metadata_rows+=1
 windows=ROOT/t['affected_unique_windows_file'];assert sha(windows)==t['affected_unique_windows_sha256'];rows=load(windows);assert len(rows)==31
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as workers:checks=list(workers.map(check_window,rows,chunksize=1))
 import torch
 latest=ROOT/'A_clean_old_pool/checkpoints/latest.pt';assert sha(latest)==t['resume_authority']['checkpoint_sha256'];state=torch.load(latest,map_location='cpu',weights_only=True)
 assert state['additional_step']==state['plan_cursor']==10 and state['absolute_optimizer_step']==5010 and state['rank_sample_cursor']==40
 assert state['source_hashes']==old['code'] and state['protocol_sha256']==sha(ROOT/'protocol.json')
 assert {int(s['step']) for s in state['optimizer']['state'].values()}=={5010}
 assert len(state['all_rank_RNG_states'])==4 and all(x is None for b in state['TCG_buffers'] for x in b.values())
 receipts=sorted((ROOT/'A_clean_old_pool/receipts').glob('*.json.gz'));assert [int(p.name.split('.')[0]) for p in receipts]==list(range(1,11))
 for p in receipts:
  receipt=load(p);step=int(p.name.split('.')[0]);assert receipt['additional_step']==step and receipt['absolute_optimizer_step']==5000+step and receipt['rank_sync']=='PASS' and receipt['finite_loss'] and receipt['finite_nonzero_gradient']
 assert state['last_committed_receipt']==load(receipts[-1])
 assert sha(latest)==sha(ROOT/'repairs/native_geometry_v1/before/A_clean_old_pool/checkpoints/latest.pt')
 result=dict(status='PASS',scope='Focused exact native-frame/crop reads and original committed checkpoint/receipt preservation; no numerical model or final-experiment acceptance',recorded_unix=time.time(),source_revision_sha256=sha(ROOT/'source_manifest.json'),transition_sha256=sha(transition_path),metadata_rows_checked=metadata_rows,native_windows_checked=len(checks),native_frames_checked=64*len(checks),checks=checks,existing_checkpoint_byte_identical=True,optimizer_absolute_step=5010,all_rank_RNG_preserved=True,committed_receipts_contiguous=10,first_two_smoke_preserved=load(ROOT/'A_clean_old_pool/smoke.json')['status']=='PASS',next_additional_update=11,no_optimizer_update_replayed=True)
 dump(ROOT/'repairs/native_geometry_v1/focused_validation.json',result)
 event('NATIVE_GEOMETRY_REPAIR_VALIDATED',metadata_rows=metadata_rows,unique_windows=len(checks),next_additional_update=11)

if __name__=='__main__':main()
