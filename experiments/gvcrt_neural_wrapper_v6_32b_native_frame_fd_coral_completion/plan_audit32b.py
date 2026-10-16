"""Exact field-level verification of frozen plans against their training manifests."""
from io32 import *
import collections
def main():
 manifests={d:load(ROOT/f'train_manifest_{d}.json')['videos'] for d in ('ulong','uvg')};lookup={d:{v['filename' if d=='ulong' else 'name']:v for v in vs} for d,vs in manifests.items()};counts={};checks=[]
 for name,n in [('shared',1000),('calibration',40)]:
  rows=load(ROOT/'plans'/f'{name}.json')['updates'];assert len(rows)==n;counter=collections.Counter()
  for i,r in enumerate(rows):
   d=r['domain'];v=lookup[d][r['video']];assert r['source_path']==v['path' if d=='ulong' else 'source_path'] and r['source_sha256']==v['sha256' if d=='ulong' else 'source_sha256'];assert len(r['source_frame_indices'])==len(r['full_RGB_sha256'])==len(r['cropped_RGB_sha256'])==64;assert r['canonical_indices']==list(range(r['canonical_indices'][0],r['canonical_indices'][0]+64))
   if d=='uvg':assert r['source_frame_indices']==[v['source_frame_indices'][j] for j in r['canonical_indices']]
   else:assert r['source_frame_indices']==r['canonical_indices']
   h,w=v.get('height',1080),v.get('width',1920);assert 0<=r['crop_x']<=w-256 and 0<=r['crop_y']<=h-256;assert r['normalization_denominator']==15 and r['supervised_window_positions']==list(range(49,64)) and len(r['actual_qps'])==15
   counter[(d,r['external_qp'])]+=1;checks.append(dict(plan=name,index=i,video=r['video'],domain=d,QP=r['external_qp'],native_H=h,native_W=w,canonical_start=r['canonical_indices'][0],source_frame_indices=r['source_frame_indices'],crop=[r['crop_x'],r['crop_y'],256,256],source_sha256=r['source_sha256'],full_RGB_sha256=r['full_RGB_sha256'],cropped_RGB_sha256=r['cropped_RGB_sha256'],status='PASS'))
  assert set(counter.values())==({50} if name=='shared' else {2}) and len(counter)==20;counts[name]={f'{d}:{q}':v for (d,q),v in counter.items()}
 test=load(ROOT/'eval_manifest.json');tp={Path(v['source_path']).resolve() for vs in test.values() for v in vs};trainp={Path(r['source_path']).resolve() for r in load(ROOT/'plans/shared.json')['updates']};assert not tp&trainp
 dump(ROOT/'audits/plan_fields.json',dict(status='PASS',manifest_video_counts={d:len(vs) for d,vs in manifests.items()},counts=counts,canonical_source_frame_indices_verified=True,crop_bounds_verified=True,exact_source_path_intersection=[],shared_plan_sha256=sha(ROOT/'plans/shared.json'),calibration_plan_sha256=sha(ROOT/'plans/calibration.json'),rows=checks))
 a=load(ROOT/'data_isolation_audit.json');a.update(actual_manifest_video_counts={d:len(vs) for d,vs in manifests.items()},plan_field_audit=str(ROOT/'audits/plan_fields.json'),source_path_intersection=[],canonical_source_frame_indices_verified=True,crop_bounds_verified=True);dump(ROOT/'data_isolation_audit.json',a)
 print('PLAN AUDIT PASS', {d:len(vs) for d,vs in manifests.items()})
if __name__=='__main__':main()
