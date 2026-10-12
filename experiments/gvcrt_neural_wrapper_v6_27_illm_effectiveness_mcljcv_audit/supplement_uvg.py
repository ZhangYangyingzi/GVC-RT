"""Complete unseen diagnostics from permitted source identities, outside v626 intervals."""
from io21 import *
import random,copy,numpy as np
from PIL import Image

def main():
 old=load(ROOT/'diagnostic_plan.json');old_sha=sha(ROOT/'diagnostic_plan.json');assert old['unseen_available']==10
 audit=ROOT/'audits/diagnostic_plan_revisions';audit.mkdir(exist_ok=True);dump(audit/'primary_50_window_plan.json',old)
 expanded_path=V18/'train_manifest_uvg_expanded.json';expanded=load(expanded_path)['videos'];base=load(ROOT/'train_manifest_uvg.json')['videos'];allowed={v['name']:v for v in base};eligible=[]
 for v in expanded:
  assert v['name'] in allowed and v['source_sha256']==allowed[v['name']]['source_sha256'] and v['source_frame_indices'][:64]==allowed[v['name']]['source_frame_indices'] and v['frame_rgb_sha256'][:64]==allowed[v['name']]['frame_rgb_sha256'] and v['first64_match']
  if v['frames']>=128:eligible.append(v)
 assert eligible;eligible.sort(key=lambda v:v['name']);rng=random.Random(20261027);rows=copy.deepcopy(old['updates']);shared=load(ROOT/'plans/shared.json')['updates'];extras=[]
 for q in range(10):
  v=rng.choice(eligible);start=rng.randrange(64,v['frames']-63);x=rng.randrange(v['width']-255);y=rng.randrange(v['height']-255);indices=list(range(start,start+64));full=[];cropped=[];files=[]
  for i in indices:
   p=Path(v['input_dir'])/f'frame_{i:06d}.png';assert sha(p)==v['frame_file_sha256'][i]
   with Image.open(p) as im:assert im.mode=='RGB' and im.size==(1920,1080);a=np.asarray(im).copy()
   h=hashlib.sha256(a.tobytes()).hexdigest();assert h==v['frame_rgb_sha256'][i];full.append(h);cropped.append(hashlib.sha256(a[y:y+256,x:x+256].copy().tobytes()).hexdigest());files.append(sha(p))
  row=dict(adaptation_step=len(rows)+1,panel='unseen',panel_id=f'unseen_uvg_q{q}_0',domain='uvg',video=v['name'],source_path=v['source_path'],source_sha256=v['source_sha256'],canonical_indices=indices,source_frame_indices=[v['source_frame_indices'][i] for i in indices],crop_x=x,crop_y=y,crop_size=256,external_qp=q,actual_qps=load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][49:64],normalization_denominator=15,supervised_window_positions=list(range(49,64)),full_RGB_sha256=full,cropped_RGB_sha256=cropped,diagnostic_input_dir=v['input_dir'],diagnostic_frame_file_sha256=files,diagnostic_source_manifest=str(expanded_path),diagnostic_source_manifest_SHA256=sha(expanded_path),diagnostic_only_source_extension=True,shuffle_panel_id=f'seen_uvg_q{q}_0')
  assert all(not (r['domain']=='uvg' and r['video']==row['video'] and set(r['source_frame_indices'])&set(row['source_frame_indices'])) for r in shared);rows.append(row);extras.append(row)
 for a,b in zip(old['updates'],rows[:50]):assert a==b
 new=dict(old,updates=rows,unseen_available=20,unavailable=[],supplement=dict(seed=20261027,requested_video_identities_unchanged=True,source_canonical_rule='same native source and stride4; select canonical interval starting>=64, outside v626 interval0..63',eligible=[v['name'] for v in eligible],not_eligible=dict(ShakeNDry='75 canonical frames: no disjoint64-frame interval'),source_manifest=str(expanded_path),source_manifest_SHA256=sha(expanded_path),original_plan_SHA256=old_sha,no_training=True,old50_windows_unchanged=True))
 dump(ROOT/'diagnostic_plan.json',new);new_sha=sha(ROOT/'diagnostic_plan.json');dump(audit/'revision.json',dict(status='PASS',old_plan_SHA256=old_sha,new_plan_SHA256=new_sha,old50_rows_identical=True,accepted_plan_hashes=[old_sha,new_sha],added_windows=10,added_checkpoint_window_points=40,reason='Initial preparation restricted to cached64-frame training pools; original permitted UVG source videos have additional legal canonical windows. Complete requested unseen panels without adding video identities or training.',selection_only_from_source_lengths_and_fixed_seed=True,no_test_or_metric_selection=True,source_path=str(expanded_path),source_sha256=sha(expanded_path)))
 ph=load(ROOT/'audits/protocol_hashes.json');ph['diagnostic_plan.json']=new_sha;dump(ROOT/'audits/protocol_hashes.json',ph);dump(ROOT/'audits/diagnostic_sampling.json',dict(status='PASS',seen=40,unseen=20,unavailable=[],source_interval_disjoint=True,training_video_identities_only=True,no_training=True,seed=20261027,plan_sha256=new_sha,revision=str(audit/'revision.json')));print('UVG UNSEEN SUPPLEMENT PASS',flush=True)
if __name__=='__main__':main()
