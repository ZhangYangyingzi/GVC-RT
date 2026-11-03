"""Deterministic expansion of existing audited samples; no validation sampling."""
import collections,copy,json,random,subprocess
from common_expanded import *
def main():
 records=load(POOL/'plans/shared.json')['updates'];assert len(records)==5000
 base=[x['source_row'] for x in records]
 videos=load(ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500/eval_manifest.json')
 held=[v for vs in videos.values() for v in vs]
 train_ids={Path(r['video']).stem.casefold() for r in base};test_ids={v['name'].casefold() for v in held}
 hashes={r['source_sha256'] for r in base};assert not train_ids&test_ids
 assert not hashes&{v['source_sha256'] for v in held}
 assert len(train_ids)>=940
 rng=random.Random(SEED);used=collections.defaultdict(set);samples={'uvg':[],'ulong':[]}
 for i,r in enumerate(base):
  key=(r['video'],tuple(r['source_frame_indices']),r['crop_x'],r['crop_y'])
  qs=list(range(10));rng.shuffle(qs);qs=[q for q in qs if q not in used[key]][:4];assert len(qs)==4
  for q in qs:
   used[key].add(q);h=hashlib.sha256(repr((key,q)).encode()).hexdigest()
   samples[r['domain']].append(dict(base_row=i,external_qp=q,sample_sha256=h))
 for a in samples.values():rng.shuffle(a);assert len(a)==10000
 schedule=[]
 for i in range(5000):
  a=samples['uvg'][2*i:2*i+2]+samples['ulong'][2*i:2*i+2];rng.shuffle(a)
  assert len({r['sample_sha256'] for r in a})==4
  schedule.append(dict(global_update=i+1,ranks=a))
 assert len({x['sample_sha256'] for a in schedule for x in a['ranks']})==20000
 shardfiles=[]
 for start in range(0,len(base),100):
  p=ROOT/f'plans/base_rows_{start//100:03d}.json'
  p.write_text(json.dumps(base[start:start+100],separators=(',',':'))+'\n')
  shardfiles.append(dict(file=p.name,sha256=sha(p),rows=min(100,len(base)-start)))
 dump(ROOT/'plans/schedule.json',dict(seed=SEED,updates=schedule))
 data=dict(seed=SEED,source_plan_sha256=sha(POOL/'plans/shared.json'),base_rows=5000,
           global_updates=5000,world_size=4,rank_samples=20000,unique_exact_combinations=20000,
           controlled_exact_repeats=0,base_content_combinations=len(used),
           base_content_reuse='Four distinct QPs per source row; one shared content key has eight distinct QPs',
           domains={d:len(a) for d,a in samples.items()},video_identities=len(train_ids),
           multiple_temporal_windows=len({(r['video'],tuple(r['source_frame_indices'])) for r in base}),
           unique_spatial_crops=len({(r['video'],r['crop_x'],r['crop_y']) for r in base}),
           external_qp_counts=dict(collections.Counter(x['external_qp'] for s in schedule for x in s['ranks'])),
           shards=shardfiles,schedule_sha256=sha(ROOT/'plans/schedule.json'))
 dump(ROOT/'data_plan.json',data)
 dump(ROOT/'split_audit.json',dict(status='PASS',training_identity_count=len(train_ids),
      training_identity_sha256=hashlib.sha256(json.dumps(sorted(train_ids)).encode()).hexdigest(),
      heldout_identities=sorted(test_ids),identity_overlap=[],source_hash_overlap=[],
      source_manifest='v6.37 eval_manifest.json',source_manifest_sha256=sha(ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500/eval_manifest.json')))
 dump(ROOT/'evaluation/manifest.local.json',videos)
 for d in ('uvg','ulong'):
  dump(ROOT/f'plans/train_manifest_{d}.local.json',load(POOL/f'train_manifest_{d}.json'))
 for n in ('GVC-RT_I.pt','GVC-RT_P.pt'):
  p=REPO/'checkpoints'/n
  assert not p.exists(),f'Unexpected model already present: {p}'
  p.symlink_to(ORIGINAL/'checkpoints'/n)
 # Only local model links; never staged or uploaded.
 docs=[ORIGINAL/'experiments/Xue 等 - 2026 - Single-step diffusion-based video coding with semantic-temporal guidance.pdf',
       ORIGINAL/'experiments/gemini-code-1793611318837.md',POOL/'protocol.json',
       ORIGINAL/'experiments/gvcrt_neural_wrapper_v6_37_loss_bridge_generator_mechanism_500/MECHANISM_PROTOCOL.md',
       ORIGINAL/'experiments/gvcrt_tcg_stage1_pilot/protocol.json',
       ORIGINAL/'experiments/gvcrt_tcg_stage1_evaluation/protocol.json']
 dump(ROOT/'audits/reference_constraints.json',dict(documents=[dict(file=p.name,sha256=sha(p)) for p in docs],
      constraints=['Paper section 3.3: previous generator features, multiscale residual fusion, zero residual projection.',
                   'Paper motion objective compares adjacent original/reconstructed RAFT flows; retain pilot motion weight 0.01 rather than paper weight 0.25.',
                   'Paper temporal gradient cascade is intentionally excluded: user requires detached histories and no BPTT.',
                   'Paper appendix warping uses next original frame against warped previous reconstruction; record repository backward sampling direction.',
                   'Local blueprint Stage 2 router and all gates are excluded.',
                   'v6.34 supplies audited identities/windows/crops; its FD/stat objectives are excluded.',
                   'v6.37 supplies heldout ULong6/UVG2/HEVC-B5; use native-I IP=-1 FP16 deployment.',
                   'Pilot preserves frame-loss VJP ordering/scaling, FP16 non-reconstruction codec, FP32 bridge/generator, frozen quality and RAFT.']))
 protocol=dict(seed=SEED,global_updates=5000,world_size=4,physical_gpus=[4,5,6,7],effective_batch_size=4,
      trainable_prefixes=list(PREFIXES),stage=1,no_router=True,no_gates=True,
      source_checkpoint=dict(filename=SOURCE.name,sha256=sha(SOURCE),size_bytes=SOURCE.stat().st_size),
      initialization='pilot step 0 zero-residual baseline; pilot step 100 is not a formal result or initialization',
      data_plan_sha256=sha(ROOT/'data_plan.json'),schedule_sha256=data['schedule_sha256'],
      train_video_count=len(train_ids),unique_exact_samples=20000,controlled_exact_repeats=0,
      frames=64,crop=[256,256],warmup_P=[1,48],supervised_P=[49,63],
      previous_reconstruction_detach=True,codec_reference_detach=True,TCG_feature_detach=True,no_BPTT=True,
      codec_precision='FP16 excluding bridge/generator; bridge/generator FP32 exactly as pilot; trainable TCG FP32',
      frame_loss='v6.40 LPIPS + 0.5 DISTS + beta(q)*rate_bpp + 0.01 proxy_L1 + 0.1383589248954108 reconstruction_L1',
      reconstruction_backward_scale=4096,motion_weight=.01,raft='torchvision raft_large C_T_SKHT_V2 local cache only',
      raft_sha256=sha(RAFT),raft_transform='C_T_SKHT_V2.transforms()',
      AdamW=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=1e-4),global_clip=1.0,
      gradient_reduction='NCCL SUM over packed FP32 gradients divided by world_size, then clipping, one logical optimizer update',
      checkpoint_steps=list(STEPS),evaluation_steps=[0,1000,2500,5000],primary_checkpoint=5000,
      evaluation_domains=list(videos),evaluation_QP=[0,5,9],evaluation_frames=64,
      evaluation_encode='fresh RANS stream per checkpoint/sequence/QP; lockstep and fresh independent decode',
      warping_error='mean L1(original[t+1], backward_grid_sample(rec[t], flow(rec[t+1],rec[t])))',
      warping_direction='PWC next-to-previous flow; backward sampler evaluates previous reconstruction on next-frame coordinates',
      no_hyperparameter_search=True,no_validation_selection=True,no_OOM_fallback=True)
 dump(ROOT/'protocol.json',protocol)
 event('PLAN_PASS',video_identities=len(train_ids),rank_samples=20000,exact_repeats=0)
if __name__=='__main__':main()
