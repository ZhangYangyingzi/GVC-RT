"""Dataset-internal stratification; references never influence sampling."""
import collections
import random
import math
from audit_io import *

def balance(tasks):
    shards={str(g):[] for g in (4,5,6,7)};loads={g:0 for g in shards}
    for task in sorted(tasks,key=lambda t:(-t['workload'],t['sample_id'],t.get('QP',-1))):
        gpu=min(loads,key=lambda g:(loads[g],int(g)));shards[gpu].append(task);loads[gpu]+=task['workload']
    return dict(status='PASS',rule='Deterministic descending workload; stable sample/QP tie-break; assign least loaded GPU with numeric tie-break',shards=shards,loads=loads)

def motion_plan():
    videos=load(ROOT/'manifests/all_sources.json')['videos'];tasks=[]
    for v in videos:
        n=min(32,v['source_frames']);ratio=512/max(v['width'],v['height'])
        tasks.append(dict(sample_id=v['sample_id'],dataset=v['dataset'],workload=n*(v['width']*v['height']+round(v['width']*ratio)*round(v['height']*ratio))))
    dump(ROOT/'motion_shards.json',balance(tasks))

def main():
    import numpy as np
    assert load(ROOT/'source_progress.json')['status']=='PASS'
    videos=load(ROOT/'manifests/all_sources.json')['videos'];selected=[];audits=[]
    for dataset in ('ulong_train1024','ulong_unused1024','vimeo_train2048'):
        own=[v for v in videos if v['dataset']==dataset];values=[]
        for v in own:
            view=next(r for r in load(ROOT/'parts/source'/f'{v["sample_id"]}.json')['views'] if r['view']=='standardized_content_view')
            values.append((view['temporal']['temporal_RGB_L1'],view['spatial']['sobel_mean']))
        data=np.asarray(values);edges=np.quantile(data,[.25,.5,.75],axis=0);bins=collections.defaultdict(list)
        for v,z in zip(own,data):bins[int(np.searchsorted(edges[:,0],z[0],side='right'))*4+int(np.searchsorted(edges[:,1],z[1],side='right'))].append(v)
        count=min(256,len(own));quotas={b:int(count*len(group)/len(own)) for b,group in bins.items()}
        order=sorted(bins,key=lambda b:(-(count*len(bins[b])/len(own)-quotas[b]),b))
        for b in order[:count-sum(quotas.values())]:quotas[b]+=1
        rng=random.Random(SEED);subset=[]
        for b in sorted(bins):subset.extend(dict(v,codec_stratum=b,codec_frames=7 if v['kind']=='vimeo' else 32) for v in rng.sample(sorted(bins[b],key=lambda v:v['sample_id']),quotas[b]))
        assert len(subset)==count;selected.extend(subset)
        audits.append(dict(dataset=dataset,candidate_count=len(own),selected=count,quantile_edges=edges.tolist(),strata={b:len(x) for b,x in bins.items()},quotas=quotas,fit_reference_used=False))
    dump(ROOT/'manifests/codec_candidates.json',dict(status='PASS',seed=SEED,videos=selected,selection_audits=audits))
    tasks=[dict(sample_id=v['sample_id'],dataset=v['dataset'],QP=q,frames=v['codec_frames'],width=v['width'],height=v['height'],workload=v['codec_frames']*v['width']*v['height']) for v in selected for q in (0,4,9)]
    shards=balance(tasks);shards.update(task_count=len(tasks),reference_points_reused=69,estimated_workload_unit='frames*width*height',one_heavy_worker_per_gpu=True)
    dump(ROOT/'gpu_sharding_audit.json',shards)
    from pathlib import Path
    sys.path.insert(0,str(ROOT.parent/'gvcrt_parallel_v5a3_v60'));from engine import eco
    from src.models.video_model_gvcrt import DMC
    model=DMC();actual={str(q):[q if i==0 else model.shift_qp(q,eco.INDEX_MAP[i%8]) for i in range(96)] for q in (0,4,9)}
    dump(ROOT/'codec_protocol_audit.json',dict(status='PASS',actual_qps=actual,INDEX_MAP=eco.INDEX_MAP,force_zero_thres=.12,model='Original GVC-RT',
        wrapper=False,frames_ulong=32,frames_vimeo=7,Vimeo7_user_approved=True,comparison_fps_candidates=30,
        source_fps_unknown_for_vimeo=True,primary_rate_for_coverage='bpp',reference_reuse='Full original V6.2-B64-frame UVG and frozen VIRAT; native duration retained'))
    reused=[]
    for v in [x for x in videos if x['role']=='reference_test']:
        records=v.get('codec_reference_records',[dict(v,dataset='uvg')])
        for ref in records:
            d=ref['dataset'];index=ref['video_index'];sid=v['sample_id'] if d=='uvg' else v['sample_id']+'_'+d
            for q in (0,4,9):
                source=V62B/'parts'/d/'original'/f'video_{index:02d}_qp{q}.json';r=load(source)
                assert r['method']=='original' and r['force_zero_thres']==.12 and r['checkpoint_sha256']==''
                assert r['source_rgb_sha256']==ref['rgb_sha256'] and r['real_RANS'] and r['independent_decode_pass'] and r['state_sync_pass']
                for k in ('bitstream','feature','frame_metrics','transitions'):assert sha(r[k+'_path'])==r[k+'_sha256']
                assert Path(r['bitstream_path']).stat().st_size==r['real_bytes']==r['bytes_consumed']
                assert r['actual_qps']==actual[str(q)][:r['frames']]
                assert r['compression_hash_before']==load(V62B/'final_integrity.json')['shared_compression_hash']
                r.update(status='PASS',dataset='uvg_reference_test' if d=='uvg' else d,sample_id=sid,content_sample_id=v['sample_id'],
                    comparison_fps=r['rate_accounting_fps'],source_effective_fps=v['fps'],width=ref['width'],height=ref['height'],
                    reused=True,reused_point_path=str(source),reused_point_sha256=sha(source),original_only=True)
                dump(ROOT/'parts/codec'/sid/f'qp{q}.json',r);reused.append(dict(sample_id=sid,QP=q,source=str(source),source_sha256=sha(source)))
    dump(ROOT/'codec_reference_reuse_audit.json',dict(status='PASS',points=reused,no_reencoding=True,protocol_exact=True))
    print('CODEC COHORT FROZEN',len(selected),'fresh points',len(tasks),'reused',len(reused),flush=True)
if __name__=='__main__':main()
