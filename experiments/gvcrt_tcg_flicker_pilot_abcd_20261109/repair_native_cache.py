"""Preformal cache migration: only GT float normalization/RAFT references.

Original streams, features and base-flow fields are verified and copied unchanged.
The retired task-private derived tensor is deletable; its SHA receipt is retained.
"""
from common_s2 import *
from cache_codec import cache_paths,load_raft
from temporal import flow_valid
import argparse

def old_key(row,qp):return digest(dict(row=row,QP=qp,training=False,original_commit=ORIGINAL_COMMIT,RAFT_SHA256=RAFT_SHA,precision='original_FP16_RGB_FP32first',version='flicker_cache_v1'))

def main(rank):
    import torch
    env();os.environ['MODEL_KIND']='original'
    from original_context import runtime
    runtime(rank);device=torch.device('cuda',rank);raft,transform=load_raft(device)
    profiles=load(ROOT/'manifests/quick_development.json');records=[]
    for j,p in enumerate(profiles):
        if j%4!=rank:continue
        row=p['row'];q0key=old_key(row,0);oldpath=ROOT/'cache/derived'/f'{q0key}.pt'
        newkey,newpath,newreceipt=cache_paths(row,0,False)
        if not oldpath.exists():
            assert newpath.exists() and newreceipt.exists();source=torch.load(newpath,map_location='cpu',weights_only=True,mmap=True)
        else:source=torch.load(oldpath,map_location='cpu',weights_only=True,mmap=True)
        GT=source['GT'];h,w=source['info']['native_shape']
        gt_id=digest(dict(source_SHA256=row['source_sha256'],indices=row['source_frame_indices'],geometry=[h,w],RAFT_SHA256=RAFT_SHA,GT_normalization='CPU_FP32_div255_before_GPU'))
        gtpath=ROOT/'cache/GT_native_v2'/f'{gt_id}.pt';gtproof=ROOT/'audits/GT_native_v2'/f'{gt_id}.json'
        if source['info'].get('GT_reference_identity')==gt_id:
            refs={k:source[k] for k in ['GT_backward','GT_forward','GT_valid','GT_FP32_tensor_SHA256','GT_reference_identity']}
        elif gtpath.exists():
            assert sha(gtpath)==load(gtproof)['sha256'];refs=torch.load(gtpath,map_location='cpu',weights_only=True,mmap=True)
        else:
            backward=[None];forward=[None];valid=[None];hashes=[];normalization_check=None;start=time.time()
            def gt(i):return (GT[i].permute(2,0,1)[None].float()/255).to(device)
            def flow(a,b):
                a=torch.nn.functional.pad(a,(0,(-w)%8,0,(-h)%8),mode='replicate');b=torch.nn.functional.pad(b,(0,(-w)%8,0,(-h)%8),mode='replicate')
                a,b=transform(a,b);return raft(a,b)[-1][...,:h,:w]
            with torch.no_grad():
                for i in range(64):
                    target=gt(i);hashes.append(tensor_sha(target))
                    if i==0:
                        old=GT[i].permute(2,0,1)[None].to(device).float()/255
                        normalization_check=dict(max_abs=float((target-old).abs().max()),different_values=int((target!=old).sum()),CPU_target_SHA256=tensor_sha(target),GPU_old_target_SHA256=tensor_sha(old))
                    else:
                        b=flow(target,previous);f=flow(previous,target);v=flow_valid(b,f)
                        backward.append(b.cpu());forward.append(f.cpu());valid.append(v.cpu())
                    previous=target
            refs=dict(GT_backward=backward,GT_forward=forward,GT_valid=valid,GT_FP32_tensor_SHA256=hashes,GT_reference_identity=gt_id)
            atomic_torch(gtpath,refs);dump(gtproof,dict(status='PASS',sha256=sha(gtpath),source_SHA256=row['source_sha256'],normalization='CPU_FP32/255 then GPU',normalization_difference=normalization_check,seconds=time.time()-start,RAFT_SHA256=RAFT_SHA))
        for qp in [0,5,9]:
            oldid=old_key(row,qp);oldfile=ROOT/'cache/derived'/f'{oldid}.pt';oldreceipt=ROOT/'audits/codec_cache'/f'{oldid}.json'
            nk,np,nr=cache_paths(row,qp,False)
            if np.exists() and nr.exists():assert sha(np)==load(nr)['cache_sha256'];continue
            old=load(oldreceipt);assert sha(oldfile)==old['cache_sha256'] and sha(old['stream_path'])==old['stream_SHA256']
            data=torch.load(oldfile,map_location='cpu',weights_only=True,mmap=True);info=dict(data['info']);assert info['source_row']==row
            original_trace=info['codec_trace'];features=[None if i==0 else tensor_sha(f) for i,f in enumerate(data['features'])]
            assert features==[t['codec_feature_SHA256'] for t in original_trace]
            data.update(refs);info.update(cache_identity=nk,GT_normalization='CPU_FP32/255 then GPU',GT_reference_identity=gt_id,GT_FP32_tensor_SHA256=refs['GT_FP32_tensor_SHA256'],
                preformal_native_cache_migration_from=oldid,old_cache_sha256=old['cache_sha256'],codec_and_base_alignment_data_unchanged=True)
            data['info']=info;atomic_torch(np,data)
            proof=dict(info,status='PASS',cache_sha256=sha(np),cache_bytes=np.stat().st_size,GT_reference_proof=str(gtproof),GT_reference_proof_SHA256=sha(gtproof))
            dump(nr,proof)
            migrated=dict(old_identity=oldid,new_identity=nk,old_cache_sha256=old['cache_sha256'],new_cache_sha256=proof['cache_sha256'],stream_SHA256_unchanged=info['stream_SHA256'],codec_feature_hashes_unchanged=True,
                original_reconstruction_and_base_flow_unmodified=True,source_uint8_RGB_unmodified=True,GT_normalization_only_repaired=True,old_derived_tensor_retired=True,time=time.time())
            dump(ROOT/'audits/native_cache_migration'/f'{nk}.json',migrated);records.append(migrated)
            del data
            oldfile.unlink() # This task's rebuildable v1 tensor only.
        del source,GT,refs
        if gtpath.exists():gtpath.unlink() # References are embedded in all three immutable QP caches.
        print('NATIVE_GT_REFERENCE_MIGRATED',p['sequence'],flush=True)
    dump(ROOT/f'audits/native_cache_migration_rank{rank}.json',dict(status='PASS',migrated=records,time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);a=p.parse_args();main(a.rank)
