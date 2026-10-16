"""Cache exactly verified historical crops, with no data/geometry/order change."""
from runtime import *
import argparse
import os
import time

def main(shard,shards):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    folder=ROOT/'cache/training_crops';folder.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(load(V32/'plans/shared.json')['updates']):
        if i%shards!=shard:continue
        key=f'{i+1:04d}';p=folder/(key+'.npy');receipt=folder/(key+'.json')
        if p.exists() and receipt.exists():continue
        f=source_frames(row,'cpu');arrays=[]
        try:
            for j in range(64):
                x=f[j];a=x[0].permute(1,2,0).mul(255).round().byte().numpy()
                assert hashlib.sha256(a.tobytes()).hexdigest()==row['cropped_RGB_sha256'][j]
                assert torch.equal(torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255,x)
                arrays.append(a)
        finally:f.close()
        tmp=p.with_suffix('.tmp')
        with tmp.open('wb') as stream:np.save(stream,np.stack(arrays))
        tmp.replace(p)
        dump(f'cache/training_crops/{key}.json',dict(status='PASS',row_sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),source_full_RGB_hashes_verified=True,all_64_cropped_RGB_hashes_verified=True,float32_model_inputs_exact=True,cache_sha256=sha(p),source_path=row['source_path'],source_sha256=row['source_sha256'],bytes=p.stat().st_size))
        if i%50<shards:print('CACHE',i+1,flush=True)
    dump(f'cache/shard_{shard}.json',dict(status='PASS',shard=shard,shards=shards))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True);a=ap.parse_args()
    main(a.shard,a.shards)
