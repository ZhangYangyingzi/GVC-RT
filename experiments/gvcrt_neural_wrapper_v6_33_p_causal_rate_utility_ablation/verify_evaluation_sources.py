"""Read-only availability/hash check of all formal canonical RGB PNG inputs."""
from audit import *
from concurrent.futures import ThreadPoolExecutor

def check(item):
    dataset,v,i=item
    name=f'im{i+1:05d}.png' if dataset=='hevc_b' else f'frame_{i:06d}.png'
    p=Path(v['input_dir'])/name
    expected=v['frame_png_sha256' if dataset=='hevc_b' else 'frame_file_sha256'][i]
    actual=sha(p) if p.is_file() else None
    return dict(dataset=dataset,video=v['video_index'],sequence=v['name'],frame=i,path=str(p),expected_sha256=expected,actual_sha256=actual,PASS=actual==expected)

if __name__=='__main__':
    manifest=load(V32/'eval_manifest.json')
    items=[(d,v,i) for d,vs in manifest.items() for v in vs for i in range(64)]
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(check,items))
    failed=[r for r in records if not r['PASS']]
    dump('evaluation_source_integrity.json',dict(status='PASS' if not failed else 'INPUT_UNAVAILABLE_OR_CHANGED',videos=sum(map(len,manifest.values())),frames=len(records),failed=failed,records=records,scope='canonical PNG bytes; actual RGB arrays are additionally hash-checked by the unchanged formal frame reader'))
    print('EVALUATION INPUT AUDIT',len(records),'failed',len(failed),flush=True)
