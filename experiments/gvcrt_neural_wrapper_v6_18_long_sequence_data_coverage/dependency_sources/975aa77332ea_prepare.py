import random
import subprocess
from diag_utils import *

def main():
    assert old.load('final_integrity.json')['status']=='PASS'
    assert not (ROOT/'sample_manifest.json').exists(),'Sampling already frozen'
    rng=random.Random(20260927);selected=[]
    for dataset,n in [('uvg',3),('ulong',2)]:
        videos=old.videos(dataset)
        for i in rng.sample(range(len(videos)),n):
            v=videos[i];path=Path(v['path'] if dataset=='uvg' else v['source_path'])
            digest=sha(path);assert digest==v.get('sha256',v.get('source_sha256'))
            selected.append(dict(dataset=dataset,video_index=i,name=v.get('sequence_name',v.get('name')),
                source_path=str(path),source_sha256=digest,random_seed=20260927,video=v))
    dump('sample_manifest.json',dict(random_seed=20260927,sampling='Python random.Random; UVG then U-Long; random.sample on frozen manifest order',
        samples=selected,external_qps=list(QPS),methods=list(METHODS)))
    protected={}
    for p in sorted(PARENT.rglob('*')):
        if p.is_file() and ROOT not in p.parents:protected[str(p)]=sha(p)
    config=old.load('config.json')
    checkpoint_hashes={v['path']:v['sha256'] for v in config['checkpoints'].values() if v['path']}
    checkpoint_hashes.update(old.load('checkpoint_metadata_audit.json')['additional_source_hashes'])
    assert all(sha(p)==h for p,h in checkpoint_hashes.items())
    dump('protected_inputs.json',dict(existing_files=protected,checkpoints_and_model_sources=checkpoint_hashes))
    for d in ('logs','parts','videos','contact_sheets','color_conversion','bt709_smoke','rd_curves/ulong'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    helptext=subprocess.check_output(['ffmpeg','-hide_banner','-h','filter=scale'],text=True,stderr=subprocess.STDOUT)
    (ROOT/'logs/ffmpeg_scale_configuration.txt').write_text(helptext)
    assert all(s in helptext for s in ('in_color_matrix','bt709','in_range','out_range'))
    print('EXPLICIT FILTER: scale=in_color_matrix=bt709:in_range=limited:out_range=full,format=rgb24',flush=True)
    dump('pipeline_status.json',dict(status='PREPARED',evaluation_only=True))
    print('SAMPLES',[(s['dataset'],s['name']) for s in selected],flush=True)

if __name__=='__main__':main()
