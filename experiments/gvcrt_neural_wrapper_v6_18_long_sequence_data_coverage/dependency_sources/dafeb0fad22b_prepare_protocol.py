"""Audit immutable sources and lossless canonical frames before fresh evaluation."""
import fcntl
import hashlib
import os
import subprocess
import tempfile
from fractions import Fraction
import numpy as np
from PIL import Image
from audit_utils import *
from canonical_loader import canonical_arrays, load_canonical_frames

V51 = ROOT.parent / 'gvcrt_neural_wrapper_v5a_1_allqp_uvg_generalization'
DIAG = V51 / 'visual_diagnostics'

def verify_hashes(mapping):
    for path, expected in mapping.items():
        assert sha(path) == expected, f'Hash mismatch: {path}'

def source_inventory():
    records = {}
    for base in (V51, V5, V41):
        for path in sorted(base.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                stat = path.stat()
                records[str(path)] = [stat.st_size, stat.st_mtime_ns]
    return records

def prepare():
    lock = (ROOT / 'parts/prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dump('pipeline_status.json', dict(status='RUNNING', phase='canonical_audit', pid=os.getpid(), evaluation_only=True))
    if not (ROOT / 'source_inventory_before.json').exists():
        dump('source_inventory_before.json', source_inventory())
    old = load(V51 / 'config.json')
    meta = load(V51 / 'checkpoint_metadata_audit.json')
    assert meta['status'] == 'PASS'
    checkpoints = {m: old['checkpoints'][m] for m in METHODS}
    verify_hashes({v['path']: v['sha256'] for v in checkpoints.values() if v['path']})
    verify_hashes(meta['additional_source_hashes'])
    verify_hashes(old['source_hashes'])
    metric = load(V51 / 'metric_implementation_audit.json')
    verify_hashes(metric['source_and_weight_sha256'])
    assert sha(ROOT / 'test_manifest.json') == old['ulong_manifest_sha256']
    assert sha(ROOT / 'uvg_available_manifest.json') == old['uvg_manifest_sha256']
    assert sha(ROOT / 'metric_implementation_audit.json') == old['metric_audit_sha256']
    frozen = dict(old['source_hashes'])
    for path in (V51/'config.json', V51/'final_integrity.json', V51/'ulong_all_qp_summary.csv',
                 V51/'uvg_all_qp_summary.csv', V51/'qp_semantics_audit.json', DIAG/'uvg_color_conversion_audit.json'):
        frozen[str(path)] = sha(path)
    config = dict(old, schema='V5-A.2 fair 30fps lossless RGB', methods=list(METHODS), checkpoints=checkpoints,
                  canonical_fps=30.0, canonical_num_frames=64, canonical_resolution=[1920,1080],
                  source_hashes=frozen, additional_source_hashes=meta['additional_source_hashes'],
                  FID_samples_per_method_QP={'ulong':512,'uvg':448}, fresh_encode_required=True,
                  allowed_gpus=[4,5,6,7], runtime_gpus=[4,6])
    dump('config.json', config)
    dump('checkpoint_audit.json', dict(status='PASS', checkpoints=checkpoints,
         original_base_checkpoints={p:h for p,h in meta['additional_source_hashes'].items() if p.endswith('.pt')},
         evidence=str(V51/'config.json'), metadata_evidence=str(V51/'checkpoint_metadata_audit.json'),
         hashes_match_existing_experiments=True))
    conversion = load(DIAG/'uvg_color_conversion_audit.json')
    assert conversion['status']=='PASS' and conversion['one_frame_smoke_pass']
    template = conversion['conversions'][0]['explicit_command']
    filter_text = template[template.index('-vf')+1]
    filters = subprocess.check_output(['ffmpeg','-hide_banner','-filters'], text=True, stderr=subprocess.STDOUT)
    scale_help = subprocess.check_output(['ffmpeg','-hide_banner','-h','filter=scale'], text=True, stderr=subprocess.STDOUT)
    assert 'scale' in filters and 'in_color_matrix' in scale_help and 'in_range' in scale_help and 'out_range' in scale_help
    dump('ffmpeg_conversion_audit.json', dict(status='PASS', inherited_command=template,
         inherited_audit_sha256=sha(DIAG/'uvg_color_conversion_audit.json'), filter=filter_text,
         scale_available=True, colorspace_available='colorspace' in filters, temporal_selection='Python index % 4 == 0 after exact inherited RGB conversion',
         ffmpeg_version=subprocess.check_output(['ffmpeg','-version'], text=True).splitlines()[0]))
    records=[]; loader_records=[]
    for dataset in ('ulong','uvg'):
        for index, video in enumerate(videos(dataset)):
            name=video.get('name',video.get('sequence_name'))
            source=video['source_path'] if dataset=='ulong' else video['path']
            expected=video['source_sha256'] if dataset=='ulong' else video['sha256']
            assert sha(source)==expected, source
            selected=list(range(64)) if dataset=='ulong' else list(range(0,253,4))
            if dataset=='ulong':
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                    '-show_entries','stream=width,height,avg_frame_rate,r_frame_rate,pix_fmt','-of','json',source],text=True))['streams'][0]
                assert (probe['width'],probe['height'])==(1920,1080)
                fps=float(Fraction(probe['avg_frame_rate']))
                assert min(abs(fps-30),abs(fps-30000/1001))<1e-5, fps
                command=['ffmpeg','-v','error','-threads','1','-i',source,'-vsync','0','-frames:v','64',
                         '-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
            else:
                assert video['width']==1920 and video['height']==1080 and video['num_frames']>=253
                fps=120.0; probe={'width':1920,'height':1080,'pix_fmt':'yuv420p','r_frame_rate':'120/1'}
                command=list(template)
                command[command.index('-i')+1]=source
                command[command.index('-frames:v')+1]='253'
            directory=ROOT/'canonical_sources'/dataset/name
            directory.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256(); frame_hashes=[]; raw_hashes=[]
            with tempfile.TemporaryFile() as errors:
                process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=errors)
                try:
                    for source_index in range(selected[-1]+1):
                        raw=process.stdout.read(1920*1080*3)
                        assert len(raw)==1920*1080*3,(name,source_index,'incomplete source frame')
                        if source_index not in selected: continue
                        i=len(frame_hashes); path=directory/f'frame_{i:06d}.png'
                        if not path.exists():
                            array=np.frombuffer(raw,np.uint8).reshape(1080,1920,3)
                            Image.fromarray(array).save(path,format='PNG',compress_level=1)
                        with Image.open(path) as image:
                            assert image.mode=='RGB' and image.size==(1920,1080),path
                            assert image.tobytes()==raw,f'Canonical RGB differs from source at {path}'
                        frame_hashes.append(sha(path)); raw_hashes.append(hashlib.sha256(raw).hexdigest());digest.update(raw)
                    assert process.stdout.read()==b''
                    assert process.wait()==0
                finally:
                    process.stdout.close()
                    if process.poll() is None: process.terminate();process.wait()
                    errors.seek(0); error=errors.read().decode()
                    if process.returncode: raise RuntimeError(error)
            assert len(frame_hashes)==64 and len(list(directory.glob('*.png')))==64
            record=dict(dataset=dataset,name=name,video_index=index,original_path=source,original_sha256=expected,
                original_width=1920,original_height=1080,original_fps=fps,original_format='MP4 source' if dataset=='ulong' else 'raw YUV420p 8-bit limited BT.709',
                original_probe=probe,canonical_width=1920,canonical_height=1080,canonical_fps=30.0,canonical_num_frames=64,
                canonical_format='RGB24 lossless PNG',canonical_dir=str(directory),selected_original_frame_indices=selected,
                canonical_frame_sha256=frame_hashes,canonical_frame_rgb_sha256=raw_hashes,whole_sequence_rgb_hash=digest.hexdigest(),
                whole_sequence_hash_representation='concatenated RGB24 bytes, row-major HWC, frame order',ffmpeg_command=command,
                source_to_png_bitexact=True)
            records.append(record)
            dump(f'parts/canonical_{dataset}_{index}.json',record)
            import torch
            torch.set_num_threads(4)
            frames=load_canonical_frames(directory)
            array_iterator=canonical_arrays(directory)
            for frame,array in zip(frames,array_iterator):
                assert frame.shape==(1,3,1080,1920) and frame.dtype==torch.float32
                assert 0<=frame.min().item()<=frame.max().item()<=1
                assert np.array_equal((frame[0].permute(1,2,0)*255).round().byte().numpy(),array)
            loader_records.append(dict(dataset=dataset,name=name,frames=64,shape=[1,3,1080,1920],dtype='float32',range=[0,1],RGB_order=True,roundtrip_bitexact=True))
            del frames
            print('CANONICAL PASS',dataset,name,'source_fps',fps,'frames',64,flush=True)
    assert len(records)==15
    dump('canonical_source_audit.json',dict(status='PASS',records=records,canonical_fps=30.0,canonical_num_frames=64,uvg_stride=4))
    dump('canonical_loader_audit.json',dict(status='PASS',same_loader_for_ulong_and_uvg=True,
        loader_path=str(ROOT/'canonical_loader.py'),loader_sha256=sha(ROOT/'canonical_loader.py'),function='load_canonical_frames',records=loader_records))
    dump('protocol_comparison.json',dict(OLD={'ulong':{'fps':'29.97002997 or 30','input':'MP4 decoded cached RGB PNG','frames':64},
        'uvg':{'fps':120,'input':'raw YUV420p to RGB using old conversion','frames':96}},
        NEW=dict(resolution=[1920,1080],fps=30.0,frames=64,format='RGB24 lossless PNG',same_loader=True,padding_bottom=8,
                 metric_crop=[1920,1080],same_metric_implementations=True,external_qps=list(range(10)),fresh_real_RANS=True,
                 uvg_stride=4,uvg_selected_indices=list(range(0,253,4)),U_Long_no_frame_resampling=True)))
    print('PREPARE PROTOCOL PASS',flush=True)

if __name__=='__main__': prepare()

