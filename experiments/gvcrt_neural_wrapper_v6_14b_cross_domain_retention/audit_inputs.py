"""Read-only historical input audit; does not load models or launch evaluation."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BASE = ROOT.parent / 'gvcrt_neural_wrapper_v6_2b_final_generalization_test'
HEVC = BASE / 'results/hevc_b'
V614 = ROOT.parent / 'gvcrt_neural_wrapper_v6_14_uvg_target_domain_finetune'
REQUESTED_SHA256 = 'fe63a6f00ca6bae59b23c8683146dbe67c1463a552af851bbfc31e9cba'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def audit_video(v):
    import numpy as np
    from PIL import Image
    out = {k: v[k] for k in ('dataset', 'video_index', 'name', 'frames', 'width', 'height', 'rate_accounting_fps', 'source_path', 'source_frame_indices')}
    try:
        assert sha(v['source_path']) == v['source_sha256'], 'source file hash mismatch'
        h = hashlib.sha256()
        is_hevc = v['dataset'] == 'hevc_b'
        files = v['frame_png_sha256'] if is_hevc else v['frame_file_sha256']
        assert len(files) == len(v['frame_rgb_sha256']) == v['frames'] == 64
        for i in range(v['frames']):
            p = Path(v['input_dir']) / (f'im{i+1:05d}.png' if is_hevc else f'frame_{i:06d}.png')
            assert sha(p) == files[i], f'PNG hash mismatch: {p}'
            with Image.open(p) as im:
                assert im.mode == 'RGB' and im.size == (v['width'], v['height'])
                raw = np.asarray(im, dtype=np.uint8).tobytes()
            assert hashlib.sha256(raw).hexdigest() == v['frame_rgb_sha256'][i], f'RGB hash mismatch: {p}'
            h.update(raw)
        assert h.hexdigest() == v['rgb_sha256']
        out.update(status='PASS', source_sha256=v['source_sha256'], rgb_sha256=h.hexdigest(), frames_verified=64)
    except Exception as error:
        out.update(status='FAIL', error=repr(error))
    print(out['dataset'], out['name'], out['status'], flush=True)
    return out

def main():
    save('logs/input_audit_command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), started_unix=time.time()))
    old = load(BASE / 'source_manifest.json')
    uv = [v for v in old['videos'] if v['dataset'] == 'ulong']
    hm = load(HEVC / 'source_manifest.json')
    hv = hm['videos']
    assert len(uv) == 8 and len(hv) == 5
    save('manifests/ulong.json', dict(videos=uv, origin=str(BASE / 'source_manifest.json'), origin_sha256=sha(BASE / 'source_manifest.json')))
    save('manifests/hevc_b.json', dict(**hm, origin=str(HEVC / 'source_manifest.json'), origin_sha256=sha(HEVC / 'source_manifest.json')))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(audit_video, uv + hv))
    save('audits/dataset_audit.json', dict(status='PASS' if all(v['status']=='PASS' for v in results) else 'FAIL', videos=results))
    cp = load(BASE / 'config.json')['checkpoints']['v62']
    actual = sha(cp['path'])
    frozen = load(V614 / 'frozen_checkpoint_index.json')
    adapted = {}
    for step in ('1000', '5000'):
        r = frozen['checkpoints'][step]
        adapted[step] = dict(source_step=r['source_step'], adaptation_step=r['adaptation_step'], training_path=r['path'], training_sha256=sha(r['path']), inference_path=r['inference']['path'], inference_sha256=sha(r['inference']['path']))
        adapted[step]['file_hashes_match'] = adapted[step]['training_sha256']==r['sha256'] and adapted[step]['inference_sha256']==r['inference']['sha256']
    record = dict(status='BLOCKED', reason='Requested v62_initial SHA256 has 58 characters; correction requires confirmation.', requested_sha256=REQUESTED_SHA256, requested_length=len(REQUESTED_SHA256), actual_sha256=actual, checkpoint_path=cp['path'], matches_historical_v62b=actual==cp['sha256'], frozen_index_path=str(V614 / 'frozen_checkpoint_index.json'), frozen_index_sha256=sha(V614 / 'frozen_checkpoint_index.json'), frozen_index_matches_current_index=frozen['checkpoint_index_sha256']==sha(V614 / 'checkpoint_index.json'), adapted_file_audit=adapted, model_parameters_loaded=False, model_parameter_hash_checks='pending', evaluation_started=False)
    save('audits/checkpoint_preflight.json', record)
    save('input_audit_status.json', dict(status='BLOCKED', expected_RD=520, completed_RD=0, expected_videos=13, verified_videos=sum(v['status']=='PASS' for v in results), errors=[record['reason']], GPU_workers_started=0))
    print('INPUT AUDIT COMPLETE; model evaluation awaiting hash confirmation', flush=True)

if __name__ == '__main__':
    main()
