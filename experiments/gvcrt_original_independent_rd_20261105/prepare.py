"""Freeze original source, original weights, and the same 45 evaluation windows."""
from common import *
import argparse
import concurrent.futures
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-frames', action='store_true')
    args = ap.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ORIGINAL,text=True).strip() == COMMIT
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ORIGINAL,text=True).strip()
    previous = list(csv.DictReader((TCG/'metrics_v3.csv').open(newline='')))
    reference = {(r['split'],r['domain'],r['sequence']):r for r in previous if r['model_state']=='parent0' and r['QP']=='0'}
    videos = []
    historical = load(PARENT/'evaluation/manifest.local.json')
    for domain, seqs in historical.items():
        for v in seqs:
            record = dict(domain=domain, sequence=v['name'], split='historical_reference', width=v['width'], height=v['height'],
                          source_frame_indices=v['source_frame_indices'], fps=v['rate_accounting_fps'],
                          source_path=v['source_path'], source_sha256=v['source_sha256'], rgb_sha256=v['rgb_sha256'], frame_rgb_sha256=v['frame_rgb_sha256'])
            if domain != 'ulong':
                record.update(input_dir=v['input_dir'], frame_file_sha256=v['frame_png_sha256' if domain=='hevc_b' else 'frame_file_sha256'])
            videos.append(record)
    for r in load(TCG/'manifests/evaluation_rows.json'):
        if r['split'] != 'new_holdout_test':
            continue
        a,b = map(int, r['source_fps'].split('/'))
        videos.append(dict(domain='ulong_new',sequence=Path(r['video']).stem,split=r['split'],width=r['width'],height=r['height'],
                           source_frame_indices=r['canonical_indices'],fps=a/b,source_path=r['source_path'],source_sha256=r['source_sha256'],
                           rgb_sha256=r['rgb_sha256'],frame_rgb_sha256=r['full_RGB_sha256']))
    videos.sort(key=lambda v:(v['split'],v['domain'],v['sequence']))
    assert len(videos)==45
    for v in videos:
        old=reference[(v['split'],v['domain'],v['sequence'])]
        assert old['source_RGB_sha256']==v['rgb_sha256'] and old['source_sha256']==v['source_sha256']
        assert len(v['source_frame_indices'])==len(v['frame_rgb_sha256'])==64
        assert Path(v['source_path']).is_file()
    dump(ROOT/'videos.json',videos)
    files = subprocess.check_output(['git','ls-tree','-r','--name-only',COMMIT,'src'],cwd=ORIGINAL,text=True).splitlines()
    metric_sources = [TCG_REPO/'experiments/gvcrt_neural_wrapper_v2_joint/core.py',
                      TCG_REPO/'experiments/gvcrt_neural_wrapper_v4_progressive_joint/eval_core.py',
                      FLO/'flolpips_compat.py',FLO/'third_party/flolpips/flolpips.py',FLO/'third_party/flolpips/pwcnet.py']
    protocol = dict(original_commit=COMMIT, original_source_sha256={f:sha(ORIGINAL/f) for f in files},
                    checkpoint_sha256={f:sha(PROJECT/'checkpoints'/f) for f in ('GVC-RT_I.pt','GVC-RT_P.pt')},
                    metric_sources_sha256={str(f):sha(f) for f in metric_sources},previous_metrics_sha256=sha(TCG/'metrics_v3.csv'),
                    videos_sha256=sha(ROOT/'videos.json'),video_count=45,frames_per_video=64,QP=[0,5,9],expected_points=135,
                    model_state='GVC_RT_original',codec_precision='FP16',metric_precision='FP32',physical_gpus=[4,5,6,7],
                    force_zero_thres=.12,padding='replicate to multiples of 64',SPS_ec_part=1,SPS_use_ada_i=0,
                    GOP='one I then 63 P, same as existing TCG evaluation',QP_pattern=list(QP_PATTERN),
                    original_codec_modules_only=True,TCG_modules_present=False,adapted_checkpoint_loaded=False,
                    fresh_encode=True,reuse_old_bitstreams=False,independent_decoder_instances=True,
                    metric_implementations='Exact spatial metric functions extracted from pinned prior scoring source; same PWC/FloLPIPS compatibility implementation.',
                    scope='Independent original codec implementation under the matched experiment protocol; not a reproduction of every default in the paper evaluator.',
                    full_benchmark_dataset_claim=False)
    if (ROOT/'protocol.json').exists():
        assert load(ROOT/'protocol.json')==protocol,'Existing protocol changed'
    dump(ROOT/'protocol.json',protocol)
    if args.verify_frames:
        from data import verify_video
        records=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for record in executor.map(verify_video,videos):
                records.append(record)
                print(f"SOURCE {len(records)}/45 {record['domain']} {record['sequence']}",flush=True)
        dump(ROOT/'source_frame_audit.json',dict(status='PASS',videos=45,frames=2880,records=records,protocol_sha256=sha(ROOT/'protocol.json')))
    print(json.dumps(dict(status='PREPARED',expected_points=135,original_commit=COMMIT)))


if __name__=='__main__':
    main()
