"""Read-only progress validation; never starts model work."""
import json
import math
import time
import numpy as np
from audit_utils import *
from evaluate import valid_point
from stream_audit import structure

def main():
    rows=[];counts={d:{m:0 for m in METHODS} for d in ('ulong','uvg')}
    for dataset in counts:
        for method in METHODS:
            for path in sorted((ROOT/'parts'/dataset/method).glob('video_*_qp*.json')):
                assert valid_point(path)
                r=load(path)
                assert r['method']==method and r['dataset']==dataset
                assert r['checkpoint_sha256']==load('config.json')['checkpoints'][method]['sha256']
                assert all(math.isfinite(float(r[k])) for k in NUMERIC)
                assert math.isclose(r['kbps'],r['bits_per_frame']*30/1000,rel_tol=1e-12)
                assert math.isclose(r['bpp'],r['bits_per_frame']/(1920*1080),rel_tol=1e-12)
                header=structure(Path(r['bitstream_path']),r['external_qp'],64)
                assert sum(f['real_bits'] for f in header)==r['real_bytes']*8
                frame=read(r['frame_metrics_path']);assert len(frame)==64
                for f,h in zip(frame,header):
                    assert int(f['actual_qp'])==h['actual_qp'] and int(f['external_qp'])==r['external_qp']
                transitions=read(Path(r['feature_path']).with_suffix('.transitions.csv'))
                assert [(int(t['from_frame']),int(t['to_frame'])) for t in transitions]==[(i,i+1) for i in range(63)]
                assert all(math.isfinite(float(t['FloLPIPS'])) for t in transitions)
                assert math.isclose(sum(float(t['FloLPIPS']) for t in transitions)/63,r['FloLPIPS'],abs_tol=1e-10)
                with np.load(r['feature_path']) as feature:
                    assert feature['real'].shape==feature['reconstruction'].shape==(64,2048)
                    assert np.isfinite(feature['real']).all() and np.isfinite(feature['reconstruction']).all()
                counts[dataset][method]+=1;rows.append(r)
    result=dict(completed_points=len(rows),expected_points=450,counts=counts,
                completed_point_checks='PASS' if rows else 'NO_COMPLETE_POINTS',pipeline=load('pipeline_status.json'),
                mean_seconds_per_completed_point=float(np.mean([r['elapsed_seconds'] for r in rows])) if rows else None)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()

