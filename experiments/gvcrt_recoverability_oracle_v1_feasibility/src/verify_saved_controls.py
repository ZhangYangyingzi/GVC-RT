"""Verify persisted control rules, ROI symbols and metric/rate aggregation on CPU."""
from base import *
import argparse
import math
import struct
import traceback


def verify(checkpoint=False):
    import numpy as np
    records, errors = [], []
    for path in sorted((ROOT / 'manifests/receipts').glob('*.json')):
        receipt = load(path)
        if receipt['status'] != 'PASS':
            continue
        job, result = receipt['job'], receipt['result']
        try:
            assert receipt['job_sha256'] == sha(ROOT / 'manifests/jobs' / (job['id'] + '.json'))
            raw = Path(result['bitstream_path']).read_bytes()
            assert raw[:8] == b'GROV1\0\0\0'
            count = struct.unpack('>H', raw[8:10])[0]
            decoded = []
            for index in range(count):
                f, y, x, h, w, level = struct.unpack('>5HB', raw[10 + 11*index:21 + 11*index])
                decoded.append(dict(frame=f, y=y, x=x, h=h, w=w, factor=(1., .5, 2.)[level]))
            assert decoded == job['plan'], 'Persisted control plan differs from job'
            roi_records = load(ROOT / 'oracle/local_sensitivity/symbols' / (job['id'] + '.json'))
            for frame, artifact in receipt['tensor_artifacts'].items():
                with np.load(artifact['path']) as tensors:
                    expected = tensors['original_q'].copy()
                    for decision in job['plan']:
                        if decision['frame'] == int(frame):
                            y, x, h, w = (decision[k] for k in ('y', 'x', 'h', 'w'))
                            roi = (slice(None), slice(None), slice(y, y+h), slice(x, x+w))
                            expected[roi] = np.maximum(expected[roi] * decision['factor'], .5)
                    assert np.array_equal(expected, tensors['effective_q']), (frame, 'step rule mismatch')
                    for record in roi_records:
                        decision = record['block']
                        if decision['frame'] != int(frame):
                            continue
                        y, x, h, w = (decision[k] for k in ('y', 'x', 'h', 'w'))
                        roi = (slice(None), slice(None), slice(y, y+h), slice(x, x+w))
                        assert tensors['symbols'][roi].flatten().tolist() == record['modified_symbols']
                        if job.get('baseline_id'):
                            baseline = load(ROOT / 'manifests/receipts' / (job['baseline_id'] + '.json'))
                            with np.load(baseline['tensor_artifacts'][frame]['path']) as original:
                                assert original['symbols'][roi].flatten().tolist() == record['original_symbols']
            with (ROOT / 'metrics/per_frame' / (job['id'] + '.csv')).open() as handle:
                frames = list(csv.DictReader(handle))
            assert len(frames) == result['frames'] == 16
            assert sum(int(row['real_bits']) for row in frames) == len(raw) * 8
            for metric in ('LPIPS', 'DISTS'):
                values = [float(row[metric]) for row in frames]
                assert all(math.isfinite(value) for value in values)
                assert math.isclose(sum(values) / len(values), result[metric], rel_tol=1e-12)
            assert result['FID'] is None and result['FID_status'] == 'NOT_COMPUTED_INSUFFICIENT_SAMPLES'
            if job['kind'] == 'repeat':
                assert Path(result['bitstream_path']).read_bytes() == Path(baseline['result']['bitstream_path']).read_bytes()
            records.append(dict(id=job['id'], status='PASS', signalled_controls=count,
                                quantization_rule_exact=True, saved_ROI_symbols_exact=True,
                                spatial_metric_aggregation_exact=True, frame_bit_sum_exact=True))
        except Exception:
            errors.append(dict(id=job['id'], traceback=traceback.format_exc()))
    report = dict(status='PASS' if not errors else 'FAILED', checkpoint_only=checkpoint,
                  validated_receipts=len(records), records=records, errors=errors, time=time.time())
    dump('manifests/control_symbol_validation' + ('_checkpoint' if checkpoint else '') + '.json', report)
    assert not errors, errors
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    print(json.dumps({k: v for k, v in verify(args.checkpoint).items() if k not in ('records', 'errors')}))
