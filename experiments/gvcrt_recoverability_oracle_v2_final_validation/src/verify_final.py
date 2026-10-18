"""Independent postflight checks of persisted rates, policy isolation and exits."""
from protocol import *
import random


def verify():
    manifest = load(ROOT / 'manifests/experiment_manifest.json')
    assert sha(ROOT / 'configs/experiment.json') == manifest['config_sha256']
    assert sha(ROOT / 'manifests/videos.json') == manifest['videos_sha256']
    videos = load(ROOT / 'manifests/videos.json')
    for split in videos.values():
        assert split['calibration']['input_dir'] != split['test']['input_dir']
        assert not (set(split['calibration']['frame_rgb_sha256'][:16]) &
                    set(split['test']['frame_rgb_sha256'][:16]))

    frozen_path = ROOT / 'oracle/rate_allocation/frozen_policy.json'
    frozen = load(frozen_path)
    assert frozen['registered_config_sha256'] == manifest['config_sha256']
    integrity = load(ROOT / 'manifests/test_policy_integrity.json')
    assert integrity['frozen_policy_sha256'] == sha(frozen_path)
    assert integrity['test_quality_used_for_selection'] is False
    commands = [load(p) for p in (ROOT / 'manifests/commands').glob('*.json')]
    jobs = {p.stem: load(p) for p in (ROOT / 'manifests/jobs').glob('*.json')}
    for c in commands:
        job_path = Path(c['command'][c['command'].index('--job') + 1])
        if jobs[job_path.stem]['phase'] == 'test':
            assert c['start'] >= frozen['time']

    actual_candidates = load(ROOT / 'oracle/rate_allocation/calibration_actual_candidates.json')
    for c in actual_candidates:
        assert c['plan']
        r = load(c['receipt_path'])
        assert Path(r['result']['bitstream_path']).stat().st_size * 8 == c['bits']
        assert c['actual_rate_matched'] == matched(c['bits'], c['target_bits'])
    for p in frozen['policies']:
        pool = [c for c in actual_candidates if c['point'] == p['point'] and c['actual_rate_matched']]
        exact = [c for c in pool if c['exact_rate']]
        if exact:
            pool = exact
        if p['status'] == 'no_feasible_candidate':
            assert not pool
            continue
        assert p['status'] == 'FROZEN' and p['frozen_plan']
        assert p['independent_test_metrics_used'] is False
        assert len(pool) == p['pool_size']
        assert p['calibration_choice'] in pool
        if p['strategy'] == 'D_oracle':
            expected = min(pool, key=lambda c: (c['LPIPS'], abs(c['bits'] - p['target_bits']), c['encoding_key']))
        elif p['strategy'] == 'B_random':
            rng = random.Random(p['seed'] + p['base_qp'] + sum(map(ord, p['dataset'])))
            expected = rng.choice(pool)
        else:
            expected = max(pool, key=lambda c: (c['heuristic_score'], -abs(c['bits'] - p['target_bits']), c['encoding_key']))
        assert p['calibration_choice'] == expected

    with (ROOT / 'metrics/per_video_results.csv').open() as f:
        rows = list(csv.DictReader(f))
    trial_path = ROOT / 'oracle/rate_allocation/all_independent_trials.csv'
    if trial_path.exists():
        with trial_path.open() as f:
            trials = list(csv.DictReader(f))
    else:
        trials = []
        assert not any(p['status'] == 'FROZEN' for p in frozen['policies'])
    native_rows = {(r['phase'], r['dataset'], r['base_QP']): r for r in rows
                   if r['strategy'] == 'A_native_baseline'}
    checks = []
    for r in rows:
        if not r.get('actual_bits'):
            assert r['status'] in ('no_feasible_candidate', 'MISSING_NATIVE_BASELINE',
                                   'NO_SUCCESSFUL_ACTUAL_ENCODING')
            continue
        path = Path(r['stream_path'])
        bits, target = int(r['actual_bits']), int(r['target_bits'])
        native = native_rows[r['phase'], r['dataset'], r['base_QP']]
        assert target == int(native['actual_bits'])
        assert path.stat().st_size * 8 == bits
        assert float(r['bpp']) == bits / (16 * 1920 * 1080)
        assert math.isclose(float(r['kbps']), bits * 30 / 16000, rel_tol=1e-14)
        assert (r['actual_rate_matched'] == 'True') == matched(bits, target)
        assert (r['exact_rate_match'] == 'True') == (bits == target)
        assert bits == sum(int(r[k]) for k in
                           ('control_header_bits', 'native_headers_bits', 'RANS_payload_bits'))
        if r['strategy'] != 'A_native_baseline':
            assert json.loads(r['plan'])
            if r['phase'] == 'test':
                for metric in ('LPIPS', 'DISTS', 'FloLPIPS'):
                    assert math.isclose(float(r['delta_' + metric.lower()]),
                                        float(r[metric]) - float(native[metric]), abs_tol=1e-15)
                group = [t for t in trials if all(t[k] == r[k] for k in
                         ('dataset', 'base_QP', 'strategy', 'seed'))]
                feasible = [t for t in group if matched(int(t['actual_bits']), target)]
                pool = feasible or group
                selected = min(pool, key=lambda t: (abs(int(t['actual_bits']) - target),
                               abs(int(t['encoding_QP']) - int(r['base_QP'])), int(t['encoding_QP'])))
                assert selected['receipt_path'] == r['receipt_path']
        else:
            assert r['encoding_QP'] == r['base_QP'] and r['control_header_bits'] == '0'
        checks.append(dict(phase=r['phase'], dataset=r['dataset'], base_QP=r['base_QP'],
                           strategy=r['strategy'], seed=r['seed'], bits=bits,
                           matched=matched(bits, target)))

    for d in videos:
        for q in cfg()['QPs']:
            seeds = [int(r['seed']) for r in rows if r['phase'] == 'test' and
                     r['dataset'] == d and int(r['base_QP']) == q and r['strategy'] == 'B_random']
            assert sorted(seeds) == sorted(cfg()['random_seeds'])
    exits = load(ROOT / 'logs/GPU_task_exit_status.json')
    assert exits['GPU_tasks_running'] == 0
    assert load(ROOT / 'logs/driver_exit.json')['exit_code'] == 0
    assert load(ROOT / 'logs/finalize_exit.json')['exit_code'] == 0
    assert exits['all_final_worker_exit_codes_zero']
    assert load(ROOT / 'manifests/artifact_validation.json')['status'] == 'PASS'
    report = dict(status='PASS', verified=time.time(), config_sha256=manifest['config_sha256'],
                  frozen_policy_sha256=sha(frozen_path), independent_video_hashes_disjoint=True,
                  test_encoding_after_policy_freeze=True, all_random_seeds_retained=True,
                  actual_rate_and_control_bytes_verified=True, test_QP_selection_rate_only=True,
                  records=checks, worker_exit_codes=exits['final_workers'])
    dump('manifests/independent_postflight_verification.json', report)
    print('PASS', len(checks), 'measured strategy rows; all registered random seeds accounted')


if __name__ == '__main__':
    verify()
