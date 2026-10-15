"""Transfer only the dispatcher after live training finishes; never restart training."""
from preflight import *
from data import load

OLD_DRIVER = 2292186
TRAINERS = {'B_rgb_P': 2292202, 'C_latent_P': 2292213}

def process_state(pid):
    p = Path(f'/proc/{pid}/status')
    if not p.exists():
        return None
    return next(x.split()[1] for x in p.read_text().splitlines() if x.startswith('State:'))

def signal_owned(pid, signal, expected):
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert str(R / expected) in command, (pid, command)
    subprocess.run(['/bin/kill', signal, str(pid)], check=True)

def main():
    import fcntl
    lock = (R / 'driver_handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.time() + 14400
    frozen = {str(R / p): sha(R / p) for p in
              ('train.py', 'runtime.py', 'data.py', 'preflight.py',
               'protocol.json', 'plans/shared.json', 'calibration/final.json')}
    audit = dict(status='WAITING_FOR_900', old_driver=OLD_DRIVER, trainers=TRAINERS,
                 training_restarted=False, optimizer_or_RNG_reset=False,
                 training_signatures=frozen, started=time.time())
    save('audits/driver_handoff.json', audit)
    paused = False
    try:
        while True:
            assert time.time() < deadline, 'Bounded four-hour dispatcher handoff deadline'
            assert load(R / 'status.json').get('PID') == OLD_DRIVER
            assert process_state(OLD_DRIVER) not in (None, 'Z')
            assert load(R / 'early_original_status.json')['status'] != 'BLOCKED'
            updates = {b: load(R / f'branches/{b}/status.json')['updates'] for b in TRAINERS}
            for b, pid in TRAINERS.items():
                assert process_state(pid) not in (None, 'Z') or updates[b] == 1000, b
            if min(updates.values()) >= 900:
                break
            time.sleep(30)
        early = load(R / 'early_original_status.json')
        assert early['status'] == 'COMPLETE' and not early.get('running'), early
        signal_owned(OLD_DRIVER, '-STOP', 'run_pipeline.py')
        paused = True
        time.sleep(1)
        assert process_state(OLD_DRIVER) == 'T'
        audit.update(status='WAITING_FOR_LIVE_TRAINING', paused_at_updates=updates,
                     early_workers_drained=True, pause_time=time.time())
        save('audits/driver_handoff.json', audit)
        print(json.dumps(dict(event='DISPATCHER_PAUSED_ONLY', updates=updates)), flush=True)
        while True:
            assert time.time() < deadline, 'Bounded handoff deadline while trainers finish'
            ready = True
            for b, pid in TRAINERS.items():
                path = R / f'branches/{b}/integrity.json'
                if not path.exists():
                    assert process_state(pid) not in (None, 'Z'), f'Trainer exited before integrity: {b}'
                    ready = False
                elif process_state(pid) not in (None, 'Z'):
                    ready = False
            if ready:
                break
            time.sleep(30)
        indexes = {}
        for b in TRAINERS:
            integrity = load(R / f'branches/{b}/integrity.json')
            assert integrity['status'] == 'PASS' and integrity['updates'] == 1000
            assert integrity['frozen_unchanged'] and integrity['supervised_frames'] == 15000
            assert load(R / f'branches/{b}/status.json')['status'] == 'COMPLETE'
            assert len(list((R / f'branches/{b}/receipts').glob('*.json'))) == 1000
            index = load(R / f'branches/{b}/checkpoint_index.json')['1000']
            assert sha(index['path']) == index['sha256']
            assert sha(index['P_path']) == index['P_sha256']
            assert Path(index['P_path']) == R / f'final_P/{b}_1000.pt'
            indexes[b] = index
        assert all(sha(p) == h for p, h in frozen.items()), 'Training signature dependency changed'
        audit.update(status='TRAINING_COMPLETE_HANDOFF', completed_updates={b:1000 for b in TRAINERS},
                     final_indexes=indexes, training_signature_dependencies_unchanged=True)
        save('audits/driver_handoff.json', audit)
        signal_owned(OLD_DRIVER, '-TERM', 'run_pipeline.py')
        signal_owned(OLD_DRIVER, '-CONT', 'run_pipeline.py')
        paused = False
        reap_deadline = time.time() + 60
        while Path(f'/proc/{OLD_DRIVER}').exists():
            assert time.time() < reap_deadline, 'Old dispatcher did not exit/reap within 60 seconds'
            time.sleep(1)
        subprocess.run([sys.executable, '-B', str(R / 'launch.py')], cwd=GLC, check=True)
        new_deadline = time.time() + 60
        while True:
            status = load(R / 'status.json')
            if status.get('PID') not in (None, OLD_DRIVER):
                break
            assert time.time() < new_deadline, 'New dispatcher failed to write its PID'
            time.sleep(1)
        audit.update(status='PASS', new_driver=status['PID'], completed=time.time(),
                     new_evaluation_scheduler_sha256=sha(R/'evaluation_stage.py'),
                     numerical_training_or_codec_changes=False)
        save('audits/driver_handoff.json', audit)
        print(json.dumps(audit), flush=True)
    except Exception:
        if paused and process_state(OLD_DRIVER) == 'T':
            signal_owned(OLD_DRIVER, '-CONT', 'run_pipeline.py')
        audit.update(status='BLOCKED', error=traceback.format_exc(), failed=time.time())
        save('audits/driver_handoff.json', audit)
        raise

if __name__ == '__main__':
    main()
