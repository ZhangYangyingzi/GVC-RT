"""Create task-local output parents, then delegate to the frozen runner unchanged."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORIES = (
    'audits/rans_roundtrip_probe',
    'audits/rans_roundtrip_probe/logs',
    'large_artifacts/engineering_rans_probe',
    'large_artifacts/engineering_rans_probe/bitstreams',
    'evaluation/development',
    'evaluation/development/logs',
    'evaluation/final',
    'evaluation/final/logs',
    'large_artifacts/development/bitstreams',
    'large_artifacts/final/bitstreams',
    'large_artifacts/viewing_previews',
)

def main():
    args = sys.argv[1:]
    prepare_only = args == ['--prepare-only']
    if '--prepare-only' in args and not prepare_only:
        raise ValueError('--prepare-only must be used alone; other arguments are forwarded to runner.py')
    created = []
    for relative in OUTPUT_DIRECTORIES:
        destination = ROOT / relative
        if not destination.resolve().is_relative_to(ROOT):
            raise RuntimeError('Output path escapes this experiment: ' + relative)
        existed = destination.exists()
        destination.mkdir(parents=True, exist_ok=True)
        if not destination.resolve().is_relative_to(ROOT):
            raise RuntimeError('Resolved output path escapes this experiment: ' + relative)
        if not existed:
            created.append(relative)
    source = Path(__file__).read_bytes()
    record = dict(
        status='DIRECTORIES_READY',
        scope='Directory initialization only; no numerical, training, or evaluation acceptance',
        recorded_unix=time.time(),
        reason='Historical evaluate.event uses logs.mkdir(exist_ok=True) without parents=True',
        launcher_filename=Path(__file__).name,
        launcher_sha256=hashlib.sha256(source).hexdigest(),
        launcher_content=source.decode(),
        output_directories=list(OUTPUT_DIRECTORIES),
        created=created,
        frozen_source_or_protocol_changed=False,
        invocation=[sys.executable, *sys.argv],
        mode='prepare_only_track_existing_runner' if prepare_only else 'exec_existing_frozen_runner',
        forwarded_arguments=[] if prepare_only else args,
    )
    audit = ROOT / 'audits' / ('launcher_initialization_' + str(time.time_ns()) + '.json')
    audit.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=record['status'],repair_audit=str(audit),mode=record['mode'],created=created)), flush=True)
    if not prepare_only:
        os.execv(sys.executable, [sys.executable, '-B', '-u', str(ROOT / 'runner.py'), *args])

if __name__ == '__main__':
    main()
