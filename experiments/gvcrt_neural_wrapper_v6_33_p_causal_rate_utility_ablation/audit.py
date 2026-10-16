"""Read-only source and split audit. Run before any training."""
import hashlib
import json
import re
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V22 = ROOT.parent / 'gvcrt_neural_wrapper_v6_22_long_reference_reconstruction_loss_control'
V32 = ROOT.parent / 'gvcrt_neural_wrapper_v6_32b_native_frame_fd_coral_completion'

def load(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(p)

def identity(s):
    m = re.search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', s)
    if m:
        return m.group()
    m = re.search(r'videoSRC\d+', s, re.I)
    return m.group().lower() if m else Path(s).stem.split('_1920')[0].lower()

def main():
    split = load(V22 / 'dataset_split.json')
    ev = load(V32 / 'eval_manifest.json')
    train = split['training']
    validation = {'uvg_validation': [v['name'] for v in ev['uvg_validation']]}
    testing = {k: [v['name'] for v in vs] for k, vs in ev.items() if k not in validation}
    train_ids = {identity(v) for vs in train.values() for v in vs}
    eval_ids = {identity(v) for vs in (*validation.values(), *testing.values()) for v in vs}
    historical = load(V32 / 'data_isolation_audit.json')
    test_sources = historical['test_sources']
    checks = {}
    manifests = []
    for domain in ('ulong', 'uvg'):
        for v in load(V32 / f'train_manifest_{domain}.json')['videos']:
            manifests.append(dict(domain=domain, identity=v.get('filename', v.get('name')),
                                  source_path=v.get('path', v.get('source_path')),
                                  source_sha256=v.get('sha256', v.get('source_sha256'))))
    test_paths = {str(Path(v['source_path']).resolve()) for v in test_sources}
    test_hashes = {v['source_sha256'] for v in test_sources}
    for folder in (V22, V32):
        rows = load(folder / 'plans/shared.json')['updates']
        sources = {r['source_path']: r for r in rows}
        checks[folder.name] = dict(
            plan_sha256=sha(folder / 'plans/shared.json'), updates=len(rows),
            identity_overlap=sorted({identity(r['video']) for r in rows} & eval_ids),
            outside_inherited_training=sorted({r['video'] for r in rows if identity(r['video']) not in train_ids}),
            source_path_overlap=sorted({str(Path(p).resolve()) for p in sources} & test_paths),
            source_sha256_overlap=sorted({r['source_sha256'] for r in rows} & test_hashes),
            sources=[dict(video=r['video'], domain=r['domain'], source_path=p,
                          source_sha256=r['source_sha256'], exists=Path(p).is_file())
                     for p, r in sorted(sources.items())])
    overlap = sorted(train_ids & eval_ids)
    clean = {d: [s for s in vs if identity(s) not in eval_ids] for d, vs in train.items()}
    candidate_path_overlap = sorted({str(Path(v['source_path']).resolve()) for v in manifests} & test_paths)
    candidate_hash_overlap = sorted({v['source_sha256'] for v in manifests} & test_hashes)
    passed = not (overlap or candidate_path_overlap or candidate_hash_overlap) and all(
        not any(c[k] for k in ('identity_overlap', 'outside_inherited_training', 'source_path_overlap', 'source_sha256_overlap'))
        for c in checks.values())
    result = dict(status='PASS' if passed else 'HISTORICAL_OVERLAP',
        scope='Identity/path overlap and recorded source SHA256 overlap; source media not all rehashed in this audit',
        training_identities=train, validation_identities=validation, test_identities=testing,
        normalized_identity_overlap=overlap, training_candidate_sources=manifests,
        candidate_source_path_overlap=candidate_path_overlap, candidate_source_sha256_overlap=candidate_hash_overlap,
        test_sources=test_sources, plan_checks=checks, clean_training_identities=clean,
        HEVC_B_test_only=True, MCL_JCV_test_only=True,
        checkpoint_selection='fixed step1000; no evaluation-based selection',
        historical_limitations=[
            'Inherited cleaned U-Long split; historical source_split metadata preserved separately.',
            'Prior audits report MCL source17 historical exposure. MCL29 must remain a sensitivity subset.',
            'Unknown original codec/receiver pretraining exposure is not ruled out.',
            'No new perceptual near-duplicate search was performed.'],
        historical_source_split=split['source_split'],
        source_files={str(p): sha(p) for p in [V22/'dataset_split.json', V32/'eval_manifest.json', V32/'data_isolation_audit.json']})
    dump('data_split_audit.json', result)
    text = '# Data split audit\n\nStatus: ' + result['status'] + '\n\n'
    for title, groups in [('Training', train), ('Validation (evaluation only)', validation), ('Test', testing)]:
        text += '## ' + title + '\n\n'
        for domain, ids in groups.items():
            text += f'### {domain} ({len(ids)})\n\n' + '\n'.join('- ' + i for i in ids) + '\n\n'
    text += '## Limits\n\n' + '\n'.join('- ' + s for s in result['historical_limitations']) + '\n\n'
    text += 'All overlap arrays and both historical training plan source inventories are in data_split_audit.json. No test-based checkpoint or hyperparameter selection is allowed. HEVC-B and MCL-JCV remain test-only.\n'
    (ROOT / 'DATA_SPLIT_AUDIT.md').write_text(text)
    cp = V22/'branches/B1_reconstruction/checkpoints/adaptation_step_1000.pt'
    expected = 'dc76f0e221dc7aaa5e5f45f648722584ce226e65828ce24151f9961985f3f6f7'
    actual = sha(cp)
    dump('checkpoint_hashes.json', dict(source=dict(path=str(cp), expected=expected, sha256=actual, verified=actual==expected)))
    dump('source_provenance.json', {str(p):sha(p) for folder, names in [
        (V22, ['config.json','dataset_split.json','adapter.py','evaluate.py','train.py','window_training.py']),
        (V32, ['config.json','evaluation_config.json','train.py','training32.py','adapter32.py','evaluate32.py','REPORT.md']),
        (ROOT.parent/'gvcrt_neural_wrapper_v6_31a_latent_fidelity_vs_generative_utility_audit', ['REPORT.md','conclusion.json','all_rd_points.csv'])]
        for n in names if (p:=folder/n).is_file()})
    print(json.dumps(dict(status=result['status'], normalized_overlap=overlap, checkpoint_verified=actual==expected)))

if __name__ == '__main__':
    main()
