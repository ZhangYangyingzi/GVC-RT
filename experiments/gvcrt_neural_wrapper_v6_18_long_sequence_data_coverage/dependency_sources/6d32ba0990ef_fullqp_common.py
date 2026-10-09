"""Isolated full-QP revision; legacy beta sweep is never resumed here."""
from v61_io import *
V61 = ROOT.parent / 'gvcrt_neural_wrapper_v6_1_large_diverse_qp09'
if (V61 / 'runtime_dependencies').is_dir():
    sys.path.insert(0, str(V61 / 'runtime_dependencies'))
REVISION = 'fullqp_schedule_v1'
SEED = 20260929
BRANCHES = {'schedule_s0p5': 0.5, 'schedule_s1p0': 1.0,
            'schedule_s1p5': 1.5, 'schedule_s2p0': 2.0,
            'fixed_beta17p3028_aligned_qp': None}
STEPS = [0, 250, 500, 1000, 2000, 3000]
QPS = [0, 2, 4, 6, 8, 9]
def lambda_q(q):
    assert q in range(10)
    return 0.08 + q * (0.9 - 0.08) / 9
def beta_q(branch, q):
    return 17.302782275540444 if BRANCHES[branch] is None else BRANCHES[branch] / lambda_q(q)
def frame_qp(model, external_qp, frame_index):
    from gvc_hooks import INDEX_MAP
    return external_qp if frame_index == 0 else model.shift_qp(external_qp, INDEX_MAP[frame_index % len(INDEX_MAP)])
def cp_path(branch, step):
    return ROOT / 'branches' / branch / 'checkpoints' / f'step_{step:04d}.pt'
def model_hashes(payload):
    return {k: tensor_hash(payload[k]) for k in ('wrapper', 'bridge', 'generator')}
def check_gate():
    cfg = load(ROOT / 'fullqp_config.json')
    assert cfg['revision'] == REVISION
    for name in ('preflight_audit', 'validation_split_integrity', 'qp_train_eval_semantics_audit', 'gvc_lambda_schedule_audit'):
        assert load(ROOT / (name + '.json'))['status'] == 'PASS', name
    assert sha(cfg['train_manifest']) == cfg['train_manifest_sha256']
    for name, digest in cfg['audit_hashes'].items():
        assert sha(ROOT / name) == digest, ('changed preflight', name)
    return cfg
def assert_old_unchanged():
    for path, digest in load(ROOT / 'fullqp_old_source_hashes.json').items():
        assert sha(path) == digest, ('old artifact modified', path)
def source_signature():
    return {p.name: sha(p) for p in sorted(ROOT.glob('fullqp_*.py'))}
