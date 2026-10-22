"""Isolated v6.44 read-only inputs and task receipts."""
from io32 import *

SOURCE43 = ROOT.parent / 'gvcrt_neural_wrapper_v6_43_frame_qp_budget_utility'
CP = ROOT.parent / 'gvcrt_neural_wrapper_v6_40_generator_loss_balanced_trend_100/evaluation/checkpoints/G_basic_100.pt'
CP_SHA = '89c1a5baa195c8ffc9695af495d43389e00021b926eae13a0337d3b2eb117fd4'

def signature44():
    names = ['protocol.json', 'data_manifest.json', 'baseline_manifest.json',
             'model_manifest.json', 'evaluation_config.json', 'intervention_plan.json',
             'evaluate44.py', 'common44.py', 'adapter35.py', 'io32.py', 'io35.py',
             'training35.py', 'source_manifest.json']
    return hashlib.sha256(json.dumps({n: sha(ROOT/n) for n in names}, sort_keys=True).encode()).hexdigest()

def key(t):
    return f"{t['dataset']}/{t['sequence']}/{t['variant']}"

def result_path(t):
    return ROOT / 'tasks' / key(t) / 'result.json'

def source_baseline():
    p = SOURCE43 / 'tasks/hevc_b/ParkScene/baseline/result.json'
    x = load(p)
    assert x['status'] == 'PASS'
    for f in x['artifacts']:
        q = Path(f['path'])
        assert q.stat().st_size == f['bytes'] and sha(q) == f['sha256'], q
    assert x['result']['total_bits'] == 977264
    assert x['result']['LPIPS'] == 0.2153701321221888
    return x

def validate_result(t):
    p = result_path(t)
    if not p.exists():
        return False
    r = load(p)
    assert r['signature'] == signature44() and r['task'] == t
    for f in r['artifacts']:
        q = Path(f['path'])
        assert q.stat().st_size == f['bytes'] and sha(q) == f['sha256'], q
    assert r['status'] == 'PASS'
    return True
