"""V6.30 namespaced IO; does not shadow the historical codec common module."""
import csv, hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = ROOT.parent / 'gvcrt_neural_wrapper_v6_29_fid_audit_feature_covariance_control'
PYTHON = '/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
METHODS = ('original', 'B1_initial', 'A_continue_1000', 'B_covstats_1000')
DATASETS = ('hevc_b', 'mcljcv')
QPS = (0, 5, 9)
WEIGHT = Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')
WEIGHT_SHA = '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
REFERENCE = '62410e1dab4061666270d28d8b7cc3e0a04113bd'
OFFICIAL_COMMIT = 'b9c18118d082cbd263c1b8963fc4221dc1cbb659'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def load(p): return json.loads(Path(p).read_text())
def dump(p, value):
    p = Path(p); assert p.is_relative_to(ROOT)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); t.replace(p)
def textfile(p, text):
    p = Path(p); assert p.is_relative_to(ROOT); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.{time.time_ns()}.tmp'); t.write_text(text); t.replace(p)
def write(p, rows, fields=None):
    p = Path(p); assert p.is_relative_to(ROOT); p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    assert fields, 'empty table needs explicit schema'
    t = p.with_name(p.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    with t.open('w', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader()
        for r in rows: w.writerow({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v for k, v in r.items()})
    t.replace(p)
def sources(d, new=False): return load(ROOT / ('new_clips_manifest.json' if new else 'old_sources.json'))[d]
def task_id(t): return '_'.join(str(t[k]) for k in ('kind', 'dataset', 'video', 'method', 'qp'))
def output(t): return ROOT / 'tasks' / (task_id(t) + '.json')
def signature(t, code_override=None):
    d = dict(task=t, frozen=sha(ROOT/'source_hashes.json'), protocol=sha(ROOT/'protocol.json'),
             code=code_override if code_override is not None else {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))})
    if t['kind']=='new': d['manifest'] = sha(ROOT/'new_clips_manifest.json')
    if t['kind']=='closure':d['fresh_GT_sha256']=sha(ROOT/'features/GT'/t['dataset']/f'video_{t["video"]:02d}.npz')
    if t['kind']=='cpu':
        # Inputs and feature array contents are part of the statistical cache signature.
        inputs=[];spec=t['spec'];ids=spec['videos'];dset=t['dataset'];kind=spec['kind']
        methods=('A_continue_1000','B_covstats_1000') if kind=='subset' else METHODS
        for i in ids:
            for m in methods:
                for q in QPS:
                    p=ROOT/'features/closure'/dset/m/f'video_{i:02d}_qp{q}.npz'
                    inputs.append((str(p),sha(p)))
                    if kind in ('new64','combined_two_clips'):
                        p=ROOT/'features/new'/dset/m/f'video_{i:02d}_qp{q}.npz';inputs.append((str(p),sha(p)))
                    if kind=='closure_pool':
                        oldr=load(point(dset,m,i,q));inputs.append((oldr['feature_path'],sha(oldr['feature_path'])))
        d['feature_inputs']=inputs
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
def valid(t):
    p = output(t)
    if not p.exists(): return False
    try:
        r = load(p)
        return r['signature']==signature(t) and r['execution_complete'] and all(Path(p).is_file() and sha(p)==h for p,h in r['artifacts'].items())
    except (OSError, KeyError, ValueError): return False
def point(d,m,v,q): return OLD/'parts/ip_-1'/d/m/f'video_{v:02d}_qp{q}.json'
def append_error(value):
    value.update(time=time.time())
    with (ROOT/'errors.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
def compare_features(fresh, cached):
    import numpy as np
    assert fresh.shape == cached.shape == (64,2048)
    assert np.isfinite(fresh).all() and np.isfinite(cached).all()
    z = fresh.astype('float64') - cached.astype('float64')
    rel = np.linalg.norm(z,axis=1)/np.maximum(np.linalg.norm(cached.astype('float64'),axis=1),1e-12)
    ma = np.abs(z).max(axis=1); tol = 1e-4+1e-4*np.abs(cached).max(axis=1)
    return [dict(frame=i,relative_L2=float(rel[i]),max_abs=float(ma[i]),max_abs_tolerance=float(tol[i]),PASS=bool(rel[i]<=1e-4 and ma[i]<=tol[i])) for i in range(64)]
def save_npz(p, **values):
    import numpy as np
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(f'.{os.getpid()}.tmp')
    with t.open('wb') as f: np.savez(f,**values)
    t.replace(p)
