"""Reuse historical codec, STE, frame replay and loss without codec edits."""
from audit import *
import importlib.util
import sys
import inspect
import random

PYTHON = '/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
BRANCHES = ('A_noP_frozenReceiver', 'B_noP_trainReceiver', 'C_P_train_frozenReceiver', 'D_P_train_trainReceiver')

def module(name, path):
    sys.path.insert(0, str(Path(path).parent))
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

def setup(branch):
    import torch
    import numpy as np
    cfg = load(ROOT/'protocol.json')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    sys.path.insert(0, str(ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
    v4 = module('causal_v4', ROOT.parent/'gvcrt_neural_wrapper_v4_progressive_joint/train.py')
    torch.set_num_threads(2)
    im, pm = v4.load_models(torch.device('cuda:0'), force_zero_thres=.12)
    im.requires_grad_(False); pm.requires_grad_(False)
    wrapper = v4.NeuralWrapper().cuda().float() if branch in BRANCHES[2:] else torch.nn.Identity().cuda()
    train_receiver = branch in (BRANCHES[1], BRANCHES[3])
    models = dict(wrapper=wrapper, bridge=pm.recon_generation_net.mlp.float(), generator=pm.recon_generation_net.decoder.float())
    models['wrapper'].train(branch in BRANCHES[2:]).requires_grad_(branch in BRANCHES[2:])
    for k in ('bridge','generator'):
        models[k].train(train_receiver).requires_grad_(train_receiver)
    quality = v4.quality_models(torch.device('cuda:0'))
    return v4, im, pm, models, quality

def runner(ctx):
    old = module('causal_window22', V22/'window_training.py')
    source = inspect.getsource(old.make_runner)
    needle = 'params=[p for m in models.values() for p in m.parameters()]'
    assert source.count(needle)==1
    source = source.replace(needle, 'params=[p for m in models.values() for p in m.parameters() if p.requires_grad]')
    ns = dict(old.__dict__)
    # No loss changes: only exclude frozen parameters from autograd targets.
    exec(compile(source, str(V22/'window_training.py'), 'exec'), ns)
    return ns['make_runner'](*ctx)

def source_frames(row, device='cuda:0'):
    old = module('causal_frames32', V32/'frames32.py')
    old.ROOT = V32
    return old.PlanFrames(row, device)

def frames(row,device='cuda:0'):
    folder=ROOT/'cache/training_crops';key=f'{row["adaptation_step"]:04d}'
    p=folder/(key+'.npy');receipt=folder/(key+'.json')
    if not p.exists() or not receipt.exists():return source_frames(row,device)
    import numpy as np
    import torch
    record=load(receipt)
    row_sha=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
    assert record['row_sha256']==row_sha and record['source_full_RGB_hashes_verified']
    array=np.load(p,mmap_mode='r');assert array.shape==(64,256,256,3) and array.dtype==np.uint8
    class CachedFrames:
        cache_used=True
        def __len__(self):return 64
        def __getitem__(self,pos):
            if pos>=64:raise IndexError(pos)
            a=array[pos].copy()
            assert hashlib.sha256(a.tobytes()).hexdigest()==row['cropped_RGB_sha256'][pos]
            return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(device).float()/255
        def close(self):array._mmap.close()
    return CachedFrames()

def hashes(ctx):
    v4, im, pm, models, quality = ctx
    return dict(compression=v4.compression_hash(im,pm), I=v4.module_hash(im),
                modules={k:v4.module_hash(m) for k,m in models.items()}, quality=[v4.module_hash(m) for m in quality])

def atomic_save(p, obj):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');torch.save(obj,t);t.replace(p)
