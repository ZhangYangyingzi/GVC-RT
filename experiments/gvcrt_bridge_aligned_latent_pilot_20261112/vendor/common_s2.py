"""Stage2-local paths, immutable identities and atomic execution records."""
from pathlib import Path
import contextlib,csv,fcntl,gzip,hashlib,importlib.util,json,math,os,random,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
WORKTREE=ROOT.parents[1]
PROJECT=Path('/Huang_group/zyyz/Projects/GVC-RT')
SOURCE=PROJECT/'GVC-RT_worktrees/tcg_stage2_convlora_coverage_ab_20261107/experiments/gvcrt_tcg_stage2_convlora_coverage_ab_20261107'
GAIN_SOURCE=SOURCE.parent/'gvcrt_tcg_stage2_convlora_A2000_qp05_gain_20261108'
INIT=SOURCE/'A_control/checkpoints/step_2000.pt'
INIT_SHA='8a69765befa6ed267d0a95d3932cd63d9513c1da9818fd2126d8bc1c165f5e4e'
REVIEW=PROJECT/'experiments/gvcrt_ulong_vimeo_mcl_data_review_20261108'
PLAN_SEED=20261109
ORIGINAL=PROJECT/'GVC-RT_worktrees/gvcrt_original_independent_20261105'
ORIGINAL_COMMIT='b2a0895c0cf31cf2be3ab1c2571ca07c05866c03'
FLO=Path('/Huang_group/zyyz/Projects/cosmos-predict1/upsample_LUVE/stage28_exact_stage23_fallback')
TCG_REPO=PROJECT/'GVC-RT_worktrees/tcg_stage1_data_expansion_ab_groupisolated_v2'
HIST=PROJECT/'GVC-RT_worktrees/tcg_stage1_data_expansion_ab_groupisolated_v2/experiments/gvcrt_tcg_stage1_data_expansion_ab_groupisolated_v2'
ORIGINAL_EVAL=PROJECT/'experiments/gvcrt_original_independent_rd_20261105'
PARENT=Path('/Huang_group/zyyz/home_dir/tmp/gvcrt_tcg_stage1_expanded_20261103_worktree/experiments/gvcrt_tcg_stage1_expanded_5000')
ARCHIVES=Path('/Huang_group/zyyz/datasets/UltraVideo-Long/clips_long_1920')
PYTHON='/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python'
BASELINE=HIST/'B_expanded_ulong/checkpoints_v3/step_1500.pt'
BASELINE_SHA=json.loads((SOURCE/'protocol.json').read_text())['baseline_sha256']
PILOT=PROJECT/'experiments/gvcrt_tcg_stage1_pilot/checkpoints/adaptation_step_0.pt'
PILOT_SHA=json.loads((SOURCE/'protocol.json').read_text())['pilot_sha256']
RAFT=Path('/Huang_group/zyyz/home_dir/.cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth')
RAFT_SHA='ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322'
TCG_PREFIXES=('tcg_low.','tcg_mid.','tcg_out.')
SEED=20261106
PREVIOUS=PROJECT/'GVC-RT_worktrees/tcg_stage2_history_gate_joint_20261106/experiments/gvcrt_tcg_stage2_history_gate_joint_20261106'
LORA_PARAMETER_MARKER='.lora.'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(p):
    p=Path(p)
    return json.loads(gzip.decompress(p.read_bytes()).decode() if p.suffix=='.gz' else p.read_text())

def dump(p,x):
    p=Path(p);assert p.is_relative_to(ROOT)
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    raw=(json.dumps(x,indent=None if p.suffix=='.gz' else 2,allow_nan=False)+'\n').encode()
    if p.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    with temp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    temp.replace(p);fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)

def atomic_torch(p,state):
    import torch
    p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    temp.replace(p);fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)

def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    fields=list(dict.fromkeys(k for r in rows for k in r));assert fields
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    temp.replace(p)

@contextlib.contextmanager
def lock(name,blocking=False):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB));yield f

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def env():
    for d in ['logs','tmp','cache/cuda','cache/cupy','cache/mpl','cache/torch_extensions','cache/triton','cache/numba','cache/media','checkpoints','receipts','audits']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',TMPDIR=str(ROOT/'tmp'),CUDA_CACHE_PATH=str(ROOT/'cache/cuda'),
        CUPY_CACHE_DIR=str(ROOT/'cache/cupy'),MPLCONFIGDIR=str(ROOT/'cache/mpl'),TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'),
        TRITON_CACHE_DIR=str(ROOT/'cache/triton'),NUMBA_CACHE_DIR=str(ROOT/'cache/numba'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XDG_CACHE_HOME=str(ROOT/'cache/xdg'),TORCH_HOME=str(RAFT.parents[2]))
    os.environ.setdefault('GIT_SSH_COMMAND','ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3')
    sys.dont_write_bytecode=True

def disk():
    d=shutil.disk_usage(ROOT)
    if d.free<8*2**30:raise RuntimeError('DISK_FREE_BELOW_8_GIB_RESERVE')
    return dict(free_bytes=d.free,total_bytes=d.total)

def event(name,**fields):
    assert 'time' not in fields and 'event' not in fields,'EVENT_RESERVED_FIELDS_MUST_BE_NESTED'
    row=dict(event=name,time=time.time(),**fields)
    with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row,allow_nan=False),flush=True)

GROUPS=('E','F')
REFERENCE_EXPERIMENT=Path('/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_lora_ulong_only_a2000_20261108/experiments/gvcrt_tcg_lora_ulong_only_a2000_20261108')
REFERENCE=Path('/Huang_group/zyyz/Projects/GVC-RT/GVC-RT_worktrees/tcg_stage2_generator_convlora_20261106/experiments/gvcrt_tcg_stage2_generator_convlora_20261106')

def group_dir(group):
    assert group in GROUPS
    return ROOT/group

def physical_gpu(rank):return int(os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')[rank])

def eval_dir(split,step=None,group='baseline'):
    base=ROOT/('reference_A2000_s05' if group=='baseline' else 'candidate')
    return base/'evaluation'/evaluation_id(split,step,'baseline' if group=='baseline' else 'lora')

def status(stage,state='RUNNING',**extra):
    record=dict(status=state,current_stage=stage,time=time.time(),experiment_directory=str(ROOT),experiment_complete=False)
    for group in GROUPS:
        p=group_dir(group)/'training_status.json';record[group]=load(p) if p.exists() else dict(committed_optimizer_updates=0)
    record.update(extra);dump(ROOT/'status.json',record)

def tensor_hash(items):
    h=hashlib.sha256()
    for name,t in items:h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def seed(value):
    import torch,numpy as np
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)

def rng_state():
    import torch,numpy as np
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],torch.tensor(n[1].astype('int64')),*n[2:]),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())

def restore_rng(s):
    import torch,numpy as np
    random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],n[1].numpy().astype('uint32'),*n[2:]));torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda'])

def plan_rows():
    idx=load(ROOT/'plans/index.json');rows=[]
    for entry in idx['shards']:
        p=ROOT/entry['file'];assert sha(p)==entry['sha256'];rows.extend(load(p))
    assert len(rows)==1200 and all(r['domain']=='ulong' and r['logical_sample'] for r in rows)
    return rows

def evaluation_id(split,step=None,candidate='lora'):
    return f'{split}_baseline' if candidate=='baseline' else f'{split}_lora_{step:04d}'

def verify_sources():
    m=load(ROOT/'source_manifest.json');assert m['status']=='FROZEN'
    assert sha(ROOT/'protocol.json')==m['protocol_sha256'] and sha(ROOT/'plans/index.json')==m['plan_index_sha256']
    for name,h in m['code'].items():assert sha(ROOT/name)==h,('CODE_CHANGED_AFTER_FREEZE',name)
    for name,h in m['protected_sources'].items():assert sha(name)==h,('READONLY_SOURCE_CHANGED',name)
    for name,h in m.get('immutable_manifests',{}).items():assert sha(ROOT/name)==h,('LOCAL_MANIFEST_CHANGED',name)
    assert sha(INIT)==INIT_SHA

def expected_source_sha(path):
    return load(GAIN_SOURCE/'sources.json')['protected_source_files'][str(Path(path).resolve())]


class FrozenGuard:
    def __init__(self,items):
        import torch
        self.items=list(items)
        self.baseline=torch.cat([t.detach().contiguous().reshape(-1).view(torch.uint8) for _,t in self.items]).clone()
        self.digest=tensor_hash(self.items)
    def check(self):
        import torch
        current=torch.cat([t.detach().contiguous().reshape(-1).view(torch.uint8) for _,t in self.items])
        assert torch.equal(current,self.baseline),'FROZEN_PERSISTENT_STATE_BYTES_CHANGED'
        return self.digest
    def full(self):
        assert tensor_hash(self.items)==self.digest,'FROZEN_STATE_SHA256_CHANGED'
        return self.digest

PARENT_EXPERIMENT=PROJECT/'GVC-RT_worktrees/tcg_flicker_pilot_abcd_20261109/experiments/gvcrt_tcg_flicker_pilot_abcd_20261109'
B300=PARENT_EXPERIMENT/'B/training/checkpoints/step_0300.pt'
B300_SHA='7465b57f20bfe2a8d6064a87155cf45c31a488cdeaa68111a228c28f6335ea85'
SOURCE_COMMIT='2e70bdc4ad99934d987fe8c484f8430345792b83'
LAMBDA_TEMP=2.603297563918803

# Task-local immutable source roots; existing weights/manifests stay read-only.
ORIGINAL=ROOT/'original_tree'
TCG_REPO=ROOT/'vendor/metric_source'
FLO=ROOT/'vendor/flo'

# Expected identities are read from existing protocols; no manually copied hash values.
_ef=PROJECT/'GVC-RT_worktrees/tcg_flicker_ef_b300_20261109/experiments/gvcrt_tcg_flicker_ef_b300_20261109/protocol.json'
_ep=load(_ef);_parent=load(_ep['inherited_protocol']['path']);_src=_parent['original_codec_and_receiver']
B300=Path(_ep['B300']['path']);B300_SHA=_ep['B300']['SHA256']
INIT=Path(_parent['parent_checkpoint']['path']);INIT_SHA=_parent['parent_checkpoint']['sha256']
PILOT=Path(_src['pilot']);PILOT_SHA=_src['pilot_SHA256']
BASELINE=Path(_src['B1500_TCG']);BASELINE_SHA=_src['B1500_SHA256']
