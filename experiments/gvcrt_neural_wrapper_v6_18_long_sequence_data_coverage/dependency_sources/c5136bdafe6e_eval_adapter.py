"""Historical V6.2b evaluation with local paths and explicit QP subset only."""
import types
from v614_io import *
_e=None
def evaluator():
    global _e
    if _e is not None:return _e
    sys.path.insert(0,str(BASE));source=(BASE/'evaluate.py').read_text()
    source=source.replace("parser.add_argument('--smoke',action='store_true');a=parser.parse_args()","parser.add_argument('--smoke',action='store_true');parser.add_argument('--qps',default='');a=parser.parse_args()")
    source=source.replace("qs=[0,9] if a.smoke else list(range(10));pending=[]","qs=list(map(int,a.qps.split(','))) if a.qps else ([0,9] if a.smoke else list(range(10)));assert all(q in range(10) for q in qs);pending=[]")
    e=types.ModuleType('v614_evaluation');e.__file__=str(BASE/'evaluate.py');exec(compile(source,e.__file__,'exec'),e.__dict__)
    e.ROOT=ROOT;e.METHODS=('original','v62_initial',*[name(s) for s in STEPS]);e.DATASETS=('uvg',);e.sources=sources;e.point=point;e.dump=dump;e.write=write;e.visualization=lambda v,q:None
    e.load=lambda p:load(ROOT/'evaluation/config.json') if Path(p)==ROOT/'config.json' else load(p)
    e.check_frozen=frozen;orig=e.validate
    def validate(r,v,q,m,check_files=True):
        orig(r,v,q,m,check_files)
        assert r['compression_hash_before']==r['compression_hash_after']==load(ROOT/'evaluation/config.json')['compression_hash']
        assert r['source_frame_indices']==v['source_frame_indices']
        if r.get('reused'):assert sha(r['reused_from'])==r['reused_sha256']
        else:assert r['config_sha256']==sha(ROOT/'config.json')
    e.validate=validate;_e=e;sys.path.insert(0,str(ROOT));return e
def validate(r,v,q,m):evaluator().validate(r,v,q,m)
def publish():
    cfg=load(ROOT/'evaluation/config.json');index=load(ROOT/'checkpoint_index.json') if (ROOT/'checkpoint_index.json').exists() else {};changed=False
    for s,cp in index.items():
        m=name(int(s))
        if m not in cfg['checkpoints']:
            inf=cp['inference'];assert sha(inf['path'])==inf['sha256'];cfg['checkpoints'][m]=inf;changed=True
    if changed:dump(ROOT/'evaluation/config.json',cfg)
    return cfg
