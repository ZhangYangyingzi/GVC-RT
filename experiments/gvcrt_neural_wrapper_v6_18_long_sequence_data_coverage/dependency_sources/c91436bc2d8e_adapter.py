"""Isolated adapter; historical real RANS implementation is never edited."""
import types
from io16 import *
def frames_for(v):
    import torch,numpy as np
    from PIL import Image
    out=[];h=hashlib.sha256();hevc=v['dataset']=='hevc_b'
    for i in range(v['frames']):
        p=Path(v['input_dir'])/(f'im{i+1:05d}.png' if hevc else f'frame_{i:06d}.png')
        assert sha(p)==v['frame_png_sha256' if hevc else 'frame_file_sha256'][i]
        with Image.open(p) as im:
            assert im.mode=='RGB' and im.size==(v['width'],v['height']);a=np.asarray(im,dtype=np.uint8).copy()
        raw=a.tobytes();assert hashlib.sha256(raw).hexdigest()==v['frame_rgb_sha256'][i];h.update(raw)
        out.append(torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255)
    assert len(out)==v['frames'] and h.hexdigest()==v['rgb_sha256'];return out
def engine():
    sys.path.insert(0,str(ENGINE));base=module('dists16_native_engine',ENGINE/'engine.py')
    class Runtime(base.Runtime):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.metric_hashes=dict(quality=[base.core.module_hash(x) for x in self.quality],flow_perceptual_inception=[base.core.module_hash(x) for x in self.metric_models])
            assert self.metric_hashes==evalcfg()['metric_module_hashes']
        def models(self,h,w):
            im,pm=super().models(h,w);cfg=evalcfg()
            assert base.core.compression_hash(im,pm)==cfg['compression_hash'] and base.core.module_hash(im)==cfg['I_hash']
            assert {str(p.dtype) for p in im.parameters()}=={str(p.dtype) for p in pm.parameters()}=={'torch.float16'}
            wh=None if self.wrapper is None else base.core.module_hash(self.wrapper)
            if self.method!='original':
                assert wh==cfg['checkpoints'][self.method]['module_hashes']['wrapper']
                assert self.loaded_receiver_hashes==cfg['deployment_receiver_hashes'][self.method]
                assert {str(p.dtype) for p in self.wrapper.parameters()}=={'torch.float32'}
            else:assert self.loaded_receiver_hashes==cfg['original_receiver_hashes']
            self.audit=dict(wrapper_hash=wh,receiver_hashes=self.loaded_receiver_hashes,compression_hash=cfg['compression_hash'],I_hash=cfg['I_hash'],metric_module_hashes=self.metric_hashes,codec_dtype='float16',wrapper_dtype='float32' if self.wrapper is not None else None)
            return im,pm
        def run(self,*args,**kwargs):
            result,rows=super().run(*args,**kwargs);result.update(runtime_audit=self.audit,protocol_sha256=sha(ROOT/'protocol.json'));return result,rows
    base.Runtime=Runtime;return base
_e=None
def evaluator():
    global _e
    if _e is not None:return _e
    sys.path.insert(0,str(BASE));src=(BASE/'evaluate.py').read_text()
    old="parser.add_argument('--smoke',action='store_true');a=parser.parse_args()"
    assert src.count(old)==1
    src=src.replace(old,"parser.add_argument('--smoke',action='store_true');parser.add_argument('--qps',default='');a=parser.parse_args()")
    src=src.replace("qs=[0,9] if a.smoke else list(range(10));pending=[]","qs=list(map(int,a.qps.split(','))) if a.qps else ([0,9] if a.smoke else list(range(10)));pending=[]")
    e=types.ModuleType('dists16_v62b_evaluate');e.__file__=str(BASE/'evaluate.py');exec(compile(src,e.__file__,'exec'),e.__dict__)
    e.ROOT=ROOT;e.METHODS=(*METHODS,'dists_fixed_0');e.DATASETS=DATASETS;e.sources=sources;e.point=point;e.dump=dump;e.write=write
    e.load=lambda p:evalcfg() if Path(p)==ROOT/'config.json' else load(p)
    e.check_frozen=frozen;e.frames_for=frames_for;e.visualization=lambda v,q:None
    orig=e.validate
    def check(r,v,q,m,check_files=True):
        orig(r,v,q,m,check_files);cfg=evalcfg()
        assert r['source_frame_indices']==v['source_frame_indices']
        assert r['compression_hash_before']==r['compression_hash_after']==cfg['compression_hash']
        assert r['evaluator_sha256']==sha(BASE/'evaluate.py')
        assert r['loaded_receiver_module_hashes']==(cfg['original_receiver_hashes'] if m=='original' else cfg['deployment_receiver_hashes'][m])
        if 'runtime_audit' in r:assert r['runtime_audit']['metric_module_hashes']==cfg['metric_module_hashes']
        else:
            assert r.get('reused') and r['metric_dependency_verification']['status']=='PASS'
            assert r['metric_dependency_verification']['dependencies_sha256']==sha(ROOT/'audits/dependencies.json')
            assert r['metric_dependency_verification']['source_metric_audit_sha256']==sha(BASE/'metric_implementation_audit.json')
        if r.get('reused'):
            assert sha(r['reused_from'])==r['reused_sha256']
            assert r['reuse_verified'] and r['reuse_audit_sha256']==sha(ROOT/'audits/baseline_reuse.json')
        else:assert r['config_sha256']==sha(ROOT/'config.json') and r['protocol_sha256']==sha(ROOT/'protocol.json')
    e.validate=check;_e=e;return e
def validate(r,v,q,m):return evaluator().validate(r,v,q,m)
