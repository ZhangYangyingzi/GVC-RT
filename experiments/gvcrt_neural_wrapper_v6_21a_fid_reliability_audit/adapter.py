"""Historical codec/metric implementation with a single first-frame input switch."""
import types,inspect,textwrap
from io21 import *
def frames_for(v):
    import torch,numpy as np
    p=Path(v['raw_rgb_path']);assert sha(p)==v['raw_rgb_sha256']
    a=np.memmap(p,dtype=np.uint8,mode='r',shape=(v['frames'],1080,1920,3))
    return [torch.from_numpy(a[i].copy()).permute(2,0,1).unsqueeze(0).float()/255 for i in range(v['frames'])]
def engine():
    sys.path.insert(0,str(ENGINE));base=module('i19_native_engine',ENGINE/'engine.py')
    source=textwrap.dedent(inspect.getsource(base.Runtime.run))
    def change(a,b):
        nonlocal source
        assert source.count(a)==1,(a,source.count(a));source=source.replace(a,b)
    change('proxy=target if self.wrapper is None else self.wrapper(target)',"proxy=target if self.wrapper is None or (i==0 and evalcfg()['i_frame_modes'][self.method]=='native_RGB_bypass') else self.wrapper(target)")
    change("decoded=idc.decompress(encoded['bit_stream'],sps,q);pdc.clear_dpb();pdc.add_ref_frame(None,decoded['x_hat'])", "decoded=idc.decompress(encoded['bit_stream'],sps,q);pdc.clear_dpb();pdc.add_ref_frame(None,decoded['x_hat'])\n            first_audit=dict(payload_sha256=eco.sha256_bytes(encoded['bit_stream']),payload_bytes=len(encoded['bit_stream']),decoded_tensor_sha256=eco.tensor_sha(decoded['x_hat']),encoder_initial_reference_sha256=eco.tensor_sha(pe.dpb[0].frame),decoder_initial_reference_sha256=eco.tensor_sha(pdc.dpb[0].frame))\n            assert first_audit['encoder_initial_reference_sha256']==first_audit['decoder_initial_reference_sha256']==first_audit['decoded_tensor_sha256']")
    change("output=core.unit(decoded['x_hat'],h,w);target=cpu.to(self.device)","output=core.unit(decoded['x_hat'],h,w);target=cpu.to(self.device)\n        if i==0:\n            first_audit['independent_decoded_tensor_sha256']=eco.tensor_sha(decoded['x_hat']);assert first_audit['independent_decoded_tensor_sha256']==first_audit['decoded_tensor_sha256']\n            first_audit['independent_initial_reference_sha256']=eco.tensor_sha(pa.dpb[0].frame);assert first_audit['independent_initial_reference_sha256']==first_audit['decoded_tensor_sha256']\n            if self.capture_first:\n                proxy=target if self.wrapper is None else self.wrapper(target)\n                first_path=feature_path.with_suffix('.first.npz')\n                first_path.parent.mkdir(parents=True,exist_ok=True)\n                with first_path.open('wb') as f:np.savez(f,GT=target.cpu().numpy(),P_GT=proxy.cpu().numpy(),reconstruction=output.cpu().numpy())\n                first_audit['visualization_cache_path']=str(first_path);first_audit['visualization_cache_sha256']=sha(first_path)")
    change("result['loaded_receiver_module_hashes']=self.loaded_receiver_hashes", "result['loaded_receiver_module_hashes']=self.loaded_receiver_hashes\n    result['first_frame_audit']=first_audit")
    change('del ie,pe,idc,pdc', 'self.assert_frozen(ie,pe);self.assert_frozen(idc,pdc)\n    del ie,pe,idc,pdc')
    change('del ia,pa', 'self.assert_frozen(ia,pa)\n    del ia,pa')
    ns=dict(base.__dict__,evalcfg=evalcfg);exec(compile(source,str(ROOT/'adapter.py'),'exec'),ns);native_run=ns['run']
    class Runtime(base.Runtime):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw);self.capture_first=False
            if self.wrapper is not None:self.wrapper.eval().requires_grad_(False)
            self.metric_hashes=dict(quality=[base.core.module_hash(x) for x in self.quality],flow_perceptual_inception=[base.core.module_hash(x) for x in self.metric_models]);assert self.metric_hashes==evalcfg()['metric_module_hashes']
        def models(self,h,w):
            im,pm=super().models(h,w);cfg=evalcfg()
            audit=dict(compression_hash=base.core.compression_hash(im,pm),I_hash=base.core.module_hash(im),wrapper_hash=None if self.wrapper is None else base.core.module_hash(self.wrapper),receiver_hashes=self.loaded_receiver_hashes,metric_module_hashes=self.metric_hashes)
            assert audit['compression_hash']==cfg['compression_hash'] and audit['I_hash']==cfg['I_hash']
            assert audit['receiver_hashes']==(cfg['original_receiver_hashes'] if self.method=='original' else cfg['deployment_receiver_hashes'][self.method])
            if self.method!='original':assert audit['wrapper_hash']==cfg['checkpoints'][self.method]['module_hashes']['wrapper']
            assert all(not p.requires_grad for m in (im,pm,*self.quality,*self.metric_models) for p in m.parameters())
            assert {str(p.dtype) for p in im.parameters()}=={str(p.dtype) for p in pm.parameters()}=={'torch.float16'}
            if self.wrapper is not None:assert {str(p.dtype) for p in self.wrapper.parameters()}=={'torch.float32'}
            if hasattr(self,'audit'):assert self.audit==audit
            self.audit=audit;return im,pm
        def assert_frozen(self,im,pm):
            assert base.core.compression_hash(im,pm)==self.audit['compression_hash']
            assert base.core.module_hash(im)==self.audit['I_hash']
            assert {k:base.core.module_hash(m) for k,m in (('bridge',pm.recon_generation_net.mlp),('generator',pm.recon_generation_net.decoder))}==self.audit['receiver_hashes']
        def run(self,*args,**kw):
            before=(None if self.wrapper is None else base.core.module_hash(self.wrapper),[base.core.module_hash(x) for x in (*self.quality,*self.metric_models)])
            result,rows=native_run(self,*args,**kw)
            after=(None if self.wrapper is None else base.core.module_hash(self.wrapper),[base.core.module_hash(x) for x in (*self.quality,*self.metric_models)]);assert before==after
            result.update(runtime_audit=self.audit,model_hashes_unchanged=True,i_frame_mode=evalcfg()['i_frame_modes'][self.method],protocol_sha256=sha(ROOT/'protocol.json'))
            return result,rows
    base.Runtime=Runtime;return base
_e=None
