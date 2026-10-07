"""Native engine with full P receiver loading and fresh entropy CDF tables."""
from v68_io import *
sys.path.insert(0,str(ENGINE))
base=module('v611_native_engine',ENGINE/'engine.py')
metrics=base.metrics
import torch

class Runtime(base.Runtime):
    def models(self,h,w):
        im,pm=super().models(h,w)
        if self.payload is not None and 'full_p' in self.payload:
            pm.load_state_dict(self.payload['full_p'],strict=True)
            pm.update(force_zero_thres=self.config['force_zero_thres'])
            assert state_hash(pm.state_dict())==state_hash(self.payload['full_p'])
            prec=self.payload['precision']
            assert base.core.module_hash(im)==prec['I_hash']
            assert base.core.compression_hash(im,pm)==prec['compression_hash']
        ih=base.core.module_hash(im);expected=load(ROOT/'audits/initialization_audit.json')['I_hash']
        assert ih==expected
        ch=base.core.compression_hash(im,pm)
        cfg=load(ROOT/'evaluation/config.json')
        if self.method in cfg['core_hashes']:assert ch==cfg['core_hashes'][self.method]
        tables={n:array_hash(getattr(pm.bit_estimator_z,n)) for n in ('_quantized_cdf','_cdf_length','_offset')}
        evidence=dict(I_hash=ih,compression_hash=ch,full_p_hash=state_hash(pm.state_dict()),CDF_hash=state_hash(tables),CDF_rebuilt=True)
        if hasattr(self,'receiver_evidence'):assert self.receiver_evidence==evidence
        self.receiver_evidence=evidence
        return im,pm
    def run(self,*args,**kwargs):
        result,rows=super().run(*args,**kwargs)
        result.update(receiver_integrity=self.receiver_evidence,
            precision=None if self.payload is None else self.payload.get('precision'),full_P_checkpoint_loaded=self.payload is not None and 'full_p' in self.payload)
        return result,rows
