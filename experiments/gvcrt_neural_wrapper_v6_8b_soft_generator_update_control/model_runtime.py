"""Exact source restoration and independent native Original GVC-RT teacher."""
import random,math
from v68_io import *
def restore_rng(s):
    import torch
    random.setstate(s['sample_rng_state']);torch.set_rng_state(s['torch_rng_state']);torch.cuda.set_rng_state(s['cuda_rng_state'])
def setup(device,cp=SOURCE,quality=True):
    import torch
    v4=module('v68_training_core',V4/'train.py');cfg=load(ROOT/'config.json')
    wrapper=v4.NeuralWrapper().to(device).float().train();im,pm=v4.load_models(device,force_zero_thres=.12)
    bridge=pm.recon_generation_net.mlp.float().train().requires_grad_(True);generator=pm.recon_generation_net.decoder.float().train().requires_grad_(True)
    models=dict(wrapper=wrapper,bridge=bridge,generator=generator)
    s=torch.load(cp,map_location='cpu',weights_only=True)
    for k,m in models.items():m.load_state_dict(s[k],strict=True)
    oc=cfg['optimizer'];opt=torch.optim.AdamW([dict(params=list(m.parameters()),lr=oc[k+'_lr'],name=k) for k,m in models.items()],weight_decay=oc['weight_decay'])
    opt.load_state_dict(s['optimizer']);assert state_hash(opt.state_dict())==state_hash(s['optimizer'])
    assert v4.compression_hash(im,pm)==cfg['compression_hash'];ids={id(p) for m in models.values() for p in m.parameters()}
    assert all(not p.requires_grad for m in (im,pm) for p in m.parameters() if id(p) not in ids)
    qm=v4.quality_models(device) if quality else None;restore_rng(s)
    return v4,im,pm,models,opt,qm,s
class Capture:
    def __init__(self,pm):
        self.values={};self.details={}
        def put(k,x):self.values[k]=x;self.details[k]=dict(shape=list(x.shape),dtype=str(x.dtype))
        self.hooks=[pm.dec.register_forward_hook(lambda m,a,o:put('compression_output',o)),pm.recon_generation_net.mlp[0].register_forward_pre_hook(lambda m,a:put('bridge_input',a[0])),pm.recon_generation_net.mlp[3].register_forward_hook(lambda m,a,o:put('bridge_output',o)),pm.recon_generation_net.decoder.register_forward_pre_hook(lambda m,a:put('generator_input',a[0])),pm.recon_generation_net.decoder.register_forward_hook(lambda m,a,o:put('final_RGB',o))]
    def latent(self):
        import torch
        assert torch.equal(self.values['bridge_output'],self.values['generator_input'])
        return self.values['generator_input']
    def clear(self):self.values.clear()
    def close(self):
        for h in self.hooks:h.remove()
        self.clear()
class Teacher:
    def __init__(self,v4,device):
        self.v4=v4;self.device=device;self.im,self.pm=v4.load_models(device,force_zero_thres=.12);self.capture=Capture(self.pm)
        self.hashes=self.hash();self.assert_frozen()
    def hash(self):return dict(I=self.v4.module_hash(self.im),P=self.v4.module_hash(self.pm))
    def assert_frozen(self):
        assert all(not m.training for root in (self.im,self.pm) for m in root.modules())
        assert all(not p.requires_grad and p.grad is None for root in (self.im,self.pm) for p in root.parameters())
    def targets(self,frames,q,verify_decode=False):
        import torch
        sys.path.insert(0,str(V62));from fullqp_common import frame_qp
        before=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone());out=[];streams=[];features=[]
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            self.pm.clear_dpb();self.pm.set_curr_poc(0);actual=[frame_qp(self.pm,q,i) for i in range(len(frames))]
            first=self.im.compress(self.v4.codec_input(frames[0]),actual[0]);self.pm.add_ref_frame(None,first['x_hat']);streams.append(first['bit_stream'])
            for i,x in enumerate(frames[1:],1):
                self.capture.clear();r=self.pm(self.v4.codec_input(x),actual[i]);z=self.capture.latent().detach().clone();out.append(z);streams.append(r['bit_stream'])
                f=self.capture.values['compression_output'].detach();self.pm.add_ref_frame(f,r['x_hat'].detach());features.append(f.clone())
            self.last_details=dict(self.capture.details);self.capture.clear()
            if verify_decode:
                di,dp=self.v4.load_models(self.device,force_zero_thres=.12);cap=Capture(dp);dp.clear_dpb();dp.set_curr_poc(0)
                sps=dict(height=frames[0].shape[-2],width=frames[0].shape[-1],ec_part=1)
                decoded=di.decompress(streams[0],sps,actual[0]);assert torch.equal(decoded['x_hat'],first['x_hat']);dp.add_ref_frame(None,decoded['x_hat'])
                for i in range(1,len(frames)):
                    dp.decompress(streams[i],sps,actual[i]);assert torch.equal(cap.latent(),out[i-1]);assert torch.equal(dp.dpb[0].feature,features[i-1])
                cap.close();del di,dp
        assert torch.equal(before[0],torch.get_rng_state()) and torch.equal(before[1],torch.cuda.get_rng_state())
        self.assert_frozen();return out,actual
def norm(gs):return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs if g is not None))
def cosine(a,b):
    na,nb=norm(a),norm(b)
    return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)/(na*nb) if na*nb else None
def disjoint_states(student,teacher):
    def ptrs(pm):return {x.data_ptr() for r in pm.dpb for x in (r.feature,r.frame) if x is not None}
    return student is not teacher and student.dpb is not teacher.dpb and not ptrs(student)&ptrs(teacher)
