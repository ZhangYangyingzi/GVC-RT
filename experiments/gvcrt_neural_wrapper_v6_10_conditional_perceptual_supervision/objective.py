"""V6.6 B AST with an external per-frame consumer; unchanged loss and detached DPB."""
from v68_io import *
import torch
from model_runtime import norm

class Objective:
    def __init__(self,v4,im,pm,models,quality,cfg):
        self.models=models;self.records=[];self.d=None
        source=(V66/'objective.py').read_text()
        start=source.index('    def consume(');end=source.index("    context['consume']=consume",start)
        source=source[:start]+"    consume = external_consume\n"+source[end:]
        needle="'consume(components,wp)'"
        assert source.count(needle)==1
        source=source.replace(needle,"'consume(components,wp,rec,frame,capt[-1],i,q)'")
        sys.path.insert(0,str(V66))
        ns=dict(__file__=str(V66/'objective.py'),external_consume=self.consume)
        exec(compile(source,str(ROOT/'objective.py'),'exec'),ns)
        self.fn=ns['bind'](v4,im,pm,models['wrapper'],quality,cfg,weight=cfg['lambda_struct'])
        self.params=[p for m in models.values() for p in m.parameters()]
        self.slices={};offset=0
        for k,m in models.items():
            n=len(list(m.parameters()));self.slices[k]=slice(offset,offset+n);offset+=n
    def consume(self,c,wp,rec,frame,feature,index,qp):
        if self.mode=='collect':
            self.records.append(dict(rec=rec.detach().clone(),real=frame.detach(),
                condition=feature.detach().float().clone(),native_feature_dtype=str(feature.dtype),index=index))
            return
        adv=rec.new_zeros(())
        if self.d is not None and index>=2:
            prev=rec if self.branch=='S_cond_image' else self.previous
            assert self.branch!='T_cond_temporal' or not prev.requires_grad
            assert rec.requires_grad and index in (2,3)
            fake=torch.cat((prev,rec),dim=1)
            condition=feature.detach().float()
            score=self.d(fake.float(),condition,qp)
            adv=-score.mean()/2
            self.adv+=float(adv.detach());self.fake_score+=float(score.detach().mean())/2
        base=c['loss']/3
        if self.measure:
            for label,term in [('L_B',base),('weighted_L_adv',self.weight*adv)]:
                if term.requires_grad:
                    gs=torch.autograd.grad(term,self.params,retain_graph=True,allow_unused=True)
                    for dst,g in zip(self.acc[label],gs):
                        if g is not None:dst.add_(g.detach())
        if self.capture_outputs:self.outputs.append(rec.detach().cpu().clone())
        # Preserve the exact B backward operation when the coefficient is zero.
        (base if self.weight==0 else base+self.weight*adv).backward()
        self.previous=rec.detach()
    def collect(self,frames,qp):
        self.mode='collect';self.records=[]
        with torch.no_grad():self.fn(frames,qp)
        assert [r['index'] for r in self.records]==[1,2,3]
        return self.records
    def __call__(self,frames,qp,branch='C_plain',disc=None,weight=0.,measure=False,capture=False):
        self.mode='generator';self.branch=branch;self.d=disc;self.weight=weight
        self.measure=measure;self.capture_outputs=capture;self.outputs=[];self.previous=None
        self.adv=0.;self.fake_score=0.
        if measure:self.acc={k:[torch.zeros_like(p) for p in self.params] for k in ('L_B','weighted_L_adv')}
        result,actual=self.fn(frames,qp)
        result.update(L_B=result['loss'],L_adv=self.adv,weighted_L_adv=weight*self.adv,
            lambda_adv=weight,total_loss=result['loss']+weight*self.adv,generator_fake_score=self.fake_score)
        self.gradient_measurements={}
        if measure:
            self.gradient_measurements={term:{k:norm(gs[sl]) for k,sl in self.slices.items()} for term,gs in self.acc.items()}
            del self.acc
        self.previous=None
        return result,actual
    def close(self):self.records=[];self.previous=None

def discriminator_update(d,optimizer,records,branch,qp):
    import torch.nn.functional as F
    real=[];fake=[];conditions=[]
    for i in (1,2):
        curr=records[i];previous=records[i-1]
        real.append(torch.cat((curr['real'] if branch=='S_cond_image' else previous['real'],curr['real']),dim=1))
        fake.append(torch.cat((curr['rec'] if branch=='S_cond_image' else previous['rec'],curr['rec']),dim=1))
        conditions.append(curr['condition'])
    r=torch.cat(real).detach().float();f=torch.cat(fake).detach().float();c=torch.cat(conditions).detach().float()
    assert r.shape==f.shape==(2,6,256,256) and not c.requires_grad and not f.requires_grad
    d.train().requires_grad_(True);optimizer.zero_grad(set_to_none=True)
    scores=d(torch.cat((r,f)),torch.cat((c,c)),qp);rs,fs=scores.chunk(2)
    loss=F.relu(1-rs).mean()+F.relu(1+fs).mean()
    loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(d.parameters(),1.))
    assert torch.isfinite(loss) and 0<gn<float('inf')
    optimizer.step();optimizer.zero_grad(set_to_none=True);d.eval().requires_grad_(False)
    return dict(L_D=float(loss.detach()),real_score=float(rs.detach().mean()),fake_score=float(fs.detach().mean()),discriminator_gradient_norm=gn,discriminator_lr=optimizer.param_groups[0]['lr'])
