from util import *
import torch,math
import torch.nn.functional as F
from model import tcg
from objective import quality_models
from temporal import temporal_errors,sobel

def highpass(x):
    k=x.new_tensor([1,4,6,4,1])/16;k=(k[:,None]*k[None,:])[None,None].expand(3,1,5,5)
    return x-F.conv2d(F.pad(x,(2,2,2,2),mode='replicate'),k,groups=3)
class Evaluator:
    def __init__(self,quality=None):
        self.quality=quality or quality_models()
        sys.path.insert(0,str(ROOT/'vendor/flo'));import flolpips_compat
        self.flow,self.flo=flolpips_compat.load_models(torch.device('cuda:0'))
        self.network_hashes=dict(LPIPS=thash(self.quality[0].state_dict().items()),DISTS=thash(self.quality[1].state_dict().items()),PWC=thash(self.flow.state_dict().items()),FloLPIPS=thash(self.flo.state_dict().items()))
    @torch.no_grad()
    def point(self,branch,data,row,label,step,condition,cp_hash,intervention=None,media=False):
        seq=row['sequence'];folder=ROOT/'metrics'/label/f'step_{step:04d}';dest=folder/f'{seq}.json'
        identity=dict(protocol=sha(ROOT/'protocol.json'),checkpoint=cp_hash,metadata=sha(ROOT/'cache/metadata'/f'{seq}.json'),intervention=intervention)
        if dest.exists():
            result=load(dest);assert result['identity']==identity and result['status']=='PASS';return result['metrics']
        start=time.time();data.get(seq);state=None;previous=None;prevgt=None;frames=[];transitions=[];rois=[]
        for i in range(47,64):
            f,p,h,x,q,m=data.tensors(seq,i)
            if m['scene_cut']:state=None
            hh,stats=branch(f,p,h,m,condition,intervention)
            out,state=data.receiver.suffix(hh,x,q,state,False)
            if i<49:continue
            gt=data.gt(seq,i);H,W=gt.shape[-2:];out=out[...,:H,:W]
            lp=float(self.quality[0](out,gt,normalize=True).mean());ds=float(self.quality[1](out,gt,require_grad=False).mean());err=out-gt;mse=float(err.square().mean())
            hp,hg=highpass(out),highpass(gt)
            frame=dict(frame=i,source_frame=row['source_frame_indices'][i],actual_QP=data.sources[seq].info['actual_qps'][i],LPIPS=lp,DISTS=ds,PSNR=-10*math.log10(max(mse,1e-15)),RGB_L1=float(err.abs().mean()),RGB_MSE=mse,highpass_signed_error_MSE=float((hp-hg).square().mean()),signed_Sobel_L1=float((sobel(out)-sobel(gt)).abs().mean()),highpass_energy=float(hp.square().mean()),GT_highpass_energy=float(hg.square().mean()),highpass_cosine=float((hp*hg).mean()/(hp.square().mean().sqrt()*hg.square().mean().sqrt()+1e-12)),**{k:float(v) for k,v in stats.items()})
            frames.append(frame)
            if previous is not None:
                aux=data.sources[seq].aux(i)
                # GT flow is scoring-only, never used by branch, prefix or sampler.
                if 'gt_back' in aux:
                    rr,re,v=temporal_errors(out,previous,gt,prevgt,aux['gt_back'].cuda().float(),aux['gt_valid'].cuda())
                else:rr=re=None;v=None
                fd=self.flow(prevgt,gt)-self.flow(previous,out)
                flo=float(self.flo(prevgt,previous,fd,normalize=True).mean())
                transitions.append(dict(from_frame=i-1,to_frame=i,FloLPIPS=flo,MC_RGB=float(rr) if rr is not None else None,MC_Sobel=float(re) if re is not None else None,GT_flow_valid_fraction=v))
            previous,prevgt=out,gt
            if media:
                xx,yy=row['ROI']['x'],row['ROI']['y'];rois.append(out[...,yy:yy+256,xx:xx+256].mul(255).round().clamp(0,255).byte().cpu())
            dump(folder/'progress.json',dict(pid=os.getpid(),sequence=seq,frame=i,time=time.time()))
        metrics=dict(label=label,step=step,sequence=seq,split=row['split'],frames=15,transitions=14,QP=5,checkpoint_sha256=cp_hash,stream_sha256=data.sources[seq].info['stream_SHA256'])
        for k in frames[0]:
            if k not in ['frame','source_frame','actual_QP']:metrics[k]=sum(r[k] for r in frames)/len(frames)
        metrics['PSNR']=-10*math.log10(metrics['RGB_MSE'])
        metrics['objective']=metrics['LPIPS']+.5*metrics['DISTS']+.1383589248954108*metrics['RGB_L1']
        for k in ['FloLPIPS','MC_RGB','MC_Sobel','GT_flow_valid_fraction']:
            vs=[r[k] for r in transitions if r[k] is not None];metrics[k]=sum(vs)/len(vs) if vs else None
        csv_write(folder/f'{seq}_frames.csv',frames);csv_write(folder/f'{seq}_transitions.csv',transitions)
        if rois:save(ROOT/'media/roi_tensors'/label/f'{seq}_step{step:04d}.pt',dict(frames=torch.cat(rois),ROI=row['ROI'],frame_indices=list(range(49,64))))
        dump(dest,dict(status='PASS',identity=identity,metrics=metrics,seconds=time.time()-start,device=device_info(),network_hashes=self.network_hashes))
        return metrics
    def all(self,branch,data,label,step,condition,cp_hash,intervention=None,media=False):
        return [self.point(branch,data,r,label,step,condition,cp_hash,intervention,media) for r in videos() if r['split']!='train']
