from util import *
import torch
import torch.nn.functional as F
from alignment import grid,sample,bounds,geometry

def load_raft():
    from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
    from common_s2 import RAFT,RAFT_SHA
    assert sha(RAFT)==RAFT_SHA
    m=raft_large(weights=None,progress=False);m.load_state_dict(torch.load(RAFT,map_location='cpu',weights_only=True),strict=True)
    return m.cuda().eval().requires_grad_(False),Raft_Large_Weights.C_T_SKHT_V2.transforms()

class OriginalData:
    def __init__(self,row):
        self.row=row;self.seq=row['sequence'];self.data=None
        if 'reuse_cache' in row:
            assert sha(row['reuse_receipt'])==row['reuse_receipt_sha256'];proof=load(row['reuse_receipt']);p=Path(row['reuse_cache']);assert p.stat().st_size==proof['cache_bytes']
            self.data=torch.load(p,map_location='cpu',weights_only=True,mmap=True);self.info=self.data['info']
            assert self.info['frames']==64 and self.info['external_QP']==5 and self.info['native_shape']==[row['height'],row['width']]
            assert self.info['source_row']['source_sha256']==row['source_sha256']
            assert sha(self.info['stream_path'])==self.info['stream_SHA256']
        else:
            self.dest=(OLD/'cache' if row.get('reuse_old_original') else PARENT/'cache')/'original'/self.seq
            assert load(self.dest/'completion.json')['status']=='PASS';self.info=load(self.dest/'info.json')
            assert sha(self.info['stream_path'])==self.info['stream_SHA256']
    def feature(self,i):
        if self.data is not None:
            t=self.data['features'][i];expected=self.info['codec_trace'][i]['codec_feature_SHA256']
            assert hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()==expected
            return t.float()
        p=self.dest/f'f_{i:02d}.pt';return torch.load(p,map_location='cpu',weights_only=True,mmap=True).float()
    def aux(self,i):
        if self.data is None:
            aux=torch.load(self.dest/f'aux_{i:02d}.pt',map_location='cpu',weights_only=True,mmap=True)
            if 'original_m1p1_fp16' in aux:aux['original']=(aux['original_m1p1_fp16'].float()+1)/2
            return aux
        h,w=self.info['native_shape'];orig=(self.data['original'][i].float().clamp(-1,1)+1)/2;prev=(self.data['original'][i-1].float().clamp(-1,1)+1)/2
        return dict(original=orig[...,:h,:w],GT=self.data['GT'][i],back=self.data['base_backward'][i],valid=self.data['base_valid'][i],scene_cut=bool((orig-prev).abs().mean()>.35),gt_back=self.data['GT_backward'][i],gt_valid=self.data['GT_valid'][i])

@torch.no_grad()
def prepare_metadata(row):
    dest=ROOT/'cache/metadata'/f"{row['sequence']}.pt";proof=dest.with_suffix('.json')
    if proof.exists():assert sha(dest)==load(proof)['sha256'];return
    src=OriginalData(row);raft,transform=load_raft();result={};start=time.time()
    previous=None
    for i in range(47,64):
        aux=src.aux(i);h,w=src.info['native_shape'];ph,pw=src.info['padded_shape'];fh,fw=ph//8,pw//8
        backward=aux['back'].cuda().float();valid=aux['valid'].cuda().bool()
        if previous is None:
            error=torch.zeros_like(backward[:,:1]);valid=valid&False
        else:
            a=previous.cuda().float();b=aux['original'].cuda().float()
            a=F.pad(a,(0,(-w)%8,0,(-h)%8),mode='replicate');b=F.pad(b,(0,(-w)%8,0,(-h)%8),mode='replicate')
            a,b=transform(a,b);forward=raft(a,b)[-1][...,:h,:w]
            coords=grid(h,w,'cuda')+backward
            error=(backward+sample(forward,coords)).square().sum(1,keepdim=True).sqrt()
            # Recompute validity using the identical original audited rule.
            fwarped=sample(forward,coords)
            valid=bounds(coords,h,w)&((backward+fwarped).square().sum(1,keepdim=True)<=.01*(backward.square().sum(1,keepdim=True)+fwarped.square().sum(1,keepdim=True))+.5)
        m=geometry(backward,valid,error,fh,fw,available=not aux['scene_cut'] and previous is not None)
        result[i]={k:v.cpu() for k,v in m.items()};result[i]['scene_cut']=aux['scene_cut']
        previous=aux['original']
    save(dest,result)
    dump(proof,dict(status='PASS',sha256=sha(dest),stream_sha256=src.info['stream_SHA256'],source_sha256=row['source_sha256'],frames=list(result),first_warmup_history_closed=True,GT_used=False,RAFT_backward_reused=True,RAFT_forward_recomputed=True,seconds=time.time()-start,device=device_info()))

class Dataset:
    def __init__(self,c,position):
        from parent_model import Receiver
        self.receiver=Receiver(c,position);self.c=c;self.items={};self.raw={};self.metadata={};self.sources={};self.rows={r['sequence']:r for r in videos()}
    def get(self,seq):
        if seq in self.items:return self.items[seq]
        r=self.rows[seq];src=OriginalData(r);self.sources[seq]=src;g=self.receiver.gen;g.reset_temporal_buffer();g.set_temporal_cascade_grad(False)
        prefix={};raw={};start=time.time();meta=torch.load(ROOT/'cache/metadata'/f'{seq}.pt',map_location='cpu',weights_only=True,mmap=True)
        with torch.no_grad():
            for i in range(1,64):
                f=src.feature(i).cuda().float();q=self.c['pm'].q_scale_recon[src.info['actual_qps'][i]:src.info['actual_qps'][i]+1].float()
                if i in meta and meta[i]['scene_cut']:g.reset_temporal_buffer()
                h,x,q=self.receiver.prefix(f,q)
                if i>=46:raw[i]=f.cpu()
                if i>=47:prefix[i]=dict(h=h.cpu(),x=x.cpu(),q=q.cpu())
        g.reset_temporal_buffer()
        self.items[seq]=prefix;self.raw[seq]=raw;self.metadata[seq]=meta
        dump(ROOT/'audits/prefix'/os.environ.get('GROUP_LABEL','cache')/self.receiver.position/f'{seq}.json',dict(status='PASS',stream_sha256=src.info['stream_SHA256'],B300_SHA256=self.c['provenance']['B300_SHA256'],prefix_frozen=True,old_terminal_h_used=False,frames=list(prefix),h_shape=list(h.shape),seconds=time.time()-start))
        return prefix
    def tensors(self,seq,i):
        d=self.get(seq)[i];raw=self.raw[seq];m={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in self.metadata[seq][i].items()}
        return raw[i].cuda(),raw[i-1].cuda(),d['h'].cuda(),d['x'].cuda(),d['q'].cuda(),m
    def gt(self,seq,i):
        # The only use of GT is targets/scoring; no GT value enters prefix or branch.
        self.get(seq);a=self.sources[seq].aux(i)['GT']
        return a.permute(2,0,1)[None].cuda().float()/255

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);a=p.parse_args();setup_torch();prepare_metadata(next(r for r in videos() if r['sequence']==a.video))
