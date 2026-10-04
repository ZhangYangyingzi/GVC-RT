"""Fixed held-out/UVG interface diagnostic; native inference receiver, no training."""
import argparse,math
from v68_io import *
from model_runtime import Capture
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch,numpy as np
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(2);cfg=frozen();e=engine();core=e.core;device=torch.device('cuda:0');plan=load(ROOT/'audits/interface_diagnostic_plan.json')
    _,wrapper=e.eco.load_joint(cfg['checkpoints'][a.method]['path'],device);state=torch.load(cfg['checkpoints'][a.method]['path'],map_location='cpu',weights_only=True)
    si,sp=core.load_models(device,force_zero_thres=.12);e.eco.apply_joint(sp,state);ti,tp=core.load_models(device,force_zero_thres=.12);sc=Capture(sp);tc=Capture(tp)
    assert all(not p.requires_grad for m in (si,sp,ti,tp,wrapper) for p in m.parameters());before=dict(teacher=core.module_hash(tp),compression=core.compression_hash(si,sp))
    clips=[]
    for v in load(ROOT/'audits/vimeo_heldout_manifest.json')['clips']:
        if v['index'] not in plan['vimeo_indices']:continue
        frames=[]
        for p,h in zip(v['paths'][:4],v['file_sha256']):
            assert sha(p)==h
            with Image.open(p) as im:arr=np.asarray(im,dtype=np.uint8).copy()
            frames.append(torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()/255)
        clips.append(('vimeo_official_test_separate',v['sequence'],frames))
    io=module('v68_diagnostic_sources',V62B/'v62b_io.py')
    for v in sources('uvg'):
        if v['video_index'] in plan['uvg_indices']:clips.append(('uvg',v['name'],io.frames_for(v)[:4]))
    rows=[]
    with torch.no_grad():
        for dataset,name,frames in clips:
            for q in plan['qps']:
                for pm in (sp,tp):pm.clear_dpb();pm.set_curr_poc(0)
                for i,frame in enumerate(frames):
                    x=frame.to(device);sq=q if i==0 else sp.shift_qp(q,e.eco.INDEX_MAP[i%8]);tq=q if i==0 else tp.shift_qp(q,e.eco.INDEX_MAP[i%8]);assert sq==tq
                    if i==0:
                        sp.add_ref_frame(None,si.compress(core.codec_input(wrapper(x)),sq)['x_hat']);tp.add_ref_frame(None,ti.compress(core.codec_input(x),tq)['x_hat']);continue
                    # Native forward returns the same interface as the independently decoded
                    # path (verified in preflight); each instance updates only its own DPB.
                    sr=sp(core.codec_input(wrapper(x)),sq);tr=tp(core.codec_input(x),tq);zs=sc.latent().float().flatten(1);zt=tc.latent().float().flatten(1);assert zs.shape==zt.shape
                    cs=float(F.cosine_similarity(zs,zt,dim=1,eps=1e-8).mean());nl=float(((zs-zt).norm(dim=1)/(zt.norm(dim=1)+1e-8)).mean());ratio=float((zs.norm(dim=1)/(zt.norm(dim=1)+1e-8)).mean());assert all(math.isfinite(z) for z in (cs,nl,ratio))
                    rows.append(dict(dataset=dataset,sequence=name,method=a.method,QP=q,actual_qp=sq,frame=i,cosine_similarity=cs,normalized_L2=nl,norm_ratio=ratio,interface='pm.recon_generation_net.decoder forward_pre_hook inputs[0]',teacher_dtype=str(tc.values['generator_input'].dtype),student_dtype=str(sc.values['generator_input'].dtype),no_training=True))
                    sp.add_ref_frame(sc.values['compression_output'].detach(),sr['x_hat'].detach());tp.add_ref_frame(tc.values['compression_output'].detach(),tr['x_hat'].detach());sc.clear();tc.clear()
    assert core.module_hash(tp)==before['teacher'] and core.compression_hash(si,sp)==before['compression']==cfg['compression_hash'];sc.close();tc.close()
    write(ROOT/'parts/diagnostics'/f'{a.method}.csv',rows);dump(ROOT/'parts'/f'diagnostic_done_{a.method}.json',dict(status='PASS',rows=len(rows),path=str(ROOT/'parts/diagnostics'/f'{a.method}.csv'),sha256=sha(ROOT/'parts/diagnostics'/f'{a.method}.csv')))
    print('INTERFACE DIAGNOSTIC PASS',a.method,len(rows),flush=True)
if __name__=='__main__':main()
