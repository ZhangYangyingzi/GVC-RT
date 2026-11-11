"""Fixed 32 ULong development videos x three QPs, all prescribed receiver states."""
from common_s2 import *
import argparse

METRICS=['PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','RGB_MCResidual','Sobel_MCResidual','RGB_MCResidual_PtoP','Sobel_MCResidual_PtoP','RGB_MCResidual_ItoP','Sobel_MCResidual_ItoP','Sobel_edge_strength','GT_Sobel_edge_strength','edge_relative_GT_bias','local16_energy_abs_bias']


class Scorer:
    def __init__(self):
        import torch
        from model_context import backend
        backend(False,0)
        from quality import lpips_model,metric_audit
        from DISTS_pytorch import DISTS
        sys.path.insert(0,str(FLO));import flolpips_compat
        self.device=torch.device('cuda:0')
        self.alex=lpips_model('alex',self.device);self.vgg=lpips_model('vgg',self.device);self.dists=DISTS().to(self.device).eval().requires_grad_(False)
        self.flow,self.flo=flolpips_compat.load_models(self.device)
        self.hashes={name:tensor_hash(model.state_dict().items()) for name,model in [('LPIPS_Alex',self.alex),('LPIPS_VGG',self.vgg),('DISTS',self.dists),('PWC',self.flow),('FloLPIPS',self.flo)]}
        self.identity=digest(dict(weights=self.hashes,precision=DEPLOYMENT,implementation={n:sha(ROOT/n) for n in ['evaluate.py','quality.py','temporal.py']}))
        dump(ROOT/'audits/metric_implementation.json',dict(status='PASS',weights=self.hashes,identity=self.identity,precision=DEPLOYMENT,LPIPS=metric_audit(),no_historical_FP16_scalar_reuse=True))

    def point(self,data,model,step,original=None):
        import torch
        import torch.nn.functional as F
        from experiment_model import construct,restore,reconstruct
        from model_context import assert_reset
        from temporal import temporal_errors,sobel
        info=data['info'];video=info['source_row'];h,w=info['native_shape'];q=info['external_QP'];n=info['frames'];assert n==64
        key=f'{video["sequence"]}_qp{q}';root=ROOT/'evaluation'/model/f'step_{step:04d}';dest=root/'points'/f'{key}.json'
        cp=None if model in ['Original','B300'] or step==0 else ROOT/model/f'training/checkpoints/step_{step:04d}.pt'
        cp_hash=B300_SHA if cp is None else sha(cp)
        if model=='Original':cp_hash=digest({k:v for k,v in load(ROOT/'inputs/H_source_inputs.json')['readonly'].items() if k.endswith(('GVC-RT_I.pt','GVC-RT_P.pt'))})
        identity=dict(metric_id=self.identity,model=model,step=step,checkpoint_SHA256=cp_hash,stream_SHA256=info['stream_SHA256'],source_SHA256=video['source_sha256'],source_frames=video['source_frame_indices'],precision=DEPLOYMENT)
        if dest.exists():
            saved=load(dest);assert saved['identity']==identity and saved['status']=='PASS'
            for path,value in saved['record_SHA256'].items():assert sha(ROOT/path)==value
            return saved
        c=None
        if model!='Original':
            c=construct(model,False,0,losses=False)
            if cp is not None:restore(c,torch.load(cp,map_location='cpu',weights_only=True),False)
            assert_reset(c);before=tensor_hash(c['pm'].state_dict().items())
            assert tensor_sha(data['qscale'])==tensor_sha(c['pm'].q_scale_recon)
        else:assert original['no_pilot_TCG_LoRA'] and original['stream_SHA256']==info['stream_SHA256']
        frames=[];transitions=[];previous=previous_gt=None;decoder_seconds=0.;begun=time.time();torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for i in range(n):
                assert (None if i==0 else tensor_sha(data['features'][i]))==info['codec_trace'][i]['codec_feature_SHA256']
                if model=='Original':raw=original['values'][i].to(self.device)
                elif i==0:raw=data['original'][0].to(self.device)
                else:
                    torch.cuda.synchronize();t=time.time();raw=reconstruct(c,data['features'][i],info['actual_qps'][i]);torch.cuda.synchronize();decoder_seconds+=time.time()-t
                out=(raw.float().clamp(-1,1)+1)/2;out=out[...,:h,:w]
                gt=(data['GT'][i].permute(2,0,1)[None].float()/255).to(self.device)
                edge=sobel(out).abs();gt_edge=sobel(gt).abs();strength=float(edge.mean());gt_strength=float(gt_edge.mean())
                frames.append(dict(frame=i,source_frame=video['source_frame_indices'][i],actual_QP=info['actual_qps'][i],real_bits=info['codec_trace'][i]['real_bits'],source_RGB_SHA256=info['codec_trace'][i]['source_RGB_SHA256'],RGB_SHA256=tensor_sha(out),pixel_SSE=float((out-gt).square().sum()),LPIPS_Alex=float(self.alex(out,gt,normalize=True).mean()),LPIPS_VGG=float(self.vgg(out,gt,normalize=True).mean()),DISTS=float(self.dists(out,gt).mean()),Sobel_edge_strength=strength,GT_Sobel_edge_strength=gt_strength,edge_relative_GT_bias=strength/gt_strength-1 if gt_strength else None,local16_energy_abs_bias=float((F.avg_pool2d(edge,16,16)-F.avg_pool2d(gt_edge,16,16)).abs().mean())))
                if i:
                    difference=self.flow(previous_gt,gt)-self.flow(previous,out)
                    flo=float(self.flo(previous_gt,previous,difference,normalize=True).mean())
                    rr,ss,valid=temporal_errors(out,previous,gt,previous_gt,data['GT_backward'][i].to(self.device),data['GT_valid'][i].to(self.device))
                    transitions.append(dict(from_frame=i-1,to_frame=i,kind='I_to_P' if i==1 else 'P_to_P',FloLPIPS=flo,RGB_MCResidual=None if rr is None else float(rr),Sobel_MCResidual=None if ss is None else float(ss),valid_fraction=valid,valid_pixels=int(data['GT_valid'][i].sum())))
                previous,previous_gt=out,gt
                dump(ROOT/'evaluation/current_progress.json',dict(model=model,step=step,sequence=video['sequence'],QP=q,frame=i,time=time.time()))
        assert sum(x['real_bits'] for x in frames)==info['real_bytes']*8
        if c:
            assert not c['pm'].dpb;assert before==tensor_hash(c['pm'].state_dict().items());assert_reset(c)
        def avg(values,key):
            v=[x[key] for x in values if x[key] is not None];return sum(v)/len(v) if v else None
        result=dict(model=model,step=step,dataset=video['dataset'],sequence=video['sequence'],QP=q,frames=n,real_bytes=info['real_bytes'],bpp=info['bpp'],stream_SHA256=info['stream_SHA256'],codec_trace_SHA256=info['codec_trace_SHA256'],source_SHA256=video['source_sha256'],PSNR=-10*math.log10(max(sum(x['pixel_SSE'] for x in frames)/(n*h*w*3),1e-15)))
        for k in ['LPIPS_Alex','LPIPS_VGG','DISTS','Sobel_edge_strength','GT_Sobel_edge_strength','edge_relative_GT_bias','local16_energy_abs_bias']:result[k]=avg(frames,k)
        for k in ['FloLPIPS','RGB_MCResidual','Sobel_MCResidual']:
            result[k]=avg(transitions,k);result[k+'_PtoP']=avg(transitions[1:],k);result[k+'_ItoP']=transitions[0][k]
        result.update(decoder_seconds=decoder_seconds,decoder_reference_fps=63/decoder_seconds if decoder_seconds else None,total_seconds=time.time()-begun,peak_allocated_bytes=torch.cuda.max_memory_allocated(),reference_path=True)
        assert all(result[k] is None or math.isfinite(result[k]) for k in METRICS)
        files={}
        for name,values in [('frames',frames),('transitions',transitions)]:
            path=root/name/f'{key}.csv';write_csv(path,values);files[str(path.relative_to(ROOT))]=sha(path)
        record=dict(status='PASS',identity=identity,result=result,record_SHA256=files,weights=self.hashes,real_RANS=True,DPB_unchanged=True,original_independent=model=='Original',RGB_clamp='FP32 clamp[-1,1] then unit[0,1]',I_frame_precision='fixed canonical FP16 codec I-frame, same for all states')
        dump(dest,record);del c;torch.cuda.empty_cache();return record


def point_worker(index,cache,original_path,through=500):
    import torch
    data=torch.load(cache,map_location='cpu',weights_only=True,mmap=True)
    original=torch.load(original_path,map_location='cpu',weights_only=True,mmap=True)
    scorer=Scorer()
    states=[('Original',0),('B300',0)]+[(g,s) for g in GROUPS for s in [0,250,500] if s<=through]
    for model,step in states:scorer.point(data,model,step,original)


def main(through=500):
    from cache_codec import cache_paths
    videos=load(ROOT/'manifests/development_videos.json')
    assert len(videos)==32 and {v['dataset'] for v in videos}=={'ULong_development_32'}
    for index,(v,q) in enumerate((v,q) for v in videos for q in [0,5,9]):
        key,path,rp=cache_paths(v,q,False)
        req=ROOT/'cache/requests/evaluation.json';dump(req,[dict(row=v,qp=q,training=False)])
        subprocess.run([PYTHON,'cache_job.py',str(req)],cwd=ROOT,check=True)
        receipt=load(rp);assert sha(path)==receipt['cache_sha256']
        original=ROOT/'cache/original_FP32'/f'{key}.pt'
        if not original.exists():subprocess.run([PYTHON,'original_reconstruction.py',str(path),str(original)],cwd=ROOT,check=True)
        subprocess.run([PYTHON,'evaluate.py','--point',str(index),'--cache',str(path),'--original',str(original),'--through',str(through)],cwd=ROOT,check=True)
        if path.is_relative_to(ROOT):path.unlink()
        if original.is_relative_to(ROOT):original.unlink()
        dump(ROOT/'evaluation/progress.json',dict(status='RUNNING',completed_common_points=index+1,total=96,time=time.time()))
        subprocess.run([PYTHON,'results.py'],cwd=ROOT,check=True)
    dump(ROOT/('evaluation/completion.json' if through==500 else f'evaluation/through_{through}.json'),dict(status='PASS',points_per_state=96,states=14 if through==500 else 10,development_only=True))


if __name__=='__main__':
    env();p=argparse.ArgumentParser();p.add_argument('--point',type=int);p.add_argument('--cache');p.add_argument('--original');p.add_argument('--through',type=int,choices=[250,500],default=500);a=p.parse_args()
    if a.point is None:main(a.through)
    else:point_worker(a.point,a.cache,a.original,a.through)
