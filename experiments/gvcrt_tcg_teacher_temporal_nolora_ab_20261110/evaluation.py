"""Uniform GPU precision, cached independent Original trajectory, two LPIPS variants."""
from common_s2 import *
from cache_codec import cache_paths
from model_context import backend,assert_reset
from experiment_model import construct,restore,reconstruct
from temporal import temporal_errors
from pipeline import wait_rows
import torch,argparse

METRICS=['PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','RGB_MCResidual_PtoP','Sobel_MCResidual_PtoP','RGB_MCResidual_ItoP','Sobel_MCResidual_ItoP']

class Evaluator:
    def __init__(self,rank):
        self.rank=rank;self.device=torch.device('cuda',rank);self.contexts={};self.networks=None
    def setup(self):
        if self.networks is not None:return
        from evaluate import metrics_setup
        from quality import lpips_model,metric_audit
        quality,flow,perceptual,legacy=metrics_setup(self.rank)
        vgg=lpips_model('vgg',self.device);self.networks=(quality[0],vgg,quality[1],flow,perceptual)
        self.hashes=dict(LPIPS_Alex=tensor_hash(quality[0].state_dict().items()),LPIPS_VGG=tensor_hash(vgg.state_dict().items()),DISTS=tensor_hash(quality[1].state_dict().items()),PWC=tensor_hash(flow.state_dict().items()),FloLPIPS=tensor_hash(perceptual.state_dict().items()))
        self.metric_id=digest(dict(code={n:sha(ROOT/n) for n in ['evaluation.py','quality.py','temporal.py']},weights=self.hashes,policy=metric_audit(),precision=DEPLOYMENT))
    def context(self,model,step):
        if model=='Original':return None
        name=model;cp=None
        if model.endswith('_selected'):
            group=GROUPS[0] if model=='G0_selected' else GROUPS[1];choice=load(ROOT/'selection.json')['groups'][group];name=group;step=choice['step'];cp=Path(choice['checkpoint']);assert sha(cp)==choice['checkpoint_SHA256']
        elif name in GROUPS and step:cp=ROOT/name/f'training/checkpoints/step_{step:04d}.pt'
        key=(model,step)
        if key not in self.contexts:
            c=construct(name,False,self.rank)
            if cp is not None:restore(c,torch.load(cp,map_location='cpu',weights_only=True),optimizer=False)
            self.contexts[key]=c
        return self.contexts[key]
    def point(self,v,q,model,step,phase,data,receipt):
        if model.endswith('_selected'):
            group=GROUPS[0] if model=='G0_selected' else GROUPS[1];step=load(ROOT/'selection.json')['groups'][group]['step']
        self.setup();info=data['info'];h,w=info['native_shape'];n=info['frames'];assert n==64
        assert info['source_row']==v.get('source_row',v)
        cp_hash=sha(ROOT/'checkpoints/B300_noLoRA_step0.pt')
        if model=='Original':cp_hash=load(ORIGINAL_EVAL/'protocol.json')['checkpoint_sha256']
        elif model=='B300_original_reference':cp_hash=B300_SHA
        elif model.endswith('_selected'):
            group=GROUPS[0] if model=='G0_selected' else GROUPS[1];cp_hash=load(ROOT/'selection.json')['groups'][group]['checkpoint_SHA256']
        elif model in GROUPS and step:cp_hash=sha(ROOT/model/f'training/checkpoints/step_{step:04d}.pt')
        identity=dict(model=model,step=step,checkpoint_SHA256=cp_hash,metric_id=self.metric_id,source_frames=v['source_frame_indices'],source_SHA256=v['source_sha256'],source_RGB_hashes=v['frame_rgb_sha256'],stream_SHA256=info['stream_SHA256'],real_bytes=info['real_bytes'],codec_trace_SHA256=digest(info['codec_trace']),QP=q,precision=DEPLOYMENT)
        key=f'{v["dataset"]}__{v["sequence"]}_qp{q}';root=ROOT/'evaluation'/phase/model/f'step_{step:04d}';dest=root/'points'/f'{key}.json'
        if dest.exists():
            a=load(dest);assert a['identity']==identity and a['status']=='PASS'
            for f,hsh in a['record_SHA256'].items():assert sha(ROOT/f)==hsh
            return a
        c=self.context(model,step);assert sha(info['stream_path'])==info['stream_SHA256'] and Path(info['stream_path']).stat().st_size==info['real_bytes']
        if c:
            assert_reset(c);before=tensor_hash(c['pm'].state_dict().items());assert tensor_sha(c['pm'].q_scale_recon.cpu())==tensor_sha(data['qscale'])
        lp_alex,lp_vgg,dists,flow,perceptual=self.networks;frames=[];transitions=[];previous=previous_gt=None;start=time.time();reuse=legacy_components(v,q,model,info,self.hashes,phase)
        with torch.no_grad():
            for i in range(64):
                feature=data['features'][i]
                assert (None if i==0 else tensor_sha(feature))==info['codec_trace'][i]['codec_feature_SHA256']
                raw=data['original'][i].to(self.device) if i==0 or c is None else reconstruct(c,feature,info['actual_qps'][i])
                out=(raw.float().clamp(-1,1)+1)/2;out=out[...,:h,:w];gt=(data['GT'][i].permute(2,0,1)[None].float()/255).to(self.device)
                out_hash=tensor_sha(out);eligible=reuse is not None and out_hash==reuse['frames'][i]['RGB_hash']
                if reuse is not None:assert eligible,('LEGACY_REUSE_RECONSTRUCTION_MISMATCH',model,key,i)
                lpv=float(lp_vgg(out,gt,normalize=True).mean())
                if eligible:
                    a=reuse['frames'][i];lpa=float(a['LPIPS']);ds=float(a['DISTS']);sse=float(a['pixel_SSE'])
                else:lpa=float(lp_alex(out,gt,normalize=True).mean());ds=float(dists(out,gt).mean());sse=float((out-gt).square().sum())
                frames.append(dict(frame=i,source_frame=v['source_frame_indices'][i],actual_QP=info['actual_qps'][i],real_bits=info['codec_trace'][i]['real_bits'],source_RGB_SHA256=info['codec_trace'][i]['source_RGB_SHA256'],RGB_SHA256=out_hash,LPIPS_Alex=lpa,LPIPS_VGG=lpv,DISTS=ds,pixel_SSE=sse))
                if i:
                    if eligible:flo=float(reuse['transitions'][i-1]['FloLPIPS'])
                    else:
                        difference=flow(previous_gt,gt)-flow(previous,out);flo=float(perceptual(previous_gt,previous,difference,normalize=True).mean())
                    rr,ss,vf=temporal_errors(out,previous,gt,previous_gt,data['GT_backward'][i].to(self.device),data['GT_valid'][i].to(self.device))
                    transitions.append(dict(from_frame=i-1,to_frame=i,kind='I_to_P' if i==1 else 'P_to_P',FloLPIPS=flo,RGB_MCResidual=None if rr is None else float(rr),Sobel_MCResidual=None if ss is None else float(ss),valid_fraction=vf,valid_pixels=int(data['GT_valid'][i].sum()),status='MISSING' if rr is None else 'VALID',GT_backward_SHA256=tensor_sha(data['GT_backward'][i]),GT_valid_SHA256=tensor_sha(data['GT_valid'][i])))
                previous,previous_gt=out,gt
                dump(root/f'progress_rank{self.rank}.json',dict(sequence=v['sequence'],QP=q,frame=i,time=time.time()))
        assert sum(r['real_bits'] for r in frames)==info['real_bytes']*8
        if c:assert_reset(c);assert before==tensor_hash(c['pm'].state_dict().items())
        def avg(items,k):
            values=[r[k] for r in items if r[k] is not None];return sum(values)/len(values) if values else None
        result=dict(model=model,step=step,phase=phase,dataset=v['dataset'],sequence=v['sequence'],QP=q,frames=64,real_bytes=info['real_bytes'],bpp=info['bpp'],stream_SHA256=info['stream_SHA256'],codec_trace_SHA256=digest(info['codec_trace']),source_SHA256=v['source_sha256'],source_RGB_SHA256=digest(v['frame_rgb_sha256']),PSNR=-10*math.log10(max(sum(r['pixel_SSE'] for r in frames)/(64*h*w*3),1e-15)),LPIPS_Alex=avg(frames,'LPIPS_Alex'),LPIPS_VGG=avg(frames,'LPIPS_VGG'),DISTS=avg(frames,'DISTS'),FloLPIPS=avg(transitions,'FloLPIPS'),FloLPIPS_PtoP=avg(transitions[1:],'FloLPIPS'),FloLPIPS_ItoP=transitions[0]['FloLPIPS'],RGB_MCResidual=avg(transitions,'RGB_MCResidual'),Sobel_MCResidual=avg(transitions,'Sobel_MCResidual'),RGB_MCResidual_PtoP=avg(transitions[1:],'RGB_MCResidual'),Sobel_MCResidual_PtoP=avg(transitions[1:],'Sobel_MCResidual'),RGB_MCResidual_ItoP=transitions[0]['RGB_MCResidual'],Sobel_MCResidual_ItoP=transitions[0]['Sobel_MCResidual'],valid_temporal_transitions=sum(t['status']=='VALID' for t in transitions),missing_temporal_transitions=sum(t['status']=='MISSING' for t in transitions))
        for k in METRICS:assert result[k] is None or math.isfinite(result[k]),('NONFINITE_METRIC',k,key)
        files={}
        for folder,rows in [('frames',frames),('transitions',transitions)]:
            p=root/folder/f'{key}.csv';write_csv(p,rows);files[str(p.relative_to(ROOT))]=sha(p)
        record=dict(status='PASS',identity=identity,result=result,record_SHA256=files,real_RANS=True,actual_I_frame_bits_included=True,codec_reference='independently decoded Original trajectory, immutable hashed feature inputs; no candidate image ever enters codec',input_feature_trace_checked=True,reconstruction_uses_GT=False,reconstruction_uses_Teacher=False,GT_flow_only_used_by_metrics=True,metric_weights=self.hashes,metric_precision='FP32 RGB first; uniform inherited GPU deployment arithmetic',original_baseline_source=dict(commit=ORIGINAL_COMMIT,I_P_SHA256=load(ORIGINAL_EVAL/'protocol.json')['checkpoint_sha256'],no_v6_TCG_LoRA_B300_weights_in_Original_codec=True) if model=='Original' else None,components_reused=[] if reuse is None else ['PSNR','LPIPS_Alex','DISTS','FloLPIPS'],components_new=['LPIPS_VGG','RGB_MCResidual','Sobel_MCResidual'] if reuse else list(METRICS),reuse_source=None if reuse is None else reuse['source'],physical_GPU=physical_gpu(self.rank),seconds=time.time()-start)
        dump(dest,record);return record

def legacy_components(v,q,model,info,hashes,phase):
    if model=='Original' and phase=='final':
        p=REFERENCE_EXPERIMENT/f'evaluation/final_original/points/{v["domain"]}_{v["sequence"]}_qp{q}.json';style='old'
    elif model=='B300_original_reference' and phase=='development':p=PARENT_EXPERIMENT/f'evaluation/B/step_0300/points/{v["sequence"]}_qp{q}.json';style='ABCD'
    else:return None
    if not p.exists():return None
    a=load(p);r=a['result']
    if a['status']!='PASS':return None
    if (r.get('bitstream_sha256',r.get('bitstream_SHA256'))!=info['stream_SHA256'] or r['real_bytes']!=info['real_bytes'] or r['bpp']!=info['bpp']):return None
    root=p.parent.parent;fname=p.with_suffix('.csv').name
    if style=='old':
        assert not a['TCG_modules_present'] and not a['LoRA_modules_present'] and a['candidate']=='original' and a['real_RANS']
        assert a['metric_hashes']['quality']==[hashes['LPIPS_Alex'],hashes['DISTS']] and a['metric_hashes']['FloLPIPS']==hashes['FloLPIPS'] and a['metric_hashes']['flow']==hashes['PWC']
        for name,h in a['record_file_SHA256'].items():assert sha(root/name/fname)==h
    else:
        assert a['identity']['checkpoint_SHA256']==B300_SHA
        for name,h in a['record_SHA256'].items():assert sha(PARENT_EXPERIMENT/name)==h
        assert a['metric_network_hashes']['quality']==[hashes['LPIPS_Alex'],hashes['DISTS']] and a['metric_network_hashes']['FloLPIPS']==hashes['FloLPIPS'] and a['metric_network_hashes']['flow']==hashes['PWC']
    fr=list(csv.DictReader((root/'frames'/fname).open()));tr=list(csv.DictReader((root/'transitions'/fname).open()));assert len(fr)==64 and len(tr)==63
    for i,x in enumerate(fr):
        assert x.get('source_frame_RGB_sha256',x.get('source_RGB_SHA256'))==v['frame_rgb_sha256'][i]
        x['RGB_hash']=x.get('decoded_RGB_tensor_sha256',x.get('RGB_tensor_SHA256'))
    return dict(frames=fr,transitions=tr,source=dict(path=str(p),SHA256=sha(p),verified_equal_input_color_frames_stream_precision_and_metric_weights=True))

def run(models,step,phase,rank,shard,shards):
    env();torch.cuda.set_device(rank);videos=load(ROOT/f'manifests/{"development" if phase=="development" else "test"}_videos.json')
    if phase=='final':assert load(ROOT/'selection.json')['status']=='FROZEN'
    evaluator=Evaluator(rank);jobs=[(v,q) for v in videos for q in [0,5,9]]
    for ix,(v,q) in enumerate(jobs):
        if ix%shards!=shard:continue
        row=v.get('source_row',v)
        while True:
            wait_rows([row],training=False,qp=q);key,path,rp=cache_paths(row,q,False)
            with lock('locks/lease_'+key+'.lock',blocking=True):
                if not path.exists():continue
                proof=load(rp);assert sha(path)==proof['cache_sha256'];data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                for model in models:evaluator.point(v,q,model,step,phase,data,proof)
                del data
                if path.is_relative_to(ROOT):os.utime(path,None)
                break
        dump(ROOT/f'audits/evaluation_worker_{phase}_{rank}_{shard}.json',dict(status='RUNNING',last_job=ix,total_jobs=len(jobs),models=models,step=step,time=time.time()))
    dump(ROOT/f'audits/evaluation_worker_{phase}_{rank}_{shard}.json',dict(status='PASS',models=models,step=step,time=time.time()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--step',type=int,default=0);p.add_argument('--phase',choices=['development','final'],default='development');p.add_argument('--rank',type=int,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.models.split(','),a.step,a.phase,a.rank,a.shard,a.shards)
