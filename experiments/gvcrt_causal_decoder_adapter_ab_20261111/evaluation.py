"""ULong development only: real canonical RANS, independent Original cache, fixed metrics."""
from common_s2 import *
import torch
import torch.nn.functional as F
from temporal import temporal_errors,sobel

METRICS=['PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS','FloLPIPS_PtoP','FloLPIPS_ItoP',
    'RGB_MCResidual','Sobel_MCResidual','RGB_MCResidual_PtoP','Sobel_MCResidual_PtoP',
    'RGB_MCResidual_ItoP','Sobel_MCResidual_ItoP','Sobel_edge_strength','edge_relative_GT_bias',
    'local16_energy_abs_bias','local16_energy_signed_bias']


def point_path(v,q,model,step):
    return ROOT/f'evaluation/development/{model}/step_{step:04d}/points/{v["dataset"]}__{v["sequence"]}_qp{q}.json'


def historical(v,q,model,info,weights):
    if model=='B300':
        p=H_SOURCE/f'evaluation/development/B300/step_0000/points/{v["dataset"]}__{v["sequence"]}_qp{q}.json'
        if not p.exists():return None
        a=load(p);idn=a['identity'];r=a['result']
        assert a['status']=='PASS' and idn['checkpoint_SHA256']==B300_SHA
        assert idn['source_frames']==v['source_frame_indices'] and idn['source_SHA256']==v['source_sha256']
        assert idn['source_RGB_hashes']==v['frame_rgb_sha256'] and a['metric_weights']==weights
        assert idn['QP']==q and idn['precision']=='inherited FP16 codec/Bridge/backbone; FP32 TCG and LoRA; RGB FP32-first'
        assert r['stream_SHA256']==info['stream_SHA256'] and r['real_bytes']==info['real_bytes'] and r['bpp']==info['bpp']
        files={}
        for f,h in a['record_SHA256'].items():
            if '/frames/' in f or '/transitions/' in f:
                path=H_SOURCE/f;assert sha(path)==h;files['frames' if '/frames/' in f else 'transitions']=path
        fr=list(csv.DictReader(files['frames'].open()));tr=list(csv.DictReader(files['transitions'].open()))
        assert len(fr)==64 and len(tr)==63
        return dict(frames=fr,transitions=tr,source=str(p),source_SHA256=sha(p),style='H_B300')
    if model=='Original':
        # The prior independent Original source includes these same development sources.
        p=REFERENCE_EXPERIMENT/f'evaluation/final_original/points/{v["domain"]}_{v["sequence"]}_qp{q}.json'
        if not p.exists():return None
        a=load(p);r=a['result']
        if not (a['status']=='PASS' and r['source_sha256']==v['source_sha256'] and a['source_frame_indices']==v['source_frame_indices']):return None
        if r['bitstream_sha256']!=info['stream_SHA256'] or r['real_bytes']!=info['real_bytes'] or r['bpp']!=info['bpp']:return None
        assert a['candidate']=='original' and not a['TCG_modules_present'] and not a['LoRA_modules_present']
        assert a['real_RANS'] and a['independent_decode']
        assert a['metric_hashes']['quality']==[weights['LPIPS_Alex'],weights['DISTS']]
        assert a['metric_hashes']['flow']==weights['PWC'] and a['metric_hashes']['FloLPIPS']==weights['FloLPIPS']
        files={}
        for name in ('frames','transitions'):
            f=p.parent.parent/name/p.with_suffix('.csv').name;assert sha(f)==a['record_file_SHA256'][name];files[name]=f
        fr=list(csv.DictReader(files['frames'].open()));tr=list(csv.DictReader(files['transitions'].open()))
        for i,f in enumerate(fr):assert f['source_frame_RGB_sha256']==v['frame_rgb_sha256'][i]
        return dict(frames=fr,transitions=tr,source=str(p),source_SHA256=sha(p),style='independent_Original')
    return None


class Evaluator:
    def __init__(self):
        from evaluate import metrics_setup
        from quality import lpips_model,metric_audit
        quality,flow,perceptual,_=metrics_setup(0)
        vgg=lpips_model('vgg',torch.device('cuda',0))
        self.networks=(quality[0],vgg,quality[1],flow,perceptual)
        self.weights={k:tensor_hash(m.state_dict().items()) for k,m in zip(('LPIPS_Alex','LPIPS_VGG','DISTS','PWC','FloLPIPS'),self.networks)}
        self.metric_id=digest(dict(version='causal_ab_development_metrics_v1',weights=self.weights,policy=metric_audit(),temporal=sha(ROOT/'temporal.py'),precision=DEPLOYMENT))
        dump(ROOT/f'audits/metric_identity_GPU{physical_gpu(0)}.json',dict(metric_id=self.metric_id,weights=self.weights,policy=metric_audit()))
        self.contexts={}

    def context(self,model,step):
        from experiment_model import construct,restore
        if model=='Original':return None
        key=(model,step)
        if key not in self.contexts:
            c=construct(model,False,0,with_adapter=model!='B300')
            if model in GROUPS:restore(c,torch.load(ROOT/model/f'training/checkpoints/step_{step:04d}.pt',map_location='cpu',weights_only=True))
            self.contexts[key]=c
        return self.contexts[key]

    def point(self,v,q,model,step,data):
        from experiment_model import reconstruct
        from model_context import assert_reset
        info=data['info'];h,w=info['native_shape'];assert info['frames']==64
        assert info['source_row']==v.get('source_row',v) and info['external_QP']==q
        assert info['base_receiver_no_TCG_LoRA_Router'] and info['original_commit']==ORIGINAL_COMMIT
        assert info['real_RANS'] and info['independent_decode']
        assert sha(info['stream_path'])==info['stream_SHA256']
        assert Path(info['stream_path']).stat().st_size==info['real_bytes']
        assert info['bpp']==info['real_bytes']*8/(64*h*w)
        cp=load(ROOT/'protocol.json')['original'] if model=='Original' else B300_SHA if model=='B300' else sha(ROOT/model/f'training/checkpoints/step_{step:04d}.pt')
        identity=dict(model=model,step=step,checkpoint_SHA256=cp,metric_id=self.metric_id,source_frames=v['source_frame_indices'],
            source_SHA256=v['source_sha256'],source_RGB_hashes=v['frame_rgb_sha256'],QP=q,
            stream_SHA256=info['stream_SHA256'],real_bytes=info['real_bytes'],actual_qps=info['actual_qps'],precision=DEPLOYMENT)
        dest=point_path(v,q,model,step);root=dest.parent.parent;key=dest.stem
        if dest.exists():
            a=load(dest);assert a['status']=='PASS' and a['identity']==identity
            for f,hashval in a['record_SHA256'].items():assert sha(ROOT/f)==hashval
            return a
        c=self.context(model,step)
        if c:assert_reset(c);before=tensor_hash(c['pm'].state_dict().items())
        lp,vgg,dists,flow,perceptual=self.networks;fr=[];tr=[];prev=prevgt=None;start=time.time()
        reuse=historical(v,q,model,info,self.weights)
        with torch.no_grad():
            for i in range(64):
                assert info['codec_trace'][i]['source_RGB_SHA256']==v['frame_rgb_sha256'][i]
                f=data['features'][i]
                assert (None if i==0 else tensor_sha(f))==info['codec_trace'][i]['codec_feature_SHA256']
                raw=data['original'][i].to('cuda:0') if i==0 or c is None else reconstruct(c,f,info['actual_qps'][i])
                out=((raw.float().clamp(-1,1)+1)/2)[...,:h,:w]
                gt=(data['GT'][i].permute(2,0,1)[None].float()/255).to('cuda:0')
                rgb_hash=tensor_sha(out);old=reuse['frames'][i] if reuse else None
                if old:
                    expected=old.get('RGB_SHA256',old.get('decoded_RGB_tensor_sha256'))
                    assert rgb_hash==expected,('HISTORICAL_RGB_IDENTITY_MISMATCH',model,key,i)
                lpa=float(old.get('LPIPS_Alex',old.get('LPIPS'))) if old else float(lp(out,gt,normalize=True).mean())
                lpv=float(old['LPIPS_VGG']) if old and 'LPIPS_VGG' in old else float(vgg(out,gt,normalize=True).mean())
                ds=float(old['DISTS']) if old else float(dists(out,gt).mean())
                sse=float(old['pixel_SSE']) if old else float((out-gt).square().sum())
                edge=sobel(out).abs();edgegt=sobel(gt).abs();strength=float(edge.mean());gtstrength=float(edgegt.mean())
                local=F.avg_pool2d(edge,16,16);localgt=F.avg_pool2d(edgegt,16,16)
                fr.append(dict(frame=i,source_frame=v['source_frame_indices'][i],actual_QP=info['actual_qps'][i],
                    real_bits=info['codec_trace'][i]['real_bits'],RGB_SHA256=rgb_hash,source_RGB_SHA256=v['frame_rgb_sha256'][i],
                    LPIPS_Alex=lpa,LPIPS_VGG=lpv,DISTS=ds,pixel_SSE=sse,Sobel_edge_strength=strength,
                    GT_Sobel_edge_strength=gtstrength,edge_relative_GT_bias=strength/gtstrength-1 if gtstrength else None,
                    local16_energy_abs_bias=float((local-localgt).abs().mean()),local16_energy_signed_bias=float((local-localgt).mean())))
                if i:
                    flo=float(reuse['transitions'][i-1]['FloLPIPS']) if reuse else float(perceptual(prevgt,prev,flow(prevgt,gt)-flow(prev,out),normalize=True).mean())
                    rr,ss,vf=temporal_errors(out,prev,gt,prevgt,data['GT_backward'][i].to('cuda:0'),data['GT_valid'][i].to('cuda:0'))
                    tr.append(dict(from_frame=i-1,to_frame=i,kind='I_to_P' if i==1 else 'P_to_P',FloLPIPS=flo,
                        RGB_MCResidual=None if rr is None else float(rr),Sobel_MCResidual=None if ss is None else float(ss),
                        valid_fraction=vf,valid_pixels=int(data['GT_valid'][i].sum()),
                        GT_backward_SHA256=tensor_sha(data['GT_backward'][i]),GT_valid_SHA256=tensor_sha(data['GT_valid'][i])))
                prev,prevgt=out,gt
                dump(root/'active_frame.json',dict(model=model,step=step,sequence=v['sequence'],QP=q,frame=i,PID=os.getpid(),physical_GPU=physical_gpu(0),time=time.time()))
        assert sum(f['real_bits'] for f in fr)==info['real_bytes']*8
        if c:
            assert tensor_hash(c['pm'].state_dict().items())==before
            if model in GROUPS:
                assert c['adapter'].forward_calls>=63 and c['models']['generator'].head.lora.forward_calls>=63
            history_bytes=c['adapter'].history_bytes() if model in GROUPS else 0
            assert_reset(c)
        else:history_bytes=0
        def avg(items,k):
            xs=[r[k] for r in items if r[k] is not None];return sum(xs)/len(xs) if xs else None
        result=dict(model=model,step=step,phase='development',dataset=v['dataset'],sequence=v['sequence'],QP=q,
            frames=64,real_bytes=info['real_bytes'],bpp=info['bpp'],stream_SHA256=info['stream_SHA256'],
            actual_qps=json.dumps(info['actual_qps']),source_SHA256=v['source_sha256'],
            source_RGB_SHA256=digest(v['frame_rgb_sha256']),PSNR=-10*math.log10(max(sum(x['pixel_SSE'] for x in fr)/(64*h*w*3),1e-15)))
        for k in ('LPIPS_Alex','LPIPS_VGG','DISTS','Sobel_edge_strength','GT_Sobel_edge_strength','edge_relative_GT_bias','local16_energy_abs_bias','local16_energy_signed_bias'):result[k]=avg(fr,k)
        for k in ('FloLPIPS','RGB_MCResidual','Sobel_MCResidual'):
            result[k]=avg(tr,k);result[k+'_PtoP']=avg(tr[1:],k);result[k+'_ItoP']=tr[0][k]
        result.update(valid_temporal_transitions=sum(x['RGB_MCResidual'] is not None for x in tr),missing_temporal_transitions=sum(x['RGB_MCResidual'] is None for x in tr))
        assert all(result[k] is None or math.isfinite(result[k]) for k in METRICS)
        if model in GROUPS:
            base=point_path(v,q,'B300',0)
            if base.exists():
                b=load(base)['result'];assert b['stream_SHA256']==result['stream_SHA256'] and b['real_bytes']==result['real_bytes'] and b['bpp']==result['bpp']
        files={}
        for folder,rows in [('frames',fr),('transitions',tr)]:
            p=root/folder/f'{key}.csv';write_csv(p,rows);files[str(p.relative_to(ROOT))]=sha(p)
        a=dict(status='PASS',identity=identity,result=result,record_SHA256=files,real_RANS=True,I_frame_bits_included=True,
            metric_weights=self.weights,metric_version='LPIPS0.1; inherited DISTS/FloLPIPS; '+__import__('temporal').VERSION,
            original_independently_loaded=info['base_receiver_no_TCG_LoRA_Router'],
            original_provenance=load(ROOT/'protocol.json')['original'] if model=='Original' else None,
            model_provenance=c['provenance'] if c else None,reconstruction_uses_GT=False,
            GT_flow='frozen RAFT estimates on GT RGB, not annotated ground-truth flow; evaluation only',
            history_bytes=history_bytes,reuse_source=None if not reuse else {k:reuse[k] for k in ('source','source_SHA256','style')},
            reused_metrics=[] if not reuse else ['PSNR','LPIPS_Alex','DISTS','FloLPIPS']+(['LPIPS_VGG'] if reuse['style']=='H_B300' else []),
            reuse_verified_every_RGB_frame=True if reuse else False,physical_GPU=physical_gpu(0),seconds=time.time()-start,
            evaluation_code_SHA256=sha(ROOT/'evaluation.py'))
        dump(dest,a);event('development_point_complete',model=model,step=step,sequence=v['sequence'],QP=q,seconds=a['seconds'])
        return a
