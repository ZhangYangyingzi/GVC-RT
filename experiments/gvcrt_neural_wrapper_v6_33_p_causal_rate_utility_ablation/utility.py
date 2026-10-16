"""Post-RANS-decoding channel interventions with fixed normal decoder context."""
from evaluate import *
import io

METRICS=('LPIPS','DISTS','FloLPIPS')

def select_pairs():
    points=[load(p)['result'] for p in sorted((ROOT/'evaluation/points').glob('*.json'))]
    pairs=[];coverage=[]
    # Pure P primary; independently adapted B/D secondary. Selection uses rate only.
    for a,b in [(BRANCHES[0],BRANCHES[2]),(BRANCHES[1],BRANCHES[3])]:
        for d,vs in load(V32/'eval_manifest.json').items():
            candidates=[]
            for v in vs:
                no=[r for r in points if r['method']==a and r['dataset']==d and r['video']==v['video_index']]
                pp=[r for r in points if r['method']==b and r['dataset']==d and r['video']==v['video_index']]
                for x in no:
                    for y in pp:
                        gap=abs(y['bpp']/x['bpp']-1)
                        if gap<=.02:candidates.append((gap,v['video_index'],x['QP'],y['QP'],x,y))
            coverage.append(dict(dataset=d,noP=a,P=b,candidates_within_1pct=sum(x[0]<=.01 for x in candidates),candidates_within_2pct=len(candidates),status='available' if candidates else 'NO_PAIR_WITHIN_2PCT'))
            if candidates:
                # One closest actual-rate pair per dataset/comparison, all channels.
                gap,i,qa,qb,x,y=min(candidates,key=lambda t:t[:4])
                pairs.append(dict(dataset=d,video=i,noP=a,P=b,noP_QP=qa,P_QP=qb,noP_bpp=x['bpp'],P_bpp=y['bpp'],relative_gap=gap,tier='primary_1pct' if gap<=.01 else 'secondary_2pct',noP_point=x,P_point=y))
    dump('utility/matched_pair_plan.json',dict(selection='minimum actual rate gap; lexicographic tie break; no quality used',frame=32,coverage=coverage,pairs=pairs))
    write('evaluation/equal_rate_pairs.csv',[{k:v for k,v in p.items() if not k.endswith('_point')} for p in pairs])
    return pairs

def run_pair(pair,which,gpu,probe=False):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
    import torch
    torch.set_num_threads(2)
    p=pair[which+'_point'];method=pair[which];name=f'{pair["dataset"]}_{pair["video"]}_{method}_{p["QP"]}'
    folder='sanity' if probe else 'utility'
    if probe:name='utility_probe_'+name
    out=ROOT/folder/f'{name}.json'
    if out.exists():return
    e,rt=runtime_for(method);torch.cuda.reset_peak_memory_stats();v=next(v for v in load(V32/'eval_manifest.json')[p['dataset']] if v['video_index']==p['video'])
    sys.path.insert(0,str(V32));from adapter32 import frames_for
    frames=frames_for(v);im,pm=rt.models(1088,1920);pm.clear_dpb();pm.set_curr_poc(0)
    before=e.core.compression_hash(im,pm);stream=io.BytesIO(Path(p['bitstream_path']).read_bytes());helper=e.eco.SPSHelper();capt=[];baseline=[];args=None
    from channel_rate import attach,finish
    rate_state,detach_rate=attach(pm)
    h=pm.dec.register_forward_pre_hook(lambda m,a:capt.append(tuple(t.detach().clone() for t in a)))
    zcap=[];hz=pm.hyper_dec.register_forward_pre_hook(lambda m,a:zcap.append((list(a[0].shape),str(a[0].dtype))))
    with torch.inference_mode():
        for idx in range(33):
            rate_state['enabled']=idx==32
            header=e.eco.read_header(stream)
            while header['nal_type']==e.eco.NalType.NAL_SPS:
                helper.add_sps_by_id(e.eco.read_sps_remaining(stream,header['sps_id']));header=e.eco.read_header(stream)
            sp=helper.get_sps_by_id(header['sps_id']);q,bits=e.eco.read_ip_remaining(stream)
            rate_state['qp']=q
            if idx==0:
                decoded=im.decompress(bits,sp,q);pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,decoded['x_hat'])
            else:decoded=pm.decompress(bits,sp,q)
            if idx>=31:baseline.append(e.core.unit(decoded['x_hat'],1080,1920).float())
            if idx==32:args=capt[-1];qa=q;zshape,zdtype=zcap[-1]
            capt.clear();zcap.clear()
    h.remove();hz.remove()
    detach_rate();rate=finish(rate_state)
    gt0=frames[31].cuda();gt=frames[32].cuda();prev,normal=baseline
    flow,perceptual,_=rt.metric_models
    with torch.inference_mode():
        gtflow=flow(gt0,gt)
        def measure(rec):
            delta=gtflow-flow(prev,rec)
            mag=delta.square().sum(1,keepdim=True).sqrt().sum()
            assert torch.isfinite(mag) and mag>0
            return dict(LPIPS=float(rt.quality[0](rec,gt,normalize=True).mean()),DISTS=float(rt.quality[1](rec,gt).mean()),FloLPIPS=float(perceptual(gt0,prev,delta,normalize=True).item()))
        # FloLPIPS historical metric is evaluated at the previous target/output,
        # with flow discrepancy caused by the intervened current frame.
        base=measure(normal)
        latent,context,qd=args;qr=pm.q_scale_recon[qa:qa+1]
        def synth(y):return e.core.unit(pm.recon_generation_net(pm.dec(y,context,qd),qr),1080,1920).float()
        replay=synth(latent);err=float((replay-normal).abs().max());assert err==0,err
        rows=[]
        for c in range(min(2,latent.shape[1]) if probe else latent.shape[1]):
            for intervention in ('zero','spatial_mean','attenuate_0.95'):
                y=latent.clone()
                if intervention=='zero':y[:,c]=0
                elif intervention=='spatial_mean':y[:,c]=latent[:,c].mean(dim=(-2,-1),keepdim=True)
                else:y[:,c]*=.95
                values=measure(synth(y))
                rows.append(dict(channel=c,intervention=intervention,**values,**{'I_'+m:values[m]-base[m] for m in METRICS}))
    assert before==e.core.compression_hash(im,pm)
    for r in rows:
        r['R_est']=rate['estimated_channel_bits'][r['channel']]
        for metric in METRICS:r['eta_'+metric]=r['I_'+metric]/max(r['R_est'],1e-9)
    result=dict(status='PASS',method=method,dataset=p['dataset'],video=p['video'],QP=p['QP'],actual_total_bpp=p['bpp'],actual_total_bytes=p['real_bytes'],bitstream_path=p['bitstream_path'],bitstream_sha256=sha(p['bitstream_path']),frame=32,transition=[31,32],channels=latent.shape[1],latent_shape=list(latent.shape),dtype=str(latent.dtype),baseline=base,replay_maxabs=err,rows=rows,rate=rate,context='baseline context/previous output held fixed; no intervention propagated into subsequent frame references',neutral='zero is not guaranteed a conditional neutral point for dequantized means; spatial_mean is the preferred structure-removal control',compression_frozen=True,per_channel_actual_rate_available=False,estimated_channel_rate_available=True)
    result['hyperlatent_shape']=zshape;result['hyperlatent_dtype']=zdtype
    result['actual_frame32_payload_bits']=len(bits)*8
    result['matched_rate_tier']=pair.get('tier','PROBE_ONLY');result['matched_relative_gap']=pair.get('relative_gap')
    result['probe_only']=probe
    result['CUDA_peak_allocated_MiB']=torch.cuda.max_memory_allocated()/2**20
    dump(folder+'/'+name+'.json',result)
    print('UTILITY PASS',name,flush=True)

if __name__=='__main__':
    import argparse,traceback
    ap=argparse.ArgumentParser();ap.add_argument('--select',action='store_true');ap.add_argument('--probe',action='store_true');ap.add_argument('--pair',type=int);ap.add_argument('--which',choices=('noP','P'));ap.add_argument('--gpu',type=int,choices=(4,5,6,7),default=7);a=ap.parse_args()
    try:
        if a.probe:
            r=load(ROOT/'evaluation/points'/f'ulong_0_{BRANCHES[0]}_0.json')['result']
            run_pair(dict(dataset='ulong',video=0,noP=BRANCHES[0],noP_point=r),'noP',a.gpu,True)
        elif a.select:select_pairs()
        else:run_pair(load(ROOT/'utility/matched_pair_plan.json')['pairs'][a.pair],a.which,a.gpu)
    except Exception:
        dump(f'logs/utility_failure_{os.getpid()}.json',dict(traceback=traceback.format_exc()));raise
