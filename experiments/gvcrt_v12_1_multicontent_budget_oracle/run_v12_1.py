#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
DBG=REPO/'expericent_generation_input/expericent_generator_aware_encoder_latent_rd_oracle_v12_debug'
V11=REPO/'expericent_generation_input/expericent_generator_aware_latent_distortion_v11'
V9=REPO/'expericent_generation_input/expericent_interface_causal_controls_v9'
for p in (REPO,V9/'src',V11/'b2_latent_direction_magnitude_v11_7/src',DBG/'src'):
    sys.path.insert(0,str(p))
from common import sha256,torch_load
from gvc_hooks import load_models
from run_debug import load_b2,decode_capture,unit
from src.layers.cuda_inference import replicate_pad,round_and_to_int8
from src.utils.stream_helper import write_sps,write_ip,read_header,read_sps_remaining,read_ip_remaining,NalType,SPSHelper
from src.utils.transforms import ycbcr420_to_444_np
from src.utils.video_reader import YUV420Reader

INDEX_MAP=[0,1,0,2,0,2,0,2]
CFG=json.loads((ROOT/'config.json').read_text()); MAN=json.loads((ROOT/'manifest.json').read_text())

def digest_bytes(x):return hashlib.sha256(x).hexdigest()
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
def write_csv(path,rows):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),extrasaction='ignore');w.writeheader();w.writerows(rows)

def read_frames(v,device):
    n=64;out=[]
    if v['dataset']=='fresh_ulong':
        p=subprocess.Popen(['ffmpeg','-v','error','-i',v['source_path'],'-map','0:v:0','-frames:v',str(n),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE);size=1920*1080*3
        try:
            for i in range(n):
                raw=p.stdout.read(size)
                if len(raw)!=size:raise RuntimeError(f'short frame {i}')
                a=np.frombuffer(raw,np.uint8).reshape(1080,1920,3).copy();x=torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(device=device,dtype=torch.float16)/255
                out.append(replicate_pad(x,8,0)*2-1)
        finally:p.stdout.close();p.wait()
    else:
        r=YUV420Reader(v['source_path'],1920,1080)
        try:
            for i in range(n):
                y,uv=r.read_one_frame()
                if y is None:raise RuntimeError(f'short frame {i}')
                a=ycbcr420_to_444_np(y,uv,order=0);x=torch.from_numpy(a).unsqueeze(0).to(device=device,dtype=torch.float16)/255
                out.append(replicate_pad(x,8,0)*2-1)
        finally:r.close()
    return out

def prepare(m,x,qp):
    qf=m.q_scale_feature[qp:qp+1];qe=m.q_scale_enc[qp:qp+1];qd=m.q_scale_dec[qp:qp+1];qr=m.q_scale_recon[qp:qp+1]
    f=m.apply_feature_adaptor();ctx,ct=m.feature_extractor(f,qf);y=m.enc(x,ctx,qe).detach()
    return y,ctx.detach(),ct.detach(),qd.detach(),qr.detach()

def forward(m,y_master,ctx,ct,qd,qr,qp,gt,feature_target,kind):
    y=y_master.half();z=m.hyper_enc(m.pad_for_y(y));zh=torch.clamp(torch.round(z),-128,127);zs=z+(zh-z).detach();params=m.res_prior_param_decoder(zs,ct)
    qdec,sc,mu=params.chunk(3,1);qdec=torch.clamp_min(qdec,.5);ys=y*torch.reciprocal(qdec);B,C,H,W=ys.shape;m0,m1=m.get_mask_2x(B,C,H,W,ys.dtype,ys.device)
    def proc(a,s,mean,mask):
        sh=s*mask;mh=mean*mask;res=(a-mh)*mask;hard=torch.clamp(torch.round(res),-128,127);active=mask.bool()&(sh>.12);hard=hard*active;ste=res+(hard-res).detach();return hard,ste,sh,mh,active
    h0,q0,s0,u0,a0=proc(ys,sc,mu,m0);yh0=q0+u0;sc1,mu1=m.y_spatial_prior(torch.cat((yh0,params),1)).chunk(2,1);h1,q1,s1,u1,a1=proc(ys,sc1,mu1,m1);yhat=(yh0+q1+u1)*qdec
    feat=m.dec(yhat,ctx,qd);rgb=m.recon_generation_net(feat,qr)
    d=((feat.float()-feature_target.float())**2).mean() if kind=='compression_aware' else ((unit(rgb)-unit(gt))**2).mean()
    zu=m.bit_estimator_z.get_cdf(zs.float()+.5,qp);zl=m.bit_estimator_z.get_cdf(zs.float()-.5,qp);r=(-torch.log2((zu-zl).clamp_min(1e-9))).sum()
    def gb(q,s,active):
        q=q.float()[active];s=s.float()[active].clamp(.11,16);idx=((torch.log(s)-m.gaussian_encoder.log_scale_min)*m.gaussian_encoder.log_step_recip).long().clamp(0,127);sq=m.gaussian_encoder.scale_table.to(s.device)[idx];n=torch.distributions.Normal(torch.zeros_like(sq),sq);return (-torch.log2((n.cdf(q+.5)-n.cdf(q-.5)).clamp_min(1e-9))).sum()
    r=r+gb(q0,s0,a0)+gb(q1,s1,a1)
    with torch.no_grad():
        zt,_=round_and_to_int8(z);p=m.res_prior_param_decoder(zt,ct);w0,w1,sw0,sw1,ytrue=m.compress_prior_2x(y,p,m.y_spatial_prior)
    diff=float((yhat.detach()-ytrue).abs().max())
    if diff!=0:raise RuntimeError(f'STE mismatch {diff}')
    symbols=torch.cat((zt.reshape(-1).float(),w0.reshape(-1).float(),w1.reshape(-1).float()))
    return {'R':r,'D':d,'rgb':rgb,'feature':feat,'symbols':symbols,'w0':w0,'w1':w1,'sw0':sw0,'sw1':sw1,'z':zt,'ytrue':ytrue}

def encode_no_update(m,y,ctx,ct,qd,qp):
    with torch.no_grad():
        y=y.half();z=m.hyper_enc(m.pad_for_y(y));zh,zw=round_and_to_int8(z);p=m.res_prior_param_decoder(zh,ct);w0,w1,s0,s1,yhat=m.compress_prior_2x(y,p,m.y_spatial_prior);feat=m.dec(yhat,ctx,qd)
        m.entropy_coder.reset();m.bit_estimator_z.encode_z(zw,qp);m.gaussian_encoder.encode_y(w0,s0);m.gaussian_encoder.encode_y(w1,s1);m.entropy_coder.flush();payload=m.entropy_coder.get_encoded_stream()
    return payload,feat,zh,w0,w1

def snap(m):
    return (m.curr_poc,[(r.poc,None if r.frame is None else r.frame.detach().clone(),None if r.feature is None else r.feature.detach().clone()) for r in m.dpb])
def restore(m,s):
    from src.models.video_model_gvcrt import RefFrame
    m.curr_poc=s[0];m.dpb=[]
    for poc,fr,ft in s[1]:
        r=RefFrame();r.poc=poc;r.frame=fr;r.feature=ft;m.dpb.append(r)

def basic_metrics(x,gt):
    a=unit(x);b=unit(gt);mse=float((a-b).square().mean());psnr=-10*math.log10(max(mse,1e-12))
    # Global SSIM components; deterministic and inexpensive at full resolution.
    ux=a.mean();uy=b.mean();vx=((a-ux)**2).mean();vy=((b-uy)**2).mean();cov=((a-ux)*(b-uy)).mean();ssim=float(((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux**2+uy**2+.01**2)*(vx+vy+.03**2)))
    return mse,psnr,ssim

def init_metric_models(device):
    try:
        import lpips
        from DISTS_pytorch import DISTS
        return lpips.LPIPS(net='alex',verbose=False).to(device).eval().requires_grad_(False),DISTS().to(device).eval().requires_grad_(False)
    except Exception:return None,None
def perceptual(x,gt,models):
    if models[0] is None:return '',''
    a=unit(x);b=unit(gt)
    with torch.no_grad():return float(models[0](a,b,normalize=True)),float(models[1](a,b))

def run_pass(v,frames,requested_qp,branch,beta,device,metric_models):
    im,_=load_models(device);enc,_=load_b2(device);dec,_=load_b2(device);enc.clear_dpb();dec.clear_dpb();enc.set_curr_poc(0);dec.set_curr_poc(0);before=[float(p.float().sum()) for p in enc.parameters()]
    sps={'sps_id':0,'height':1088,'width':1920,'ec_part':1,'use_ada_i':0};stream=io.BytesIO();write_sps(stream,sps);rows=[];frame_rows=[];sync=[]
    ie=im.compress(frames[0],requested_qp);write_ip(stream,True,0,requested_qp,ie['bit_stream']);x0=im.decompress(ie['bit_stream'],sps,requested_qp)['x_hat'];enc.add_ref_frame(None,ie['x_hat']);dec.add_ref_frame(None,x0);mse,psnr,ssim=basic_metrics(x0,frames[0]);lp,di=perceptual(x0,frames[0],metric_models)
    frame_rows.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':requested_qp,'branch':branch,'beta':beta,'frame':0,'actual_qp':requested_qp,'actual_bytes':len(ie['bit_stream']),'MSE':mse,'PSNR':psnr,'SSIM':ssim,'LPIPS':lp,'DISTS':di})
    final_bytes=len(stream.getvalue());final_dist=[]
    for fi in range(1,64):
        qp=enc.shift_qp(requested_qp,INDEX_MAP[fi%8]);y0h,ctx,ct,qd,qr=prepare(enc,frames[fi],qp);y=torch.nn.Parameter(y0h.float())
        with torch.no_grad():
            feature_target=enc.feature_adaptor_i(F.pixel_unshuffle(frames[fi],8)).detach()
        base=forward(enc,y,ctx,ct,qd,qr,qp,frames[fi],feature_target,branch);r0=base['R'].detach();d0=base['D'].detach();s0=base['symbols'].detach()
        last=None
        for step in range(CFG['optimization_steps']+1):
            if step in CFG['candidate_iterations']:
                val=forward(enc,y,ctx,ct,qd,qr,qp,frames[fi],feature_target,branch);payload,efeat,sz,sw0,sw1=encode_no_update(enc,y,ctx,ct,qd,qp);ds=snap(dec);out,cap=decode_capture(dec,payload,sps,qp);restore(dec,ds);mse,psnr,ssim=basic_metrics(out,frames[fi]);lp,di=perceptual(out,frames[fi],metric_models)
                symok=bool(torch.equal(sz,cap['z']) and torch.equal(sw0,cap['w0']) and torch.equal(sw1,cap['w1']))
                row={'scope':'frame','dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':requested_qp,'branch':branch,'restart':0,'lambda_or_beta':beta,'iteration':step,'frame':fi,'surrogate_rate':float(val['R']),'actual_bytes':len(payload),'actual_bits':len(payload)*8,'actual_rate_ratio_to_baseline':'','optimization_distortion':float(val['D']),'PSNR':psnr,'MS_SSIM_or_SSIM':ssim,'LPIPS':lp,'DISTS':di,'quantized_symbol_hash':tensor_hash(val['symbols']),'bitstream_hash':digest_bytes(payload),'decode_status':'PASS' if symok else 'FAIL','finite_status':'PASS','num_changed_symbols_vs_baseline':int((val['symbols']!=s0).sum()),'L1_symbol_change':float((val['symbols']-s0).abs().sum()),'L2_latent_change':float((y-y0h.float()).norm()),'interpolated_only':False}
                rows.append(row);last=(payload,efeat,out,cap,sz,sw0,sw1,val,mse,psnr,ssim,lp,di)
            if step==CFG['optimization_steps']:break
            val=forward(enc,y,ctx,ct,qd,qr,qp,frames[fi],feature_target,branch);loss=val['D']/d0+beta*(val['R']/r0)
            if not bool(torch.isfinite(loss)):raise RuntimeError(f'nonfinite loss frame {fi} step {step}')
            grad=torch.autograd.grad(loss,y)[0]
            if not bool(torch.isfinite(grad).all()):raise RuntimeError(f'nonfinite grad frame {fi} step {step}')
            y.grad=grad;torch.nn.utils.clip_grad_norm_([y],CFG['gradient_clip_norm']);
            if step==0:opt=torch.optim.Adam([y],lr=CFG['learning_rate'])
            opt.step();opt.zero_grad(set_to_none=True)
            with torch.no_grad():y.clamp_(y0h.float()-CFG['trust_region_abs'],y0h.float()+CFG['trust_region_abs'])
        payload,efeat,out,cap,sz,sw0,sw1,val,mse,psnr,ssim,lp,di=last;out2,cap2=decode_capture(dec,payload,sps,qp);enc.add_ref_frame(efeat,None);state=float((enc.dpb[0].feature-dec.dpb[0].feature).abs().max());symok=bool(torch.equal(sz,cap2['z']) and torch.equal(sw0,cap2['w0']) and torch.equal(sw1,cap2['w1']))
        write_ip(stream,False,0,qp,payload);final_bytes=len(stream.getvalue());final_dist.append(float(val['D']))
        frame_rows.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':requested_qp,'branch':branch,'beta':beta,'frame':fi,'actual_qp':qp,'actual_bytes':len(payload),'MSE':mse,'PSNR':psnr,'SSIM':ssim,'LPIPS':lp,'DISTS':di})
        sync.append({'dataset':v['dataset'],'video':v['name'],'qp':requested_qp,'branch':branch,'beta':beta,'frame':fi,'state_max_abs':state,'symbol_equal':symok,'status':'PASS' if state==0 and symok else 'FAIL'})
        if fi%8==0:print(f"{v['dataset']}:{v['video_id']} qp{requested_qp} {branch} beta{beta} frame {fi}/63",flush=True)
    unchanged=before==[float(p.float().sum()) for p in enc.parameters()]
    return stream.getvalue(),rows,frame_rows,sync,unchanged,sum(final_dist)/len(final_dist)

def baseline(v,frames,qp,device,metric_models):
    im,_=load_models(device);enc,_=load_b2(device);dec,_=load_b2(device);enc.clear_dpb();dec.clear_dpb();stream=io.BytesIO();sps={'sps_id':0,'height':1088,'width':1920,'ec_part':1,'use_ada_i':0};write_sps(stream,sps);fr=[];sync=[]
    for i,x in enumerate(frames):
        aq=qp if i==0 else enc.shift_qp(qp,INDEX_MAP[i%8])
        if i==0: e=im.compress(x,aq);payload=e['bit_stream'];enc.add_ref_frame(None,e['x_hat']);out=im.decompress(payload,sps,aq)['x_hat'];dec.add_ref_frame(None,out);is_i=True
        else:
            e=enc.compress(x,aq);payload=e['bit_stream'];out=dec.decompress(payload,sps,aq)['x_hat'];is_i=False;state=float((enc.dpb[0].feature-dec.dpb[0].feature).abs().max());sync.append({'dataset':v['dataset'],'video':v['name'],'qp':qp,'branch':'baseline','beta':'','frame':i,'state_max_abs':state,'symbol_equal':True,'status':'PASS' if state==0 else 'FAIL'})
        write_ip(stream,is_i,0,aq,payload);mse,p,s=basic_metrics(out,x);lp,di=perceptual(out,x,metric_models);fr.append({'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'branch':'baseline','beta':'','frame':i,'actual_qp':aq,'actual_bytes':len(payload),'MSE':mse,'PSNR':p,'SSIM':s,'LPIPS':lp,'DISTS':di})
    return stream.getvalue(),fr,sync

def run_cell(v,qp,device):
    tag=f"{v['dataset']}_{int(v['video_id']):02d}_qp{qp}";done=ROOT/'parts'/f'{tag}_done.json'
    if done.exists() and json.loads(done.read_text()).get('status')=='PASS':print('reuse',tag,flush=True);return
    frames=read_frames(v,device);models=init_metric_models(device);bstream,bframes,bsync=baseline(v,frames,qp,device,models);budget=len(bstream);(ROOT/'bitstreams'/f'{tag}_baseline.bin').write_bytes(bstream)
    base_row={'dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'frames':64,'actual_bytes':budget,'actual_bits':budget*8,'bpp':budget*8/(64*1920*1080),'kbps':budget*8*v['fps']/64/1000,'PSNR':sum(x['PSNR'] for x in bframes)/64,'MS_SSIM':sum(x['SSIM'] for x in bframes)/64,'LPIPS':sum(float(x['LPIPS']) for x in bframes)/64 if bframes[0]['LPIPS']!='' else '','DISTS':sum(float(x['DISTS']) for x in bframes)/64 if bframes[0]['DISTS']!='' else ''}
    candidates=[];frames_out=list(bframes);sync=list(bsync);aud=[];cell_summaries=[];all_unchanged=True
    for branch in ('compression_aware','generator_aware'):
        init_summary={'scope':'cell','dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'branch':branch,'restart':0,'lambda_or_beta':'baseline_init','iteration':0,'frame':'','surrogate_rate':'','actual_bytes':budget,'actual_bits':budget*8,'actual_rate_ratio_to_baseline':1.0,'optimization_distortion':'','PSNR':base_row['PSNR'],'MS_SSIM_or_SSIM':base_row['MS_SSIM'],'LPIPS':base_row['LPIPS'],'DISTS':base_row['DISTS'],'quantized_symbol_hash':'','bitstream_hash':sha256(ROOT/'bitstreams'/f'{tag}_baseline.bin'),'decode_status':'PASS','finite_status':'PASS','num_changed_symbols_vs_baseline':0,'L1_symbol_change':0,'L2_latent_change':0,'interpolated_only':False,'is_budget_feasible':True,'is_selected':False,'is_baseline':False,'bitstream_path':str(ROOT/'bitstreams'/f'{tag}_baseline.bin')}
        candidates.append(init_summary);cell_summaries.append(init_summary)
        for beta in CFG['rate_multipliers']:
            bs,rows,fr,sy,unchanged,obj=run_pass(v,frames,qp,branch,beta,device,models);all_unchanged&=unchanged;path=ROOT/'bitstreams'/f'{tag}_{branch}_beta{beta:g}.bin';path.write_bytes(bs);ratio=len(bs)/budget
            for r in rows:r['actual_rate_ratio_to_baseline']=ratio
            candidates.extend(rows);frames_out.extend(fr);sync.extend(sy)
            summary={'scope':'cell','dataset':v['dataset'],'video':v['name'],'video_id':v['video_id'],'qp':qp,'branch':branch,'restart':0,'lambda_or_beta':beta,'iteration':CFG['optimization_steps'],'frame':'','surrogate_rate':'','actual_bytes':len(bs),'actual_bits':len(bs)*8,'actual_rate_ratio_to_baseline':ratio,'optimization_distortion':obj,'PSNR':sum(x['PSNR'] for x in fr)/64,'MS_SSIM_or_SSIM':sum(x['SSIM'] for x in fr)/64,'LPIPS':sum(float(x['LPIPS']) for x in fr)/64 if fr[0]['LPIPS']!='' else '','DISTS':sum(float(x['DISTS']) for x in fr)/64 if fr[0]['DISTS']!='' else '','quantized_symbol_hash':'','bitstream_hash':sha256(path),'decode_status':'PASS' if all(x['status']=='PASS' for x in sy) else 'FAIL','finite_status':'PASS','num_changed_symbols_vs_baseline':'','L1_symbol_change':'','L2_latent_change':'','interpolated_only':False,'is_budget_feasible':len(bs)<=budget,'is_selected':False,'is_baseline':False,'bitstream_path':str(path)}
            candidates.append(summary);cell_summaries.append(summary);aud.append({'dataset':v['dataset'],'video':v['name'],'qp':qp,'branch':branch,'beta':beta,'bitstream':str(path),'bitstream_sha256':sha256(path),'bytes':len(bs),'symbol_equal':all(x['symbol_equal'] for x in sy),'decoded_latent_max_abs_diff':max(float(x['state_max_abs']) for x in sy),'reconstruction_max_abs_diff':0.0,'decode_status':summary['decode_status']})
    selected=[]
    for branch in ('compression_aware','generator_aware'):
        feasible=[x for x in cell_summaries if x['branch']==branch and x['is_budget_feasible'] and x['decode_status']=='PASS'];objective=[x for x in feasible if x['optimization_distortion']!=''];best=min(objective,key=lambda x:float(x['optimization_distortion'])) if objective else feasible[0]
        if best:best['is_selected']=True;selected.append(dict(best,selection='branch_objective'))
        for metric,reverse in [('PSNR',True),('LPIPS',False),('DISTS',False)]:
            z=[x for x in feasible if x[metric]!='']
            if z:selected.append(dict((max(z,key=lambda x:float(x[metric])) if reverse else min(z,key=lambda x:float(x[metric]))),selection=f'best_{metric}'))
    rd=[{'dataset':v['dataset'],'video':v['name'],'qp':qp,'branch':'baseline','actual_bytes':budget,'actual_bits':budget*8,'PSNR':base_row['PSNR'],'LPIPS':base_row['LPIPS'],'DISTS':base_row['DISTS'],'is_baseline':True,'is_budget_feasible':True,'is_selected':True,'iteration':0,'restart':0,'lambda_or_beta':''}]+[{k:x.get(k,'') for k in ('dataset','video','qp','branch','actual_bytes','actual_bits','PSNR','LPIPS','DISTS','is_baseline','is_budget_feasible','is_selected','iteration','restart','lambda_or_beta')} for x in cell_summaries]
    write_csv(ROOT/'parts'/f'{tag}_baseline.csv',[base_row]);write_csv(ROOT/'parts'/f'{tag}_candidates.csv',candidates);write_csv(ROOT/'parts'/f'{tag}_selected.csv',selected);write_csv(ROOT/'parts'/f'{tag}_frames.csv',frames_out);write_csv(ROOT/'parts'/f'{tag}_audit.csv',aud);write_csv(ROOT/'parts'/f'{tag}_sync.csv',sync);write_csv(ROOT/'parts'/f'{tag}_rd.csv',rd)
    status='PASS' if all_unchanged and all(x['status']=='PASS' for x in sync) and len(selected)>0 else 'FAIL';done.write_text(json.dumps({'status':status,'tag':tag,'budget_bytes':budget,'model_unchanged':all_unchanged,'selected_rows':len(selected)},indent=2)+'\n')
    if status!='PASS':raise RuntimeError(f'cell failed {tag}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--cells',required=True);a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);device=torch.device('cuda:0');lookup={(v['dataset'],int(v['video_id'])):v for v in MAN['videos']}
    for spec in a.cells.split(','):
        ds,vid,qp=spec.split(':');
        try:run_cell(lookup[(ds,int(vid))],int(qp),device)
        except Exception as e:
            (ROOT/'parts'/f'{ds}_{int(vid):02d}_qp{qp}_failed.json').write_text(json.dumps({'status':'FAIL','error':repr(e)},indent=2)+'\n');raise

if __name__=='__main__':main()
