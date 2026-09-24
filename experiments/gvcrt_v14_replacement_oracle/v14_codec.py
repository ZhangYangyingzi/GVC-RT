import hashlib, math, struct, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent; V131=ROOT.parent/'gvcrt_v13_1_structured_mixed_precision'
sys.path.insert(0,str(V131))
import v13_codec as old
from src.layers.cuda_inference import restore_y_2x_with_cat_after,restore_y_2x,add_and_multiply
from src.models.entropy_models import EntropyCoder
MAGIC=b'GVR1'; HEADER=struct.Struct('>4sBBBBHHIIIIII')
ACTION={'FULL':0,'PRED_ONLY':1,'CORR2':2,'CORR4':3,'CORR1':4}; ACTION_NAME={v:k for k,v in ACTION.items()}; ACTION_STEP={0:1,1:0,2:2,3:4,4:1}
tensor_hash=old.tensor_hash; BitWriter=old.BitWriter; BitReader=old.BitReader; block_grid=old.block_grid
encode_original=old.encode_original; _encode_z=old._encode_z; _encode_y=old._encode_y

def all_full_modes(h,w,tile):
    gh,gw=block_grid(h,w,tile,tile); return [0]*(gh*gw)

def _scale_indexes(model,scales):
    th=model.gaussian_encoder.force_zero_thres; model.gaussian_encoder.force_zero_thres=None
    try: indexes,_=model.gaussian_encoder.build_indexes_decoder(scales)
    finally:model.gaussian_encoder.force_zero_thres=th
    return indexes.reshape(scales.shape).long()

def predictor_lookup(model,device,dtype):
    cache=getattr(model.gaussian_encoder,'_v14_predictor_lookup',None)
    if cache is None:
        cdf,lengths,offsets=model.gaussian_encoder.get_cdf_info(); vals=[]
        for row,length,offset in zip(cdf,lengths,offsets):
            freq=np.diff(row[:int(length)])[:-1]; vals.append(int(offset)+int(np.argmax(freq)))
        cache=torch.tensor(vals,dtype=torch.float32); model.gaussian_encoder._v14_predictor_lookup=cache
    return cache.to(device=device,dtype=dtype)

def active_mask(model,scales):
    th=model.gaussian_encoder.force_zero_thres
    return torch.ones_like(scales,dtype=torch.bool) if th is None else scales>th

def predictor_symbols(model,scales):
    idx=_scale_indexes(model,scales); pred=predictor_lookup(model,scales.device,scales.dtype)[idx]
    return pred*active_mask(model,scales)

def estimated_symbol_bits(model, symbols, scales, correction_step=None):
    """Integer-CDF ideal length for active symbols; diagnostic only."""
    active=active_mask(model,scales)
    if not bool(active.any()): return 0.0
    idx=_scale_indexes(model,scales)[active].long().cpu()
    sym=symbols[active].long().cpu()
    if correction_step is None:
        cdf,lengths,offsets=model.gaussian_encoder.get_cdf_info()
        cdf=torch.as_tensor(cdf,dtype=torch.int64).cpu()
        lengths=torch.as_tensor(lengths,dtype=torch.long).cpu()
        offsets=torch.as_tensor(offsets,dtype=torch.long).cpu()
    else:
        info=_correction_group(model,correction_step)
        cdf=info['cdf'].to(torch.int64).cpu(); lengths=info['lengths'].long().cpu()+2
        offsets=torch.full_like(lengths,int(info['cmin']))
    pos=sym-offsets[idx]; pos=torch.minimum(torch.maximum(pos,torch.zeros_like(pos)),lengths[idx]-2)
    lo=cdf[idx,pos]; hi=cdf[idx,pos+1]; total=cdf[idx,lengths[idx]-1]
    prob=(hi-lo).double()/total.double()
    if not bool(torch.isfinite(prob).all()) or bool((prob<=0).any()):
        raise FloatingPointError('non-finite/zero integer-CDF probability')
    return float((-torch.log2(prob)).sum())

def action_tensor(modes,h,w,tile,device):
    gh,gw=block_grid(h,w,tile,tile)
    if len(modes)!=gh*gw: raise ValueError('action-map tile count mismatch')
    grid=torch.as_tensor(modes,device=device,dtype=torch.uint8).reshape(1,1,gh,gw)
    return grid.repeat_interleave(tile,2).repeat_interleave(tile,3)[:,:,:h,:w]

def prepare_frame(model,x,qp,modes=None,tile=8):
    out=old.prepare_frame(model,x,qp,None,tile,tile); h,w=out['shape'][-2:]
    out['q_true']=out['w1'].detach(); out['q_pred']=predictor_symbols(model,out['s1'])
    if modes is None:modes=all_full_modes(h,w,tile)
    return apply_actions(model,out,modes,tile,True)

def apply_actions(model,prepared,modes,tile,reconstruct_rgb=True):
    h,w=prepared['shape'][-2:]; acts=action_tensor(modes,h,w,tile,prepared['q_true'].device).expand_as(prepared['q_true'])
    qt,qp=prepared['q_true'],prepared['q_pred']; qh=qt.clone(); corr=torch.zeros_like(qt)
    for action,step in ((1,0),(2,2),(3,4),(4,1)):
        sel=acts==action
        if not bool(sel.any()):continue
        if step==0:qh[sel]=qp[sel]
        else:
            c=torch.round((qt[sel]-qp[sel])/float(step));corr[sel]=c;qh[sel]=qp[sel]+step*c
    qh=qh*active_mask(model,prepared['s1']); yh1=restore_y_2x(qh,prepared['_means1'],prepared['_mask1'])
    latent=add_and_multiply(prepared['_yhat0'].clone(),yh1,prepared['_q_dec']);out=dict(prepared)
    out.update(modes=list(map(int,modes)),action_tensor=acts.detach(),w1=qh.detach(),correction=corr.detach(),latent=latent.detach())
    if reconstruct_rgb:
        feature=model.dec(latent,prepared['ctx'],prepared['qd']);rgb=model.recon_generation_net(feature,prepared['qr'])
        out.update(feature=feature.detach(),rgb=rgb.detach())
    return out

def _runs(modes):
    out=[]
    for v in modes:
        if not out or out[-1][0]!=v:out.append([int(v),1])
        else:out[-1][1]+=1
    return [(v,n) for v,n in out]

def encode_map_rle(modes):
    w=BitWriter();runs=_runs(modes)
    for mode,n in runs:w.put(mode,2);w.ue(n-1)
    data,bits=w.bytes();return data,bits,runs

def decode_map_rle(data,bits,count):
    r=BitReader(data,bits);out=[]
    while len(out)<count:
        mode=r.get(2);n=r.ue()+1
        if mode not in range(4) or len(out)+n>count:raise ValueError('invalid action RLE')
        out.extend([mode]*n)
    if r.pos!=bits:raise ValueError('unused RLE bits')
    return out

def encode_map_dominant(modes):
    counts=[modes.count(i) for i in range(4)];dom=int(np.argmax(counts));exc=[(i,int(x)) for i,x in enumerate(modes) if x!=dom]
    w=BitWriter();w.put(dom,2);w.ue(len(exc));prev=-1
    for idx,mode in exc:w.ue(idx-prev-1);w.put(mode,2);prev=idx
    data,bits=w.bytes();return data,bits,{'dominant':dom,'exceptions':len(exc)}

def decode_map_dominant(data,bits,count):
    r=BitReader(data,bits);dom=r.get(2);n=r.ue();out=[dom]*count;prev=-1
    for _ in range(n):
        idx=prev+1+r.ue();mode=r.get(2)
        if idx>=count or mode not in range(4):raise ValueError('invalid dominant map')
        out[idx]=mode;prev=idx
    if r.pos!=bits:raise ValueError('unused dominant bits')
    return out

def encode_map_packed(modes):
    w=BitWriter()
    for m in modes:w.put(int(m),2)
    data,bits=w.bytes();return data,bits,None

def decode_map_packed(data,bits,count):
    if bits!=2*count:raise ValueError('packed map size')
    r=BitReader(data,bits);return [r.get(2) for _ in range(count)]

def encode_mode_map_best(modes):
    cs=[]
    for kind,name,fn in ((1,'RASTER_RLE',encode_map_rle),(3,'DOMINANT_SPARSE',encode_map_dominant),(4,'PACKED_2BIT',encode_map_packed)):
        data,bits,detail=fn(modes);cs.append((len(data),bits,kind,name,data,detail))
    b=min(cs,key=lambda x:(x[0],x[1],x[2]))
    return {'data':b[4],'bits':b[1],'kind':b[2],'syntax':b[3],'detail':b[5],
      'rle_bits':cs[0][1],'dominant_bits':cs[1][1],'packed_bits':cs[2][1],
      'rle_bytes':cs[0][0],'dominant_bytes':cs[1][0],'packed_bytes':cs[2][0]}

def _correction_group(model,step):
    cache=getattr(model.gaussian_encoder,'_v14_correction_groups',None);coder=model.entropy_coder
    if cache is None or cache.get('coder') is not coder:cache={'coder':coder};model.gaussian_encoder._v14_correction_groups=cache
    if step in cache:return cache[step]
    dev=model.gaussian_encoder.scale_table.device;q=torch.arange(-128,128,device=dev,dtype=torch.float64);c=torch.round(q/float(step)).long()
    cmin,cmax=int(c.min()),int(c.max());n=cmax-cmin+1;rows=[]
    for scale in model.gaussian_encoder.scale_table.double():
        normal=torch.distributions.Normal(torch.tensor(0.,device=dev,dtype=torch.float64),scale)
        p=normal.cdf(q+.5)-normal.cdf(q-.5);p[0]+=normal.cdf(torch.tensor(-127.5,device=dev,dtype=torch.float64));p[-1]+=1-normal.cdf(torch.tensor(127.5,device=dev,dtype=torch.float64))
        grouped=torch.zeros(n,device=dev,dtype=torch.float64);grouped.scatter_add_(0,c-cmin,p);rows.append(grouped.float())
    pmf=torch.stack(rows).cpu();lengths=torch.full((len(rows),),n,dtype=torch.int32);tail=torch.full((len(rows),1),1e-12)
    cdf=EntropyCoder.pmf_to_cdf(pmf,tail,lengths,n);group=coder.add_cdf(cdf.numpy(),(lengths+2).numpy(),torch.full_like(lengths,cmin).numpy())
    cache[step]={'group':group,'cdf':cdf,'lengths':lengths,'cmin':cmin,'cmax':cmax};return cache[step]

def encode_w1_streams(model,p):
    acts=p['action_tensor'].expand_as(p['q_true']);active=active_mask(model,p['s1']);idx=_scale_indexes(model,p['s1'])
    sel=(acts==0)&active;model.entropy_coder.reset()
    if bool(sel.any()):model.entropy_coder.encode_y((p['q_true'][sel].to(torch.int16)<<8)+idx[sel].to(torch.int16),model.gaussian_encoder.cdf_group_index)
    model.entropy_coder.flush();ret=model.entropy_coder.get_encoded_stream();model.entropy_coder.reset()
    for action,step in ((4,1),(2,2),(3,4)):
        sel=(acts==action)&active
        if bool(sel.any()):
            info=_correction_group(model,step);combined=(p['correction'][sel].to(torch.int16)<<8)+idx[sel].to(torch.int16)
            model.entropy_coder.encode_y(combined,info['group'])
    model.entropy_coder.flush();corr=model.entropy_coder.get_encoded_stream();return ret,corr

def decode_w1_streams(model,ret,corr,scales,modes,tile,qpred):
    h,w=scales.shape[-2:];acts=action_tensor(modes,h,w,tile,scales.device).expand_as(scales);active=active_mask(model,scales);idx=_scale_indexes(model,scales);out=qpred.clone()
    sel=(acts==0)&active
    if bool(sel.any()):
        if not ret:raise ValueError('missing retained stream')
        model.entropy_coder.set_stream(ret);out.masked_scatter_(sel,model.entropy_coder.decode_and_get_y(idx[sel],model.gaussian_encoder.cdf_group_index,scales.device,scales.dtype))
    elif ret:raise ValueError('unexpected retained stream')
    anycorr=any(bool(((acts==a)&active).any()) for a in (4,2,3))
    if anycorr:
        if not corr:raise ValueError('missing correction stream')
        model.entropy_coder.set_stream(corr)
        for action,step in ((4,1),(2,2),(3,4)):
            sel=(acts==action)&active
            if bool(sel.any()):
                info=_correction_group(model,step);c=model.entropy_coder.decode_and_get_y(idx[sel],info['group'],scales.device,scales.dtype);out.masked_scatter_(sel,qpred[sel]+step*c)
    elif corr:raise ValueError('unexpected correction stream')
    return out*active

def encode_replacement(model,p,qp,tile=8,global_action=None):
    model.entropy_coder.set_use_two_entropy_coders(False);z=_encode_z(model,p['z_write'],qp);w0=_encode_y(model,p['w0'],p['s0']);ret,corr=encode_w1_streams(model,p);h,w=p['shape'][-2:]
    if global_action is None:mi=encode_mode_map_best(p['modes']);kind,gm,ms,mb=mi['kind'],255,mi['data'],mi['bits']
    else:kind,gm,ms,mb=2,int(global_action),b'',0;mi={'syntax':'GLOBAL','rle_bits':0,'dominant_bits':0,'packed_bits':0}
    head=HEADER.pack(MAGIC,1,kind,gm,0,h,w,mb,len(z),len(w0),len(ret),len(corr),len(ms));payload=head+ms+z+w0+ret+corr
    return payload,{'z_bytes':len(z),'w0_bytes':len(w0),'original_retained_w1_bytes':len(ret),'replacement_correction_bytes':len(corr),'mode_map_bytes':len(ms),'mode_map_bits':mb,'headers_bytes':len(head),'alignment_bytes':len(ms)-math.ceil(mb/8),'selected_syntax':mi['syntax'],'rle_bits':mi['rle_bits'],'dominant_bits':mi['dominant_bits'],'packed_bits':mi['packed_bits']}

def parse_replacement(payload,tile=8):
    if len(payload)<HEADER.size or payload[:4]!=MAGIC:return None
    _,ver,kind,gm,flags,h,w,mb,zl,w0l,rl,cl,ml=HEADER.unpack(payload[:HEADER.size])
    if ver!=1 or flags!=0 or len(payload)!=HEADER.size+ml+zl+w0l+rl+cl:raise ValueError('replacement header')
    pos=HEADER.size;ms=payload[pos:pos+ml];pos+=ml;count=math.ceil(h/tile)*math.ceil(w/tile)
    if kind==1:modes=decode_map_rle(ms,mb,count);syntax='RASTER_RLE'
    elif kind==2 and gm in ACTION_NAME and ml==0 and mb==0:modes=[gm]*count;syntax='GLOBAL'
    elif kind==3:modes=decode_map_dominant(ms,mb,count);syntax='DOMINANT_SPARSE'
    elif kind==4:modes=decode_map_packed(ms,mb,count);syntax='PACKED_2BIT'
    else:raise ValueError('action map kind')
    z=payload[pos:pos+zl];pos+=zl;w0=payload[pos:pos+w0l];pos+=w0l;ret=payload[pos:pos+rl];pos+=rl;corr=payload[pos:pos+cl]
    return {'h':h,'w':w,'modes':modes,'syntax':syntax,'map_bits':mb,'map_stream':ms,'z_stream':z,'w0_stream':w0,'retained_stream':ret,'correction_stream':corr}

def decode_replacement(model,payload,sps,qp,tile=8):
    parsed=parse_replacement(payload,tile)
    if parsed is None:return old.decode_mixed(model,payload,sps,qp,tile,tile)
    dtype,device=next(model.parameters()).dtype,next(model.parameters()).device;qf=model.q_scale_feature[qp:qp+1];qd=model.q_scale_dec[qp:qp+1];qr=model.q_scale_recon[qp:qp+1]
    model.entropy_coder.set_use_two_entropy_coders(False);zsize=model.get_downsampled_shape(sps['height'],sps['width'],64);model.entropy_coder.set_stream(parsed['z_stream']);model.bit_estimator_z.decode_z(zsize,qp)
    ref=model.apply_feature_adaptor();c1,ct=model.feature_extractor.forward_part1(ref,qf);z=model.bit_estimator_z.get_z(zsize,device,dtype);params=model.res_prior_param_decoder(z,ct);qdec,scales,means=model.separate_prior_for_video_decoding(params)
    b,c,h,w=means.shape
    if (h,w)!=(parsed['h'],parsed['w']):raise ValueError('latent dimensions')
    m0,m1=model.get_mask_2x(b,c,h,w,dtype,device);s0=model.single_part_for_writing_2x(scales*m0);model.entropy_coder.set_stream(parsed['w0_stream']);w0=model.gaussian_encoder.decode_and_get_y(s0,dtype,device)
    yh0,cat=restore_y_2x_with_cat_after(w0,means,m0,params);s1,means1=model.y_spatial_prior(cat).chunk(2,1);s1w=model.single_part_for_writing_2x(s1*m1);qpred=predictor_symbols(model,s1w);w1=decode_w1_streams(model,parsed['retained_stream'],parsed['correction_stream'],s1w,parsed['modes'],tile,qpred)
    yh1=restore_y_2x(w1,means1,m1);yhat=add_and_multiply(yh0,yh1,qdec);ctx=model.feature_extractor.forward_part2(c1);rgb,feature=model.get_recon_and_feature(yhat,ctx,qd,qr);model.add_ref_frame(feature,rgb)
    return rgb,{'modes':parsed['modes'],'z':z.detach(),'w0':w0.detach(),'w1':w1.detach(),'q_pred':qpred.detach(),'latent':yhat.detach(),'feature':feature.detach(),'mixed':True,'syntax':parsed['syntax']}

def predictor_stats(qtrue,qpred,scales):
    e=(qtrue-qpred)[scales!=0].float()
    if not e.numel():return {'num_symbols':0,'exact_match_rate':1.,'MAE':0.,'RMSE':0.,'signed_error':0.}
    return {'num_symbols':int(e.numel()),'exact_match_rate':float((e==0).float().mean()),'MAE':float(e.abs().mean()),'RMSE':float(e.square().mean().sqrt()),'signed_error':float(e.mean())}
