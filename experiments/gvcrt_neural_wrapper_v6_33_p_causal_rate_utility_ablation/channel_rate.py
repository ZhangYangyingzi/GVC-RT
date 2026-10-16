"""Ideal symbol code lengths from installed CDFs, including native escape bypass.

These are estimated entropy-model channel rates, never actual per-channel RANS
bytes. State flush, byte renormalization, headers and I-frame costs are excluded.
"""
from audit import *
import numpy as np

def cdf_bits(symbols,indexes,coder):
    symbols=np.asarray(symbols,dtype=np.int64);indexes=np.asarray(indexes,dtype=np.int64)
    cdf,lengths,offsets=coder.get_cdf_info()
    pos=symbols-offsets[indexes];tail=lengths[indexes]-2
    escape=(pos<0)|(pos>=tail)
    raw=np.where(pos<0,-2*pos-1,2*(pos-tail));raw=np.maximum(raw,0)
    nb=np.zeros_like(raw)
    for j in range(1,17):nb+=(raw >= (1 << (2*(j-1))))
    bypass=np.where(escape,2*(nb+nb//3+1),0)
    bins=np.where(escape,tail,pos)
    freq=cdf[indexes,bins+1].astype(np.int64)-cdf[indexes,bins].astype(np.int64)
    assert (freq>0).all() and (freq<=65536).all()
    cdf_length=-np.log2(freq.astype(np.float64)/65536.)
    return cdf_length+bypass,dict(symbols=len(symbols),escape_symbols=int(escape.sum()),CDF_bits=float(cdf_length.sum()),bypass_bits=int(bypass.sum()),total_estimated_bits=float((cdf_length+bypass).sum()))

def attach(pm):
    import torch
    source=(REPO/'src/cpp/py_rans/rans.cpp').read_text()
    assert 'bypass_precision = 2' in source and 'max_bypass_val = (1 << bypass_precision) - 1' in source
    original=pm.gaussian_encoder.get_y;original_z=pm.bit_estimator_z.get_z
    state=dict(enabled=False,groups=[])
    def get_y(shape,numel,dtype,device,skip_cond,indexes):
        out=original(shape,numel,dtype,device,skip_cond,indexes)
        if state['enabled']:
            group=len(state['groups']);assert group in (0,1)
            mask=pm.get_mask_2x(shape[0],shape[1]*2,shape[2],shape[3],dtype,device)[group]
            channels=torch.arange(shape[1],device=device).reshape(1,-1,1,1).expand(shape)
            channels=channels+shape[1]*(1-mask[:,:shape[1]]).long()
            active=torch.ones(numel,device=device,dtype=torch.bool) if skip_cond is None else skip_cond
            symbols=out.flatten()[active].float().cpu().numpy();ix=indexes.cpu().numpy();ids=channels.flatten()[active].cpu().numpy()
            assert np.array_equal(symbols,np.round(symbols)) and len(symbols)==len(ix)==len(ids)
            bits,summary=cdf_bits(symbols,ix,pm.gaussian_encoder)
            per=np.bincount(ids,weights=bits,minlength=shape[1]*2)
            assert abs(per.sum()-bits.sum())<1e-7
            state['groups'].append(dict(estimated_channel_bits=per.tolist(),packed_shape=list(shape),dtype=str(dtype),**summary))
        return out
    pm.gaussian_encoder.get_y=get_y
    def get_z(size,device,dtype):
        out=original_z(size,device,dtype)
        if state['enabled']:
            indexes=pm.bit_estimator_z.build_indexes(out.shape,state['qp']).numpy().flatten()
            symbols=out.float().flatten().cpu().numpy()
            bits,summary=cdf_bits(symbols,indexes,pm.bit_estimator_z)
            per=bits.reshape(out.shape).sum(axis=(0,2,3))
            state['hyper']=dict(estimated_channel_bits=per.tolist(),shape=list(out.shape),dtype=str(dtype),**summary)
        return out
    pm.bit_estimator_z.get_z=get_z
    def detach():
        pm.gaussian_encoder.get_y=original;pm.bit_estimator_z.get_z=original_z
    return state,detach

def finish(state):
    assert len(state['groups'])==2
    rates=np.sum([g['estimated_channel_bits'] for g in state['groups']],axis=0)
    total=sum(g['total_estimated_bits'] for g in state['groups'])
    error=float(abs(rates.sum()-total));assert error<1e-7
    hyper=state['hyper'];assert abs(sum(hyper['estimated_channel_bits'])-hyper['total_estimated_bits'])<1e-7
    return dict(estimated_channel_bits=rates.tolist(),estimated_y_total_bits=float(total),hyperlatent=hyper,estimated_y_plus_z_total_bits=float(total+hyper['total_estimated_bits']),group_diagnostics=state['groups'],sum_channel_vs_combined_y_estimate_abs_error=error,
        per_channel_actual_rate_available=False,estimated_channel_rate_available=True,
        definition='-log2(actual installed 16-bit CDF mass) plus exact uniform-bypass symbol lengths from native escape rule; main y stream only',
        excluded='Main y channel vector excludes z (reported separately), I-frame, SPS/header, RANS state flush and byte renormalization overhead; actual complete stream rate reported separately',
        bypass_rule_source=str(REPO/'src/cpp/py_rans/rans.cpp'),bypass_rule_source_sha256=sha(REPO/'src/cpp/py_rans/rans.cpp'))
