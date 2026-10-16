"""Real independent RANS decode at fixed sanity checkpoints; no optimizer updates."""
from runtime import *
import os
import time
import io

def check(branch,step):
    import torch
    os.environ.update(CUDA_VISIBLE_DEVICES='7',CUDA_DEVICE_ORDER='PCI_BUS_ID')
    ctx=setup(branch);v4,im,pm,models,quality=ctx
    path=ROOT/'branches'/branch/'checkpoints'/f'adaptation_step_{step:04d}.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    for k,m in models.items():m.load_state_dict(state[k])
    pm.recon_generation_net.half()
    idec,pdec=v4.load_models(torch.device('cuda:0'),force_zero_thres=.12)
    for k,m in [('bridge',pdec.recon_generation_net.mlp),('generator',pdec.recon_generation_net.decoder)]:m.load_state_dict(state[k])
    for m in (im,pm,idec,pdec):m.set_use_two_entropy_coders(False)
    row=load(V32/'plans/shared.json')['updates'][0];f=frames(row);x0=f[0];x1=f[1];f.close();qp=row['external_qp']
    sps=dict(sps_id=0,height=256,width=256,ec_part=0,use_ada_i=0)
    sys.path.insert(0,str(REPO));from src.utils.stream_helper import write_sps,write_ip
    buffer=io.BytesIO();write_sps(buffer,sps)
    with torch.inference_mode():
        pm.clear_dpb();pdec.clear_dpb();pm.set_curr_poc(0);pdec.set_curr_poc(0)
        i=im.compress(v4.codec_input(x0),qp);d=idec.decompress(i['bit_stream'],sps,qp);assert torch.equal(i['x_hat'],d['x_hat'])
        pm.add_ref_frame(None,d['x_hat']);pdec.add_ref_frame(None,d['x_hat']);write_ip(buffer,True,0,qp,i['bit_stream'])
        qa=pm.shift_qp(qp,1);encoded=pm.compress(v4.codec_input(models['wrapper'](x1)),qa);decoded=pdec.decompress(encoded['bit_stream'],sps,qa)
        assert torch.equal(pm.dpb[0].feature,pdec.dpb[0].feature)
        assert decoded['x_hat'].isfinite().all();write_ip(buffer,False,0,qa,encoded['bit_stream'])
    p=ROOT/'bitstreams'/f'sanity_{branch}_{step}.bin';p.write_bytes(buffer.getvalue())
    logs=load(ROOT/'branches'/branch/'training_rows.json');g=logs[step-1]['gradients'];hashes=state['hashes'];initial=state['initial']
    assert hashes['compression']==initial['compression']
    if branch==BRANCHES[1]:assert not list(models['wrapper'].parameters()) and g['wrapper']['norm']==0
    else:assert hashes['modules']['wrapper']!=initial['modules']['wrapper'] and g['wrapper']['norm']>0
    if branch==BRANCHES[2]:
        for k in ('bridge','generator'):assert hashes['modules'][k]==initial['modules'][k] and g[k]['norm']==0
    else:
        for k in ('bridge','generator'):assert hashes['modules'][k]!=initial['modules'][k] and g[k]['norm']>0
    dump(f'sanity/checkpoint_{branch}_{step}.json',dict(status='PASS',branch=branch,step=step,checkpoint_sha256=sha(path),actual_RANS_total_bytes=p.stat().st_size,bitstream_path=str(p),bitstream_sha256=sha(p),independent_decoder=True,feature_state_exact=True,gradients=g,initial=initial,hashes=hashes,scope='2-frame 256x256 training sample with persisted SPS/header/I/P bytes, not formal evaluation'))
    print('CHECKPOINT SANITY PASS',branch,step,flush=True)

def main():
    pending=[(b,s) for s in (100,250) for b in BRANCHES[1:]]
    while pending:
        for b,s in list(pending):
            if (ROOT/'branches'/b/'checkpoints'/f'adaptation_step_{s:04d}.pt').exists():
                receipt=ROOT/'sanity'/f'checkpoint_{b}_{s}.json'
                if not receipt.exists():check(b,s)
                pending.remove((b,s))
        if pending:time.sleep(30)

if __name__=='__main__':
    import traceback
    try:main()
    except Exception:
        dump('sanity/checkpoint_failure.json',dict(traceback=traceback.format_exc()));raise
