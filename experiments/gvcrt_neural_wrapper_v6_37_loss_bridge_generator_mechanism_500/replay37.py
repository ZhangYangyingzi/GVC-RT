from receiver37 import *
import argparse,fcntl,math,traceback,resource

def run(t,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch,numpy as np,copy
 from adapter35 import frames_for,backend
 backend();key=tag(t);lock=(ROOT/'tasks'/f'replay_{key}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 dest=ROOT/'tasks'/f'replay_{key}.json'
 if dest.exists() and load(dest)['status']=='PASS':return
 eng,rt=runtime_original();v=metadata(t);frames=frames_for(v);names=['O_original','B_basicLoss_noP_500','F_fullLoss_noP_500'];methods=[m for m in METHODS if not m.startswith('D_')]
 im,pm=rt.models(1088,1920);bank=receiver_bank(pm,names);decoders={'O_original':pm}
 for n in names[1:]:
  unused,p=rt.models(1088,1920);del unused
  p.recon_generation_net.load_state_dict(bank[n].state_dict());p.eval().requires_grad_(False);decoders[n]=p
 for p in decoders.values():p.clear_dpb();p.set_curr_poc(0)
 before={n:th for n,th in [(n,eng.core.module_hash(p)) for n,p in decoders.items()]};bankhash={n:eng.core.module_hash(p) for n,p in bank.items()}
 paths={n:sourcepoint(t,n) for n in names};stream=paths['O_original']['bitstream_path'];assert len({p['bitstream_sha256'] for p in paths.values()})==1 and sha(stream)==paths['O_original']['bitstream_sha256']
 captures={n:{} for n in names};hooks=[]
 for n,p in decoders.items():
  hooks.append(p.dec.register_forward_pre_hook(lambda m,a,n=n:captures[n].update(y=a[0])))
  hooks.append(p.dec.register_forward_hook(lambda m,a,o,n=n:captures[n].update(f=o)))
  hooks.append(p.recon_generation_net.decoder.register_forward_pre_hook(lambda m,a,n=n:captures[n].update(z=a[0],q=a[1])))
 quality=rt.quality;flow,perceptual,official=rt.metric_models;rows={n:[] for n in methods};trans={n:[] for n in methods};feat={n:[] for n in methods};real=[];previous={};prevgt=None
 hashes={n:[] for n in methods};decodedhash={n:[] for n in names};tensorproof=[];endpoint=[];stats=[];changes=[];zc={'z0':[],'zB':[],'zF':[]};covs=[];covi=[];tick=time.time();torch.cuda.reset_peak_memory_stats()
 with torch.inference_mode():
  for i,sp,q,bits,realbits in parse(stream,eng.eco):
   assert q==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(t['qp'])][i]
   target=frames[i].cuda();decoded={}
   if i==0:
    native=im.decompress(bits,sp,q)['x_hat'];raw={n:native for n in methods}
    for n,p in decoders.items():p.clear_dpb();p.set_curr_poc(0);p.add_ref_frame(None,native);decodedhash[n].append(eng.eco.tensor_sha(native))
   else:
    used={n:dict(feature_present=p.dpb[0].feature is not None,feature_hash=None if p.dpb[0].feature is None else th(p.dpb[0].feature)['sha256'],frame_hash=None if p.dpb[0].frame is None else th(p.dpb[0].frame)['sha256']) for n,p in decoders.items()}
    if i>1:assert all(x['feature_present'] for x in used.values())
    for n,p in decoders.items():decoded[n]=p.decompress(bits,sp,q)['x_hat'];decodedhash[n].append(eng.eco.tensor_sha(decoded[n]))
    c=captures['O_original'];proof={n:{k:th(captures[n][k]) for k in ('y','f','q')} for n in names}
    assert all(len({proof[n][k]['sha256'] for n in names})==1 for k in ('y','f','q'))
    tensorproof.append(dict(frame=i,decoded=proof,reference=used,RGB_reference_consumed=False if i>1 else 'native I only',feature_exact=True))
    raw={};z={}
    for n in names:
     raw[n],z[n]=forward(bank[n],c['f'],c['q']);e=difference(raw[n],decoded[n]);assert e['maxabs']<=1e-5 and e['relative_L2']<=1e-6,(n,i,e)
     assert torch.equal(z[n],captures[n]['z']),(n,i,'actual generator input differs')
     endpoint.append(dict(frame=i,method=n,**e))
    z0,zB,zF=(z[n] for n in names);half=(z0+zF)/2
    assert torch.equal((1-0)*z0+0*zF,z0) and torch.equal((1-1)*z0+1*zF,zF)
    raw['X_fullBridge_originalG']=bank['O_original'].decoder(zF,c['q'])
    raw['Y_originalBridge_fullG']=bank['F_fullLoss_noP_500'].decoder(z0,c['q'])
    raw['I_halfLatent_fullG']=bank['F_fullLoss_noP_500'].decoder(half,c['q'])
    for name,zz in [('z0',z0),('zB',zB),('zF',zF)]:zc[name].append(zz.cpu().numpy())
    for n,zz in [('O_original',z0),('B_basicLoss_noP_500',zB),('F_fullLoss_noP_500',zF),('I_halfLatent_fullG',half)]:
     st=latent_stats(zz);covs.append(st.pop('covariance18x18'));st.pop('cross_sum');st.pop('sum');covi.append(dict(frame=i,method=n));stats.append(dict(frame=i,method=n,covariance_index=len(covs)-1,**st))
    changes.append(dict(frame=i,comparison='z0_to_zF',**latent_delta(z0,zF)))
   for n in methods:hashes[n].append(eng.eco.tensor_sha(raw[n]))
   real.append(official(target).cpu().numpy());gtflow=flow(prevgt,target) if prevgt is not None else None
   for n in methods:
    output=eng.core.unit(raw[n],1080,1920);assert output.shape==target.shape and torch.isfinite(output).all()
    spatial=eng.eco.frame_metrics(output,target,quality);rows[n].append(dict(frame=i,actual_qp=q,real_bits=realbits,**spatial));feat[n].append(official(output).cpu().numpy())
    if i:
     fd=gtflow-flow(previous[n],output);magnitude=fd.square().sum(1,keepdim=True).sqrt().sum();assert torch.isfinite(magnitude) and magnitude>0
     val=float(perceptual(prevgt,previous[n],fd,normalize=True).item());assert math.isfinite(val);trans[n].append(dict(from_frame=i-1,to_frame=i,FloLPIPS=val))
    previous[n]=output
   if v['video_index']==load(ROOT/'eval_manifest.json')[t['dataset']][0]['video_index'] and i in (16,32,48,63):
    for n in ('GT',*methods):
     rgb=target if n=='GT' else eng.core.unit(raw[n],1080,1920)
     eng.eco.save_png(rgb,ROOT/'artifacts/examples'/key/n/f'{i}.png')
   prevgt=target
 for h in hooks:h.remove()
 assert before=={n:eng.core.module_hash(p) for n,p in decoders.items()} and bankhash=={n:eng.core.module_hash(p) for n,p in bank.items()}
 outputs={};verification=[]
 for n in methods:
  fp=ROOT/'features/replay'/f'{key}_{n}.npz';save_npz(fp,real=np.concatenate(real),reconstruction=np.concatenate(feat[n]));fr=ROOT/'raw'/f'{key}_{n}_frames.csv';tr=ROOT/'raw'/f'{key}_{n}_transitions.csv';write(fr,rows[n]);write(tr,trans[n])
  rh=eng.eco.sha256_bytes(''.join(hashes[n]).encode());out=dict(dataset=t['dataset'],video=t['video'],sequence=v['name'],QP=t['qp'],method=n,actual_bpp=float(paths['O_original']['actual_bpp']),real_bytes=int(paths['O_original']['real_bytes']),frames=64,num_transitions=63,LPIPS=sum(x['LPIPS'] for x in rows[n])/64,DISTS=sum(x['DISTS'] for x in rows[n])/64,FloLPIPS=sum(x['FloLPIPS'] for x in trans[n])/63,feature_path=str(fp),feature_sha256=sha(fp),frame_metrics_path=str(fr),frame_metrics_sha256=sha(fr),transitions_path=str(tr),transitions_sha256=sha(tr),reconstruction_sha256=rh,bitstream_path=stream,bitstream_sha256=sha(stream),reuse_bitstream=True,real_RANS=True,independent_decode_pass=True,state_sync_pass=True,receiver_module_hashes=bankhash,code_sha256=sha(ROOT/'replay37.py'),source_rgb_sha256=v['rgb_sha256'])
  outputs[n]=out
  if n in names:
   decoded_sha=eng.eco.sha256_bytes(''.join(decodedhash[n]).encode());assert decoded_sha==paths[n]['reconstruction_sha256'],(n,'historical independent endpoint mismatch')
   errors={k:abs(out[k]-float(paths[n][k])) for k in ('LPIPS','DISTS','FloLPIPS')};assert max(errors.values())<=1e-5,(n,errors)
   verification.append(dict(method=n,historical_reconstruction_exact=rh==paths[n]['reconstruction_sha256'],independent_deployment_historical_exact=True,metric_abs_errors=errors,source_receipt=paths[n]['reuse_source_receipt']))
 latent=ROOT/'artifacts/latents'/f'{key}.npz';save_npz(latent,**{k:np.stack(a) for k,a in zc.items()},covariance=np.asarray(covs,dtype=np.float64));
 dump(dest,dict(status='PASS',task=t,result=outputs,endpoint_comparison=endpoint,historical_verification=verification,tensor_equality=tensorproof,latent_statistics=stats,latent_pair_changes=changes,latent_archive=dict(path=str(latent),sha256=sha(latent),covariance_index=covi),interpolation_endpoints=dict(alpha0_is_Y=True,alpha1_is_F=True,latent_endpoints_bitwise=True,I0_native_shared=True),physical_GPU=gpu,seconds=time.time()-tick,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
 print('REPLAY PASS',key,time.time()-tick,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--task',required=True);a.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);x=a.parse_args()
 try:run(json.loads(x.task),x.gpu)
 except Exception:error(dict(stage='replay',task=json.loads(x.task),error=traceback.format_exc()));raise
