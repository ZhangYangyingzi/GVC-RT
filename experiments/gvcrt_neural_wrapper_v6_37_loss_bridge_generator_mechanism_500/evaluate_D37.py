from receiver37 import *
import argparse,fcntl,traceback

def prepare_D():
 import torch
 path=ROOT/'branches/D_basicDISTS_noP_500/checkpoint_index.json';ix=load(path)['500'];inf=ix['inference'];assert sha(inf['path'])==inf['sha256'];s=torch.load(inf['path'],map_location='cpu',weights_only=True)
 cfg=load(ROOT/'evaluation_config.json');name='D_basicDISTS_noP_500';cfg['methods'][name]=[None,name];cfg['checkpoints'][name]=dict(inf,module_hashes=ix['module_hashes']);cfg['deployment_receiver_hashes'][name]={k:tensor_hash({n:t.half() for n,t in s[k].items()}) for k in ('bridge','generator')};dump(ROOT/'evaluation_config.json',cfg)
 print('D DEPLOY CONFIG PASS',flush=True)
def run(t,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch,numpy as np
 from adapter35 import runtime,frames_for,backend
 backend();key=tag(t);dest=ROOT/'tasks'/f'D_{key}.json';lock=dest.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 if dest.exists() and load(dest)['status']=='PASS':return
 assert load(ROOT/'branches/D_basicDISTS_noP_500/training_status.json')['committed_updates']==500
 eng,rt=runtime(dict(method='D_basicDISTS_noP_500',stage='adapt'));v=metadata(t);frames=frames_for(v);bp=ROOT/'bitstreams'/f'{key}_D.bin';fp=ROOT/'features/D'/f'{key}.npz';tick=time.time();torch.cuda.reset_peak_memory_stats()
 rp=ROOT/'tasks'/f'D_encode_{key}.json'
 if rp.exists():r=load(rp)['result'];assert sha(r['bitstream_path'])==r['bitstream_sha256'] and sha(r['feature_path'])==r['feature_sha256']
 else:
  r,fr=rt.run(frames,t['qp'],bp,fp);r.update(engine_internal_fps_before_protocol_accounting_override=r['rate_accounting_fps'],rate_accounting_fps=20,kbps=r['bits_per_frame']*20/1000,kbps_20fps=r['bits_per_frame']*20/1000);frpath=ROOT/'raw'/f'{key}_D_frames.csv';write(frpath,fr);r.update(frame_metrics_path=str(frpath),frame_metrics_sha256=sha(frpath),actual_qps=[x['actual_qp'] for x in fr]);dump(rp,dict(status='PASS',result=r))
 old=sourcepoint(t,'O_original');assert r['real_bytes']==int(old['real_bytes']) and r['bitstream_sha256']==old['bitstream_sha256']
 assert r['independent_decode_pass'] and r['state_sync_pass'] and r['bytes_consumed']==r['real_bytes']
 cached=load(ROOT/'tasks'/f'replay_{key}.json');la=cached['latent_archive'];assert sha(la['path'])==la['sha256'];archive=np.load(la['path']);stats=[];changes=[];proof=[];equiv=[];cov=[];hashes=[]
 im,pm=rt.models(1088,1920);bank=receiver_bank(pm,['D_basicDISTS_noP_500']);m=bank['D_basicDISTS_noP_500'];cap={};hooks=[pm.dec.register_forward_pre_hook(lambda m,a:cap.update(y=a[0])),pm.dec.register_forward_hook(lambda m,a,o:cap.update(f=o)),pm.recon_generation_net.decoder.register_forward_pre_hook(lambda m,a:cap.update(z=a[0],q=a[1]))]
 with torch.inference_mode():
  for i,sp,q,bits,realbits in parse(bp,eng.eco):
   if not i:
    x=im.decompress(bits,sp,q)['x_hat'];pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,x)
   else:
    actual=pm.decompress(bits,sp,q)['x_hat'];x,z=forward(m,cap['f'],cap['q']);diff=difference(x,actual);assert diff['bitwise'],('D endpoint replay must be exact',i,diff);assert torch.equal(z,cap['z'])
    original=cached['tensor_equality'][i-1]['decoded']['O_original'];record={k:th(cap[k]) for k in ('y','f','q')};assert all(record[k]==original[k] for k in record),(key,i,'D and O decoded feature differs')
    proof.append(dict(frame=i,decoded=record,all_equal_to_O_F=True));equiv.append(dict(frame=i,**diff));st=latent_stats(z);cov.append(st.pop('covariance18x18'));st.pop('sum');st.pop('cross_sum');stats.append(dict(frame=i,method='D_basicDISTS_noP_500',covariance_index=len(cov)-1,**st))
    zB=torch.from_numpy(archive['zB'][i-1]).cuda();zF=torch.from_numpy(archive['zF'][i-1]).cuda()
    changes.extend([dict(frame=i,comparison='B_to_D',**latent_delta(zB,z)),dict(frame=i,comparison='D_to_F',**latent_delta(z,zF))])
   hashes.append(eng.eco.tensor_sha(x))
   if v['video_index']==load(ROOT/'eval_manifest.json')[t['dataset']][0]['video_index'] and i in (16,32,48,63):eng.eco.save_png(eng.core.unit(x,1080,1920),ROOT/'artifacts/examples'/key/'D_basicDISTS_noP_500'/f'{i}.png')
 for h in hooks:h.remove()
 assert eng.eco.sha256_bytes(''.join(hashes).encode())==r['reconstruction_sha256'];rt.assert_frozen(im,pm)
 ar=ROOT/'artifacts/latents'/f'{key}_D.npz';save_npz(ar,covariance=np.asarray(cov,dtype=np.float64));tr=Path(r['feature_path']).with_suffix('.transitions.csv')
 out=dict(dataset=t['dataset'],video=t['video'],sequence=v['name'],QP=t['qp'],method='D_basicDISTS_noP_500',actual_bpp=r['bpp'],real_bytes=r['real_bytes'],frames=64,num_transitions=63,LPIPS=r['LPIPS'],DISTS=r['DISTS'],FloLPIPS=r['FloLPIPS'],feature_path=r['feature_path'],feature_sha256=r['feature_sha256'],frame_metrics_path=r['frame_metrics_path'],frame_metrics_sha256=r['frame_metrics_sha256'],transitions_path=str(tr),transitions_sha256=sha(tr),bitstream_path=str(bp),bitstream_sha256=sha(bp),reuse_bitstream=False,real_RANS=True,independent_decode_pass=True,state_sync_pass=True,reconstruction_sha256=r['reconstruction_sha256'],code_sha256=sha(ROOT/'evaluate_D37.py'),source_rgb_sha256=v['rgb_sha256'],checkpoint_sha256=payloads()['D_basicDISTS_noP_500']['sha256'])
 dump(dest,dict(status='PASS',task=t,result=out,latent_statistics=stats,latent_pair_changes=changes,tensor_equality=proof,endpoint_comparison=equiv,latent_archive=dict(path=str(ar),sha256=sha(ar)),physical_GPU=gpu,seconds=time.time()-tick,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
 print('D EVAL PASS',key,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--task');a.add_argument('--gpu',type=int,choices=(4,5,6,7),default=6);x=a.parse_args()
 try:prepare_D() if x.prepare else run(json.loads(x.task),x.gpu)
 except Exception:error(dict(stage='D_evaluation',task=x.task,error=traceback.format_exc()));raise
