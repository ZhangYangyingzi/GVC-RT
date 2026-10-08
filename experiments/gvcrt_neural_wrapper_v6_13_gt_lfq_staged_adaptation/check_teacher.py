"""Fixed non-test teacher/ori compatibility measurements; never trains."""
import argparse, importlib.util, traceback, hashlib, io, math
from io_utils import *
def arrays(sample):
    import cv2,numpy as np
    from PIL import Image
    x,y,w,h=sample['crop']
    if sample['domain']=='vimeo':
        result=[]
        for p,expected in zip(sample['paths'],sample['source_sha256']):
            assert sha(p)==expected
            with Image.open(p) as im:result.append(np.asarray(im.convert('RGB'),dtype=np.uint8)[y:y+h,x:x+w].copy())
    else:
        assert sha(sample['path'])==sample['source_sha256']
        cap=cv2.VideoCapture(sample['path']);cap.set(cv2.CAP_PROP_POS_FRAMES,sample['frame_indices'][0]);result=[]
        for _ in sample['frame_indices']:
            ok,bgr=cap.read();assert ok;result.append(cv2.cvtColor(bgr[y:y+h,x:x+w],cv2.COLOR_BGR2RGB).copy())
        cap.release()
    assert [hashlib.sha256(a.tobytes()).hexdigest() for a in result]==sample['frame_rgb_sha256']
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);p.add_argument('--candidate',choices=('imagenet256','pretrain262144'),default='imagenet256');p.add_argument('--raw',action='store_true');a=p.parse_args()
    command([sys.executable,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    sys.path.insert(0,str(ROOT.parent/'gvcrt_neural_wrapper_v6_1_large_diverse_qp09/runtime_dependencies'))
    sys.path.insert(0,str(ROOT.parent/'gvcrt_neural_wrapper_v2_joint'))
    import torch,numpy as np
    from PIL import Image
    import core,gvc_hooks
    from teacher import Teacher
    torch.set_num_threads(2);torch.manual_seed(20261007);device=torch.device('cuda:0')
    teacher=Teacher(device,ema=not a.raw,candidate=a.candidate);teacher_hash=core.module_hash(teacher);out=teacher.audit_root
    im,pm=core.load_models(device);idc,pdc=core.load_models(device)
    quality=core.quality_models(device)
    for m in (im,pm,idc,pdc):m.set_use_two_entropy_coders(False)
    source_hashes={n:core.module_hash(m) for n,m in [('im',im),('pm',pm),('idc',idc),('pdc',pdc)]}
    assert source_hashes['im']==source_hashes['idc'] and source_hashes['pm']==source_hashes['pdc']
    cfg=load(ROOT/'config.json');samples=load(ROOT/'manifests/teacher_compatibility_samples.json')['samples']
    rows=[];codec=[];latent_rows=[];cosines=[];correlation=[]
    captured={}
    def hook(_m,args):captured.update(z=args[0].detach().clone(),quant=args[1].detach().clone())
    hi=idc.recon_generation_net.decoder.register_forward_pre_hook(hook)
    hp=pdc.recon_generation_net.decoder.register_forward_pre_hook(hook)
    def stats(t):
        return dict(shape=list(t.shape),dtype=str(t.dtype),finite=bool(torch.isfinite(t).all()),min=float(t.min()),max=float(t.max()),mean=float(t.float().mean()),std=float(t.float().std()),norm_mean=float(t.float().norm(dim=1).mean()))
    with torch.no_grad():
        for sample in samples:
            frames=[torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(device).float()/255 for x in arrays(sample)]
            cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
            latents=[teacher(x) for x in frames]
            assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
            assert all(z.shape==(1,18,16,16) and bool(torch.isfinite(z).all()) for z in latents)
            for q in cfg['compatibility_QPs']:
                for m in (pm,pdc):m.clear_dpb();m.set_curr_poc(0)
                sps=dict(sps_id=0,height=256,width=256,ec_part=0,use_ada_i=0)
                stream=io.BytesIO();gvc_hooks.write_sps(stream,sps);state_differences=[]
                for i,frame in enumerate(frames):
                    actual=q if i==0 else pm.shift_qp(q,gvc_hooks.INDEX_MAP[i%8])
                    captured.clear()
                    if i==0:
                        encoded=im.compress(core.codec_input(frame),actual);decoded=idc.decompress(encoded['bit_stream'],sps,actual)
                        assert torch.equal(encoded['x_hat'],decoded['x_hat'])
                        pm.add_ref_frame(None,encoded['x_hat']);pdc.add_ref_frame(None,decoded['x_hat'])
                    else:
                        encoded=pm.compress(core.codec_input(frame),actual);decoded=pdc.decompress(encoded['bit_stream'],sps,actual)
                        d=float((pm.dpb[0].feature-pdc.dpb[0].feature).abs().max());assert d==0.;state_differences.append(d)
                    gvc_hooks.write_ip(stream,i==0,0,actual,encoded['bit_stream'])
                    student=captured['z'];quant=captured['quant'];gtlatent=latents[i];assert student.shape==gtlatent.shape
                    decoder=(idc if i==0 else pdc).recon_generation_net.decoder
                    ori=core.unit(decoded['x_hat'])
                    teacher_image=core.unit(decoder(gtlatent.half(),quant))
                    assert teacher_image.shape==ori.shape==frame.shape and torch.isfinite(teacher_image).all()
                    _,ol,od=core.perceptual(ori,frame,quality);_,tl,td=core.perceptual(teacher_image,frame,quality)
                    cosine=float(torch.nn.functional.cosine_similarity(student.float(),gtlatent.float(),dim=1,eps=1e-8).mean())
                    row=dict(domain=sample['domain'],sample_index=sample['index'],video_id=sample['video_id'],external_qp=q,frame=i,source_frame_index=sample['frame_indices'][i],actual_qp=actual,frame_type='I' if i==0 else 'P',ori_LPIPS=float(ol),ori_DISTS=float(od),teacher_continuous_LPIPS=float(tl),teacher_continuous_DISTS=float(td),latent_cosine=cosine,GT_rgb_sha256=sample['frame_rgb_sha256'][i],teacher=stats(gtlatent),student=stats(student),quant_step=stats(quant),input_RGB_range=[float(frame.min()),float(frame.max())],teacher_RGB_range=[float(teacher_image.min()),float(teacher_image.max())],teacher_rng_unchanged=True)
                    rows.append(row)
                    if i>0:
                        t=gtlatent.flatten(2).squeeze(0).double();s=student.flatten(2).squeeze(0).double()
                        t=t-t.mean(1,keepdim=True);s=s-s.mean(1,keepdim=True)
                        matrix=(s@t.T)/(s.norm(dim=1)[:,None]*t.norm(dim=1)[None,:]).clamp_min(1e-12)
                        correlation.append(matrix.cpu().numpy());cosines.append(cosine)
                    if q==4 and i==1 and sample['index']<2:
                        pics=[]
                        for image in (frame,ori,teacher_image):pics.append((image[0].float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
                        path=out/'compatibility_images'/f'{sample["domain"]}_{sample["index"]:02d}_GT_ori_teacher.png';path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.concatenate(pics,axis=1)).save(path)
                path=out/'bitstreams/compatibility'/f'{sample["domain"]}_{sample["index"]:02d}_q{q}.bin';path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_bytes(stream.getvalue());tmp.replace(path)
                codec.append(dict(domain=sample['domain'],sample_index=sample['index'],external_qp=q,real_bytes=path.stat().st_size,bitstream_sha256=sha(path),independent_decoder=True,DPB_max_differences=state_differences,teacher_not_in_bitstream=True))
            write(out/'teacher_compatibility_per_frame.csv',rows)
            dump(out/'teacher_compatibility_progress.json',dict(status='RUNNING',completed_samples=len(rows)//12,total_samples=32,rows=len(rows),gpu=a.gpu))
            print('COMPATIBILITY',sample['domain'],sample['index'],'rows',len(rows),flush=True)
        hi.remove();hp.remove()
    assert len(rows)==384 and len(codec)==96
    assert source_hashes=={n:core.module_hash(m) for n,m in [('im',im),('pm',pm),('idc',idc),('pdc',pdc)]}
    assert core.module_hash(teacher)==teacher_hash and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    matrix=np.stack(correlation).mean(0)
    write(out/'teacher_student_channel_correlation.csv',[dict(student_channel=i,**{f'teacher_channel_{j}':float(matrix[i,j]) for j in range(18)}) for i in range(18)])
    dump(out/'teacher_compatibility_raw.json',dict(status='MEASUREMENTS_COMPLETE_NOT_GATE_PASS',samples_per_domain=16,samples=32,frames_per_sample=4,external_qps=cfg['compatibility_QPs'],metric_rows=len(rows),codec_checks=codec,teacher_frozen=True,teacher_rng_unchanged=True,native_models_unchanged=True,native_model_hashes=source_hashes,teacher_hash=teacher_hash,latent_shape=[1,18,16,16],coordinate_checks=dict(channel_permutation_applied=False,latent_interpolation_applied=False,additional_projection_applied=False,additional_normalization_applied=False,mean_P_latent_cosine=float(np.mean(cosines)),channel_correlation=matrix.tolist()),checkpoint_to_native_generator_binding='requires separate provenance verification'))
    dump(out/'teacher_compatibility_progress.json',dict(status='MEASUREMENTS_COMPLETE',completed_samples=32,total_samples=32,rows=len(rows),gpu=a.gpu))
    print('COMPATIBILITY MEASUREMENTS COMPLETE; provenance gate still required',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'logs/failures'/f'teacher_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
