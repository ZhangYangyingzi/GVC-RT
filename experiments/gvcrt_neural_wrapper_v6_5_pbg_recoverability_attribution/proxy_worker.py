"""Direct wrapper diagnostics, without codec or receiver execution."""
import argparse,ast,fcntl,math,traceback
from v65_io import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--video',type=int,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch,numpy as np,cv2
    torch.set_num_threads(2);cv2.setNumThreads(1);cfg=frozen()
    v=next(v for v in videos() if v['dataset']==a.dataset and v['video_index']==a.video)
    lock=(ROOT/'parts'/f'proxy_{sid(v)}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tree=ast.parse((AUDIT/'source_stats.py').read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('gray','gradient','camera')]
    ns=dict(cv2=cv2,np=np,math=math);exec(compile(tree,str(AUDIT/'source_stats.py'),'exec'),ns)
    gray,gradient,camera=[ns[k] for k in ('gray','gradient','camera')]
    e=engine();device=torch.device('cuda:0');quality=e.core.quality_models(device);flow,_=e.metrics.flo.load_models(device)
    frames=frames_for(v);assert len(frames)==64
    array=lambda x:x[0].permute(1,2,0).float().cpu().numpy()
    u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    def ratio(r,key,num,den):
        r[key]=float(num/den) if den>0 else None;r[key+'_status']='PASS' if den>0 else 'UNDEFINED_ZERO_DENOMINATOR'
    for theta in ('V62','V64'):
        path=ROOT/'parts/proxy'/f'{sid(v)}_{theta}.json'
        if path.exists():assert load(path)['status']=='PASS';continue
        _,wrapper=e.eco.load_joint(cfg['checkpoints'][theta]['path'],device)
        assert e.core.module_hash(wrapper)==cfg['checkpoints'][theta]['module_hashes']['wrapper']
        rows=[];prev=None
        with torch.inference_mode():
            for i,cpu in enumerate(frames):
                x=cpu.to(device);y=wrapper(x);assert torch.isfinite(y).all()
                xx,yy=array(x),array(y);gx,gy=gray(xx),gray(yy);ex,ey=gradient(gx),gradient(gy)
                lx=float(cv2.Laplacian(gx,cv2.CV_32F,ksize=3).var());ly=float(cv2.Laplacian(gy,cv2.CV_32F,ksize=3).var())
                hx=float(np.square(gx-cv2.GaussianBlur(gx,(5,5),1,borderType=cv2.BORDER_REFLECT_101)).mean());hy=float(np.square(gy-cv2.GaussianBlur(gy,(5,5),1,borderType=cv2.BORDER_REFLECT_101)).mean())
                r=dict(dataset=v['dataset'],sequence=v['name'],video_index=v['video_index'],theta=theta,frame=i,
                    **e.eco.frame_metrics(y,x,quality),L1=float((y-x).abs().mean()),edge_L1=float(np.abs(ey-ex).mean()),
                    laplacian_x=lx,laplacian_proxy=ly,HF_x=hx,HF_proxy=hy,edge_energy_x=float(np.square(ex).mean()),edge_energy_proxy=float(np.square(ey).mean()),
                    difference_abs_max=float((y-x).abs().max()),temporal_status='FIRST_FRAME' if i==0 else 'PASS',camera_quantization='round_RGB_times_255_uint8')
                ratio(r,'laplacian_ratio',ly,lx);ratio(r,'HF_ratio',hy,hx);ratio(r,'edge_relative_energy_change',r['edge_energy_proxy']-r['edge_energy_x'],r['edge_energy_x'])
                if prev is not None:
                    px,py,pxx,pyy=prev
                    for label,t,pt,arr,pa in [('x',x,px,xx,pxx),('proxy',y,py,yy,pyy)]:
                        delta=t-pt;r['temporal_RGB_L1_'+label]=float(delta.abs().mean());r['temporal_RGB_MSE_'+label]=float(delta.square().mean())
                        f=flow(pt,t);r['flow_magnitude_'+label]=float(f.square().sum(1).sqrt().mean())
                        cam=camera(u8(pa),u8(arr));r['camera_status_'+label]=cam['status']
                        for k in ('compensated_residual_L1','compensated_residual_MSE'):r[k+'_'+label]=cam.get(k)
                    for k in ('temporal_RGB_L1','temporal_RGB_MSE','flow_magnitude','compensated_residual_L1','compensated_residual_MSE'):
                        r[k+'_change']=r[k+'_proxy']-r[k+'_x'] if r[k+'_proxy'] is not None and r[k+'_x'] is not None else None
                assert all(math.isfinite(z) for z in r.values() if isinstance(z,float))
                rows.append(r);prev=(x,y,xx,yy)
                if chosen(v) and i in visual_indices(v):
                    dest=ROOT/'visualizations/proxy'/sid(v)/f'frame_{i:06d}'
                    e.eco.save_png(x,dest/'GT.png');e.eco.save_png(y,dest/f'{theta}.png');e.eco.save_png((y-x).abs(),dest/f'abs_{theta}.png')
                if i%16==0:print('PROXY',sid(v),theta,i,flush=True)
        csvpath=path.with_suffix('.csv');write(csvpath,rows)
        dump(path,dict(status='PASS',dataset=v['dataset'],video_index=v['video_index'],theta=theta,frames=64,source_rgb_sha256=v['rgb_sha256'],wrapper_hash=e.core.module_hash(wrapper),csv=str(csvpath),csv_sha256=sha(csvpath)))
        del wrapper
    dump(ROOT/'parts'/f'proxy_{sid(v)}_done.json',dict(status='PASS',frames=128))

if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'logs'/f'proxy_failure_{os.getpid()}.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
