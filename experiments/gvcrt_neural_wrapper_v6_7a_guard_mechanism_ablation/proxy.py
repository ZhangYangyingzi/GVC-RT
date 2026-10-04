"""V6.5 spatial proxy diagnostics for A/B/C on the same 15 sequences."""
import argparse,ast,math
from v67_io import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--video',type=int,required=True);p.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch,numpy as np,cv2
    torch.set_num_threads(2);cv2.setNumThreads(1);frozen();v=next(v for v in sources(a.dataset) if v['video_index']==a.video);assert a.dataset in ('uvg','ulong')
    frames=module('v67_proxy_frames',V62B/'v62b_io.py').frames_for(v);e=engine();device=torch.device('cuda:0');quality=e.core.quality_models(device)
    src=ROOT.parent/'dataset_distribution_audit_v1/source_stats.py';tree=ast.parse(src.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('gray','gradient')];ctx=dict(np=np,cv2=cv2);exec(compile(tree,str(src),'exec'),ctx)
    gray,gradient=ctx['gray'],ctx['gradient'];cps=load(ROOT/'checkpoint_hashes.json')
    for method in REPORT_METHODS:
        path=ROOT/'parts/proxy'/f'{a.dataset}_{a.video:02d}_{method}.json'
        if path.exists():r=load(path);assert r['status']=='PASS' and sha(r['csv'])==r['csv_sha256'];continue
        _,wrapper=e.eco.load_joint(cps[method]['path'],device);assert e.core.module_hash(wrapper)==cps[method]['module_hashes']['wrapper'];rows=[]
        with torch.inference_mode():
            for i,cpu in enumerate(frames):
                x=cpu.to(device);y=wrapper(x);xx=x[0].permute(1,2,0).cpu().numpy();yy=y[0].permute(1,2,0).cpu().numpy();gx,gy=gray(xx),gray(yy);ex,ey=gradient(gx),gradient(gy)
                lx=float(cv2.Laplacian(gx,cv2.CV_32F,ksize=3).var());ly=float(cv2.Laplacian(gy,cv2.CV_32F,ksize=3).var());hx=float(np.square(gx-cv2.GaussianBlur(gx,(5,5),1,borderType=cv2.BORDER_REFLECT_101)).mean());hy=float(np.square(gy-cv2.GaussianBlur(gy,(5,5),1,borderType=cv2.BORDER_REFLECT_101)).mean());enx=float(np.square(ex).mean());eny=float(np.square(ey).mean())
                row=dict(dataset=a.dataset,sequence=v['name'],video_index=a.video,method=method,frame=i,L1=float((x-y).abs().mean()),**e.eco.frame_metrics(y,x,quality),edge_L1=float(np.abs(ey-ex).mean()),laplacian_x=lx,laplacian_proxy=ly,HF_x=hx,HF_proxy=hy)
                for k,num,den in [('edge_relative_energy_change',eny-enx,enx),('HF_ratio',hy,hx),('Laplacian_ratio',ly,lx)]:row[k]=num/den if den>0 else None;row[k+'_status']='PASS' if den>0 else 'UNDEFINED_ZERO_DENOMINATOR'
                assert all(math.isfinite(z) for z in row.values() if isinstance(z,float));rows.append(row)
        csvpath=path.with_suffix('.csv');write(csvpath,rows);dump(path,dict(status='PASS',csv=str(csvpath),csv_sha256=sha(csvpath),source_rgb_sha256=v['rgb_sha256'],checkpoint_sha256=cps[method]['sha256']))
        print('PROXY PASS',a.dataset,a.video,method,flush=True)
    dump(ROOT/'parts'/f'proxy_done_{a.dataset}_{a.video}.json',dict(status='PASS'))
if __name__=='__main__':main()
