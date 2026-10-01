"""Seven-frame Vimeo Original-codec endpoint smoke; excluded from formal selection."""
from audit_io import *
from frames import native_frames
def main():
    os.environ['CUDA_VISIBLE_DEVICES']='6'
    import torch
    import math
    torch.set_num_threads(2);device=torch.device('cuda:0')
    v=next(v for v in load(ROOT/'manifests/all_sources.json')['videos'] if v['kind']=='vimeo')
    raw,_,_=native_frames(v);assert len(raw)==7
    frames=[torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float()/255 for x in raw]
    sys.path.insert(0,str(ROOT.parent/'gvcrt_parallel_v5a3_v60'));from engine import Runtime
    runtime=Runtime(dict(experiment='B',methods={'original':[None,None]},checkpoints={},force_zero_thres=.12),'original',device)
    rows=[]
    for q in (0,9):
        base=ROOT/'parts/codec_smoke'/f'vimeo_q{q}'
        r,perframe=runtime.run(frames,q,base.with_suffix('.bin'),base.with_suffix('.npz'))
        assert r['frames']==7 and r['independent_decode_pass'] and r['real_RANS']
        assert all(math.isfinite(r[m]) for m in ('LPIPS','DISTS','FloLPIPS','PSNR','SSIM','MS_SSIM'))
        write(base.with_suffix('.frames.csv'),perframe);dump(base.with_suffix('.json'),r)
        rows.append(dict(QP=q,real_bytes=r['real_bytes'],frames=7,independent_decode_pass=True,metrics_finite=True));print('VIMEO SMOKE',q,'PASS',flush=True)
    dump(ROOT/'codec_smoke_audit.json',dict(status='PASS',sample_id=v['sample_id'],rows=rows,formal_codec_selection_affected=False,excluded_from_formal_profiles=True))
if __name__=='__main__':main()
