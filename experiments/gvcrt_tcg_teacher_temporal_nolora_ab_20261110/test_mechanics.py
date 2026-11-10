"""Synthetic engineering fixtures only; never used as experimental points."""
from common_s2 import *

def main():
    env();import selection as s
    task_root=ROOT;fixture=ROOT/'tmp/mechanical_fixtures';fixture.mkdir(parents=True,exist_ok=True);s.ROOT=fixture
    def measured(model,step,temporal,spatial=1.):
        for q in [0,5,9]:
            for i in range(32):dump(fixture/f'evaluation/development/{model}/step_{step:04d}/points/synthetic{i}_qp{q}.json',dict(status='SYNTHETIC_TEST_ONLY',result=dict(dataset='SYNTHETIC_TEST_ONLY',model=model,step=step,sequence=f'synthetic{i}',QP=q,Sobel_MCResidual_PtoP=temporal,LPIPS_VGG=spatial,DISTS=spatial)))
        cp=fixture/model/f'training/checkpoints/step_{step:04d}.pt';cp.parent.mkdir(parents=True,exist_ok=True);cp.write_bytes(b'SYNTHETIC_NOT_A_MODEL')
    if (fixture/'selection.json').exists():(fixture/'selection.json').unlink()
    for ref in ['B300_original_reference','B300_noLoRA_step0']:measured(ref,0,1.)
    for g in GROUPS:
        measured(g,50,1.1);measured(g,100,1.1)
    a=s.decide(100);assert a['stop'] and all(v['step']==50 and not v['qualified'] and v['qualification']=='NOT_QUALIFIED' and v['patience']==2 for v in a['groups'].values())
    (fixture/'selection.json').unlink()
    for g in GROUPS:
        measured(g,50,.9);measured(g,100,.8997);measured(g,200,.8996)
    a=s.decide(200);assert a['stop'] and all(v['step']==200 and v['qualified'] and v['patience']==2 for v in a['groups'].values())
    (fixture/'selection.json').unlink()
    for g in GROUPS:measured(g,50,.9998)
    a=s.decide(50);assert all(v['qualified'] and v['patience']==1 for v in a['groups'].values())
    (fixture/'selection.json').unlink()
    for g in GROUPS:measured(g,50,.8,1.02)
    a=s.decide(50);assert not a['stop'] and all(not v['qualified'] for v in a['groups'].values())
    import numpy as np,media
    from videos import open_lossless
    media.ROOT=fixture;seq='SYNTHETIC_TEST_ONLY';roi=dict(x=0,y=0,width=16,height=16,first=48,last=79);dump(fixture/'visual_ROIs.json',{seq:roi});(fixture/'tmp').mkdir(exist_ok=True);records=[]
    for i,m in enumerate(media.MODELS):
        p=fixture/f'{m}.mkv';w=open_lossless(p,16,16,30);frames=np.stack([np.full((16,16,3),i*20+j%20,np.uint8) for j in range(128)]);w.stdin.write(frames.tobytes());w.stdin.close();assert w.wait()==0;records.append(dict(model=m,path=str(p),step=50,fps=30))
    a=media.montage(seq,records);assert a['native_pixels_unchanged'] and a['frames']==32
    import results
    results.ROOT=fixture
    final=[];development=[]
    for j,m in enumerate(results.FINAL_MODELS):
        for q in [0,5,9]:
            values=dict(PSNR=30+j*.1,LPIPS_Alex=.2+j*.001,LPIPS_VGG=.3+j*.001,DISTS=.1+j*.001,FloLPIPS=.2+j*.001,Sobel_MCResidual_PtoP=.01+j*.001)
            final.append(dict(dataset='SYNTHETIC_TEST_ONLY',sequence='synthetic',model=m,QP=q,bpp=.1/(q+1),**values))
    for g in GROUPS:
        for q in [0,5,9]:
            for i in range(32):development.append(dict(model=g,QP=q,step=0,LPIPS_VGG=.3,DISTS=.1,Sobel_MCResidual_PtoP=.01))
    results.plots(dict(aggregate=final,final=final,dev=development))
    assert len(load(fixture/'plots/manifest.json')['plots'])==15
    dump(task_root/'audits/mechanical_tests.json',dict(status='PASS',synthetic_only=True,not_experiment_metrics=True,tests=['NOT_QUALIFIED fallback uses real trained checkpoint; earliest tie','qualified patience respects min_delta and selects lowest actual S','per_QP spatial guard rejects 2pct deterioration','six-model FFV1 ROI montage preserves original pixels and frame numbering','dataset/per-video/step plot paths and six separate metric axes'],time=time.time()))
    print('MECHANICAL_SELECTION_AND_LOSSLESS_ROI_PASS')
if __name__=='__main__':main()
