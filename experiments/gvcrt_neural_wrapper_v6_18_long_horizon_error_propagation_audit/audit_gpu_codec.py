"""Check live GPU memory and actual frozen original codec/metric modules."""
import argparse,traceback
from io18 import *
def main():
    a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=(4,5,6,7),required=True);args=a.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    torch.set_num_threads(2);frozen();live=gpu_snapshot();g=next(g for g in live if g['index']==args.gpu);assert g['free_MiB']>=14000
    command([PYTHON,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    from adapter import engine
    base=engine();runtime=base.Runtime(dict(experiment='B',methods={'original':[None,None]},checkpoints={},force_zero_thres=.12),'original',torch.device('cuda:0'))
    im,pm=runtime.models(1088,1920);cfg=evalcfg()
    assert base.core.compression_hash(im,pm)==cfg['compression_hash']
    assert runtime.wrapper is None and runtime.loaded_receiver_hashes==cfg['original_receiver_hashes']
    dump(ROOT/'audits/live_codec_modules.json',dict(status='PASS',gpu=args.gpu,start_GPU_snapshot=live,runtime_audit=runtime.audit,compression_parameters_frozen=all(not p.requires_grad for x in (im,pm) for p in x.parameters()),no_backward=True,no_optimizer=True,no_training=True))
    print('LIVE CODEC MODULES PASS GPU',args.gpu,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'audits/live_codec_modules.json',dict(status='FAIL',error=traceback.format_exc()));raise
