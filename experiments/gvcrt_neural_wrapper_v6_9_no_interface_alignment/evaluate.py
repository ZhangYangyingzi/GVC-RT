from v68_io import *
if __name__=='__main__':
    import torch
    eval_adapter().main()
    dump(ROOT/'parts/memory'/f'eval_{os.getpid()}.json',dict(status='PASS',peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated()))
