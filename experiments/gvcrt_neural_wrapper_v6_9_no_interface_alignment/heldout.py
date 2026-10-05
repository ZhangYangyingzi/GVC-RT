from v68_io import *
if __name__=='__main__':
    adapted_script('heldout').main()
    import torch
    dump(ROOT/'parts/memory'/f'heldout_{os.getpid()}.json',dict(status='PASS',peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated()))
