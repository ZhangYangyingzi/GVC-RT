"""Read-only authorized-GPU mapping and disk snapshot, no process control."""
from common_s2 import *
def main():
    env();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4,5,6,7'
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==4
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows={int(x.split(',')[0]):[y.strip() for y in x.split(',')] for x in raw.splitlines()};devices=[]
    for rank in range(4):
        record=rows[4+rank];uuid=str(torch.cuda.get_device_properties(rank).uuid)
        assert uuid.removeprefix('GPU-')==record[1].removeprefix('GPU-')
        devices.append(dict(logical_rank=rank,physical_GPU=4+rank,torch_uuid=uuid,nvidia_record=record))
    dump(ROOT/'audits/GPU_mapping.json',dict(status='PASS',devices=devices,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        disk=disk(),other_processes_not_terminated=True,time=time.time()))
    print('GPU_MAPPING_AND_DISK_PASS',devices)
if __name__=='__main__':main()
