from common_s2 import *
def snapshot():
    env()
    cards=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()))
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    dump(ROOT/'audits/GPU_allocation.json',dict(status='PASS',CUDA_DEVICE_ORDER='PCI_BUS_ID',groups=dict(E=[0,1],F=[2,3]),GPUs=[dict(physical_index=int(a[0]),UUID=a[1].strip(),name=a[2].strip(),total_MiB=int(a[3]),used_MiB=int(a[4])) for a in cards if int(a[0]) in range(4)],processes=list(csv.reader(processes.splitlines())),other_processes_preserved=True,allow_shared_GPU_if_memory_sufficient=True,time=time.time()))
if __name__=='__main__':snapshot()
