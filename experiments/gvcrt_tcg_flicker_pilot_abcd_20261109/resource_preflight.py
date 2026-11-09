from common_s2 import *

def snapshot():
    env();assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7';assert os.environ['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
    rows=list(csv.reader(raw.splitlines()));cards={int(r[0]):dict(uuid=r[1].strip(),name=r[2].strip(),total_MiB=int(r[3]),free_MiB=int(r[4])) for r in rows}
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    conflicts=[];own=[]
    for r in csv.reader(processes.splitlines()):
        if len(r)<3:continue
        pid=int(r[0]);uuid=r[1].strip()
        if uuid not in {cards[i]['uuid'] for i in range(4,8)}:continue
        try:cwd=Path(f'/proc/{pid}/cwd').resolve();args=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        p=dict(pid=pid,uuid=uuid,used_MiB=r[2].strip(),cwd=str(cwd),args=args)
        (own if cwd==ROOT else conflicts).append(p)
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==4
    mapping=[dict(logical_GPU=i,physical_GPU=i+4,CUDA_name=torch.cuda.get_device_name(i),**cards[i+4]) for i in range(4)]
    result=dict(status='WAITING_GPU' if conflicts else 'PASS',devices=mapping,own_processes=own,other_processes=conflicts,disk=disk(),time=time.time(),no_other_GPU_used=True)
    dump(ROOT/'audits/resource_preflight_latest.json',result);return result

if __name__=='__main__':print(json.dumps(snapshot()))
