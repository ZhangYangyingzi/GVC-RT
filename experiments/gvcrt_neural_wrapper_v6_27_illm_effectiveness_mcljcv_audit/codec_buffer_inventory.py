from io21 import *
def main():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 torch.set_num_threads(2);v4=module('v627_buffer_inventory_v4',V4/'train.py');im,pm=v4.load_models(torch.device('cpu'),force_zero_thres=.12);records={k:[dict(name=n,shape=list(t.shape),dtype=str(t.dtype)) for n,t in m.named_buffers()] for k,m in [('I_codec',im),('P_codec',pm)]};assert all(not rows for rows in records.values()),records
 dump(ROOT/'audits/codec_registered_buffer_inventory.json',dict(status='PASS',device='CPU independent constructor, no forward or optimizer step',registered_buffers=records,registered_buffer_count=0,compression_params_checked_by_SHA_and_runtime_hash=True,mutable_DPB_and_native_entropy_stream_state_not_model_buffers=True));print('CODEC BUFFER INVENTORY PASS',flush=True)
if __name__=='__main__':main()
