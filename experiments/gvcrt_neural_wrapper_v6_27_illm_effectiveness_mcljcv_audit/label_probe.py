from io21 import *
def main():
 import torch
 import torch.nn.functional as F
 from window_training import replay
 from illm_runtime import labeler,local_ce,WEIGHT
 torch.set_num_threads(2);row=load(ROOT/'diagnostic_plan.json')['updates'][0];frames=replay(row,'C_decoder_illm',torch.device('cpu'));gt=frames[49];model=labeler(torch.device('cpu'));before=tensor_hash(model.state_dict())
 with torch.no_grad():
  raw=model(gt);raw2=model(gt);y=raw.reshape(1,32,32).long();feature=model.encoder(gt);z=F.normalize(model.bottleneck_op.input_conv(feature),dim=1);flat=z.permute(0,2,3,1).reshape(-1,8);vq=model.bottleneck_op.child.child;cb=vq.codebooks[0];cb=F.normalize(cb,dim=1) if vq.normalize_codebook else cb;manual=(cb[None]-flat[:,None]).square().sum(2).argmin(1).reshape(1,32,32)
 assert torch.equal(raw,raw2) and torch.equal(y,manual) and y.shape==(1,32,32) and y.min()>=0 and y.max()<1024 and before==tensor_hash(model.state_dict())
 logits=torch.randn(1,1025,32,32,requires_grad=True);original=F.cross_entropy(logits,y+1);adapted=local_ce(logits,y+1);g1=torch.autograd.grad(original,logits,retain_graph=True)[0];g2=torch.autograd.grad(adapted,logits)[0];loss_error=abs(float(original-adapted));gradient_error=float((g1-g2).norm()/g1.norm());assert loss_error<1e-6 and gradient_error<1e-5
 dump(ROOT/'audits/label_layout_and_CE_integrity.json',dict(status='PASS',device='CPU independent replica',source_panel=row['panel_id'],frame=49,source_crop_SHA256=row['cropped_RGB_sha256'][49],weight_SHA256=sha(WEIGHT),shape=list(y.shape),dtype=str(y.dtype),min=int(y.min()),max=int(y.max()),layout='row-major BHW restored from official flattened indices',manual_nearest_normalized_codebook_identical=True,labels_repeat_identical=True,parameters_buffers_unchanged=True,target_shift_real='y+1 exactly once',target_fake=0,CE_reference_loss_error=loss_error,CE_reference_gradient_relative_error=gradient_error,cache_key='float RGB crop bytes SHA256 + pretrained weight SHA256',no_optimizer_step=True));print('LABEL AND CE PROBE PASS',flush=True)
if __name__=='__main__':main()
