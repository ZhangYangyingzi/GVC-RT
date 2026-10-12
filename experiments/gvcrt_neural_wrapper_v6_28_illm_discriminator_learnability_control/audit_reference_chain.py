from runtime import *
from window_training import replay
device=init(4);row=load(ROOT/'plans/validation.json')['updates'][0];frames=replay(row,'C_decoder_illm',device);v4,im,pm,models,quality=load_training(device)
for m in models.values():m.requires_grad_(False)
original={k:v4.module_hash(m) for k,m in models.items()};core=v4.compression_hash(im,pm);assert core==load(ROOT/'config.json')['source_checkpoint']['compression_hash'];legacy=V27/'cache/diagnostic/B1'/f"{row['panel_id']}.pt";s=torch.load(legacy,map_location='cpu',weights_only=True);cap=[];lat=[];hh=pm.dec.register_forward_hook(lambda m,a,o:cap.append(o));lh=models['bridge'][0].register_forward_pre_hook(lambda m,a:lat.append(a[0].detach()));common=module('reference_chain_qp',V62/'fullqp_common.py');records=[]
with torch.no_grad():
 pm.clear_dpb();pm.set_curr_poc(0);pm.add_ref_frame(None,im.compress(v4.codec_input(frames[0]),row['external_qp'])['x_hat'])
 for pos in range(1,64):
  qa=common.frame_qp(pm,row['external_qp'],pos);rec,_,_=v4.joint_ste_forward(pm,models['wrapper'](frames[pos]),qa);c=lat.pop();pm.add_ref_frame(cap.pop().detach(),(rec.detach()*2-1).half())
  if pos>=49:
   i=pos-49;r=rec.cpu();cc=c.cpu();same_r=torch.equal(r,s['reconstructions'][i]);same_c=torch.equal(cc,s['conditions'][i]);records.append(dict(position=pos,reconstruction_equal=same_r,condition_equal=same_c,reconstruction_max_abs_error=float((r-s['reconstructions'][i]).abs().max()),condition_max_abs_error=float((cc-s['conditions'][i]).abs().max()),actual_QP=qa));assert same_r and same_c,(pos,records[-1])
hh.remove();lh.remove();assert original=={k:v4.module_hash(m) for k,m in models.items()} and core==v4.compression_hash(im,pm)
dump(ROOT/'audits/fresh_B1_v627_hash_equivalence.json',dict(status='PASS',panel_id=row['panel_id'],legacy_path=str(legacy),legacy_sha256=sha(legacy),frames=records,model_modules=original,compression_hash=core,parameters_buffers_unchanged=True,G_updates=0,new_full_codec_evaluation=False,scope='cropped fixed training-forward window only; no RD evaluation'));print('FRESH B1 HASH EQUIVALENCE PASS',flush=True)
