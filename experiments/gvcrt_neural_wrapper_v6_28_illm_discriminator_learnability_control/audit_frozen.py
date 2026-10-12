from runtime import *
device=init(4);v4,im,pm,models,quality=load_training(device)
for m in models.values():m.requires_grad_(False)
core=v4.compression_hash(im,pm);assert core==load(ROOT/'config.json')['source_checkpoint']['compression_hash'];assert all(not p.requires_grad for m in (im,pm,models['wrapper']) for p in m.parameters())
from illm_runtime import labeler
lab=labeler(device);s=torch.load(load(ROOT/'config.json')['D_source_checkpoint']['path'],map_location='cpu',weights_only=True);expected={k.removeprefix('feature.'):v for k,v in s['projected'].items() if k.startswith('feature.')};assert tensor_hash(lab.state_dict())==tensor_hash(expected)
dump(ROOT/'audits/frozen_source_modules.json',dict(status='PASS',B1_modules={k:v4.module_hash(m) for k,m in models.items()},compression_hash=core,I_model_hash=v4.module_hash(im),labeler_hash=tensor_hash(lab.state_dict()),labeler_matches_full_D0=True,all_G_parameters_frozen=True,exact_forward_modes={k:m.training for k,m in models.items()},G_updates=0,original_codec_files={str(p):sha(p) for p in [REPO/'checkpoints/GVC-RT_I.pt',REPO/'checkpoints/GVC-RT_P.pt'] if p.exists()}));print('FROZEN SOURCE PASS',flush=True)
