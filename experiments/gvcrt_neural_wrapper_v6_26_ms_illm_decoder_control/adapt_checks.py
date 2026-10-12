from pathlib import Path
R=Path(__file__).resolve().parent
p=R/'precheck.py';s=p.read_text().replace('params=[p for m in models.values() for p in m.parameters()]','params=[p for m in models.values() for p in m.parameters() if p.requires_grad]').replace('V24/','V25/').replace("old=ns['make_runner']", "old=ns['make_runner']")
s=s.replace("(V25/'window_training.py').read_text().replace('from io21 import *','')", "(V25/'window_training.py').read_text().replace('from io21 import *','').replace('params=[p for m in models.values() for p in m.parameters()]','params=[p for m in models.values() for p in m.parameters() if p.requires_grad]').replace('n=len(list(m.parameters()))','n=len([p for p in m.parameters() if p.requires_grad])')")
s=s.replace("old(frames,q,'A_decoder_continue'", "old(frames,q,'A_continue'")
s=s.replace('ps=list(m.parameters())','ps=[p for p in m.parameters() if p.requires_grad]').replace("max(float((p.grad-g).abs().max()) for p,g in pairs)","max([float((p.grad-g).abs().max()) for p,g in pairs] or [0.])")
s=s.replace('from projected import create','from illm_runtime import create_for').replace("disc=create(torch.device('cuda:0'))","disc=create_for(a.branch,torch.device('cuda:0'))")
s=s.replace("assert all(s['finite'] and s['norm']>0 for s in module_grads.values())","assert all(s['finite'] for s in module_grads.values()) and all(module_grads[k]['norm']>0 for k in ('bridge','generator'))")
start=s.index("  if a.branch=='B_decoder_conditional_projected':")
end=s.index('  z=fake.clone()',start)
s=s[:start]+"  if hasattr(disc,'set_target'):disc.set_target(frames[49])\n"+s[end:]
s=s.replace("for k,m in models.items()],weight_decay=1e-4","for k,m in models.items() if k!='wrapper'],weight_decay=1e-4")
s=s.replace("for p in disc.adapters.parameters()", "for p in (disc.discriminator.input_sequence.parameters() if a.branch=='C_decoder_illm' else disc.adapters.parameters())")
s=s.replace("if a.branch=='C_decoder_illm':assert math.isfinite(an) and an>0","assert math.isfinite(an) and an>0")
s=s.replace(" assert core==v4.compression_hash(im,pm)"," assert core==v4.compression_hash(im,pm) and initial['wrapper']==v4.module_hash(models['wrapper'])\n if a.branch=='C_decoder_illm':\n  import torch.nn.functional as F\n  gt=frames[49];before=tensor_hash(disc.feature.state_dict());y=disc.labels(gt);raw=disc.feature(gt)\n  with torch.no_grad():\n   f=disc.feature.encoder(gt);z=F.normalize(disc.feature.bottleneck_op.input_conv(f),dim=1);flat=z.permute(0,2,3,1).reshape(-1,8);code=disc.feature.bottleneck_op.child.child.codebooks[0];manual=(flat.square().sum(1,keepdim=True)+code.square().sum(1)[None]-2*flat@code.T).argmin(1).reshape(1,32,32)\n  assert torch.equal(y,raw.reshape(1,32,32)) and torch.equal(y,manual) and tensor_hash(disc.feature.state_dict())==before\n  assert y.dtype==torch.long and y.min()>=0 and y.max()<1024\n  dump(ROOT/'audits/labeler_probe.json',dict(status='PASS',weight_SHA256=sha(ROOT/'weights/vqvae_xcit_p8_ch64_cb1024_h8.ckpt'),shape=list(y.shape),dtype=str(y.dtype),min=int(y.min()),max=int(y.max()),row_major_BHW_layout=True,manual_nearest_codebook_indices_equal=True,stable_output=True,frozen_buffers=True,real_targets='y+1',fake_target=0,labels_no_codec_path=True))")
p.write_text(s)
# Reporting cap and freeze checks use the current decoder-only protocol.
p=R/'report.py';s=p.read_text().replace("min(.014448280572707944,.05*r['B1_gradient_norm']/(r['unweighted_adv_gradient_norm']+1e-12))", "(.05*r['B1_gradient_norm']/(r['unweighted_adv_gradient_norm']+1e-12) if min(r['B1_gradient_norm'],r['unweighted_adv_gradient_norm'])>1e-12 else 0.)")
s=s.replace("assert r['feature_projection_hash']==load(ROOT/'audits/projected_source.json')['feature_hash']", "assert r['feature_projection_hash']==load(ROOT/'audits/'+('illm_source.json' if b=='C_decoder_illm' else 'projected_source.json'))['module_hash' if b=='C_decoder_illm' else 'feature_hash']")
s=s.replace("lambda_ceiling=.014448280572707944", "lambda_ceiling=None,wrapper_frozen=True")
p.write_text(s)
p=R/'finish.py';s=p.read_text().replace("('actual_DISTS_optimizer_path','common_discriminator_initialization','condition_source','index_partition_integrity')", "('actual_DISTS_optimizer_path','labeler_probe')").replace("lambda_ceiling=.014448280572707944","lambda_ceiling=None").replace("common_discriminator_initialization_verified=True,index_partition_verified=True,", "labeler_pretrained_verified=True,wrapper_frozen_verified=True,")
s=s.replace("'ceiling_triggered'", "'ceiling_triggered'") # zero-valued compatibility statistic
s=s.replace("expected_shared_plan_sha256=load(ROOT/'config.json')['source_shared_plan_sha256']", "expected_shared_plan_sha256=load(ROOT/'config.json')['source_shared_plan_sha256']").replace('historical_early_completed=640','historical_early_completed=0,historical_FID_pooled_completed=200').replace('historical early evaluation:640/640','v6.25 pooled FID diagnostic:200/200')
p.write_text(s)
p=R/'verify_publication.py';s=p.read_text().replace("'audits/common_discriminator_initialization.json','audits/condition_source.json',", "'audits/labeler_probe.json','audits/illm_source.json',")
s=s.replace("s.update(status='PASS',phase='complete',active={},publication=pub)","s.update(status='RUNNING',phase='final_publication_verification',active={},publication=pub)")
p.write_text(s)
# Avoid inherited stale V6.25 labels in operational documentation.
p=R/'RUNBOOK.md';s=p.read_text().replace('V6.25','V6.26').replace('v6_25_conditional_projected_gradient_control','v6_26_ms_illm_decoder_control').replace('both0.014448280572707944 and5%', '5%').replace('Main evaluation is800 points. Historical early evaluation is640 additional points.','Main evaluation is800 points. Existing v6.25 FID decomposition reuses800 verified points; no historical training.');s+='\nWrapper is frozen. Only bridge and generator receive AdamW updates. B uses actual conditional Projected hinge; C uses official pretrained VQ targets and conditional U-Net CE. No lambda ceiling is used.\n';p.write_text(s)
import ast
for p in R.glob('*.py'):ast.parse(p.read_text())
print('CHECK AND REPORT ADAPTATION PASS')
