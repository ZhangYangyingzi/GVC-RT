from pathlib import Path
import json,shutil,hashlib,ast
R=Path(__file__).resolve().parent;V25=R.parent/'gvcrt_neural_wrapper_v6_25_conditional_projected_gradient_control'
rename={'A_continue':'A_decoder_continue','B_capped_projected':'B_decoder_conditional_projected','C_capped_conditional_projected':'C_decoder_illm'}
for p in R.glob('*.py'):
 if p.name in ('setup_adaptation.py','illm_runtime.py'):continue
 s=p.read_text().replace('v625','v626').replace('V625','V626')
 for old,new in rename.items():s=s.replace(old,new)
 p.write_text(s)
for p in V25.glob('train_manifest*.json'):shutil.copy2(p,R/p.name)
shutil.copy2(V25/'qp_semantics_audit.json',R/'qp_semantics_audit.json')
for n in ('data_split_audit.json','historical_test_exclusion.json','condition_source.json','gradient_field_semantics.json'):
 if (V25/'audits'/n).exists():shutil.copy2(V25/'audits'/n,R/'audits'/('inherited_'+n))
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
c=json.loads((R/'config.json').read_text());c.update(schema='v626',branches=list(rename.values()),methods=['original','B1_initial',*[b+'_1000' for b in rename.values()]],source_experiment='v625_protocol_v622_B1_initial',historical_expected_new_points=0,condition_modes={b:'actual_decoded_P' for b in list(rename.values())[1:]},lambda_ceiling=None,lambda_adv=None,frozen_wrapper=True,trainable_modules=['bridge','generator'],gradient_zero_threshold=1e-12)
c['optimizer'].pop('wrapper_lr',None);c['objective']='full B1 plus window-wise norm-controlled adversarial supervision for B/C; rho0.05, no lambda ceiling';save(R/'config.json',c)
p=json.loads((R/'protocol.json').read_text());p.update(schema='v626',methods=c['methods'],source_experiment='v625',objective=c['objective'],frozen_wrapper=True,trainable_modules=['bridge','generator'],lambda_ceiling=None,adversarial_control='0.05*norm(fullB1 bridge+generator gradient)/(norm(unweighted GAN gradient)+1e-12)',illm=dict(official_commit='3f122808255a2278eea049119367eb3abc434862',local_classes=1025,fake_class=0,real_targets='y+1',context_in=1024,context_out=16,output_downsampling_factor=8,norm_type='identity',image_input='RGB[0,1]',labels_input='GT RGB[0,1]',labels_do_not_enter_codec=True));save(R/'protocol.json',p)
p=R/'io21.py';s=p.read_text().replace("V24=ROOT.parent/", "V25=ROOT.parent/'gvcrt_neural_wrapper_v6_25_conditional_projected_gradient_control'\nV24=ROOT.parent/",1)
s=s.replace("EVAL_METHODS=(*METHODS,*(b+'_0' for b in BRANCHES),*(f'historical_{b}_{s}' for b in ('A_decoder_continue','B_projected') for s in (250,500,1000)))","EVAL_METHODS=(*METHODS,*(b+'_0' for b in BRANCHES),'v625_A_continue_1000','v625_B_capped_projected_1000','v625_C_capped_conditional_projected_1000')")
s=s.replace('for folder in (V24,','for folder in (V25,V24,').replace("    ids={id(p) for m in models.values() for p in m.parameters()}","    models['wrapper'].eval().requires_grad_(False)\n    ids={id(p) for m in models.values() for p in m.parameters() if p.requires_grad}")
p.write_text(s)
p=R/'window_training.py';s=p.read_text().replace('params=[p for m in models.values() for p in m.parameters()]','params=[p for m in models.values() for p in m.parameters() if p.requires_grad]').replace("                if branch=='B_decoder_conditional_projected':condition=torch.zeros_like(condition)","                if discriminator is not None and hasattr(discriminator,'set_target'):discriminator.set_target(frame)")
s=s.replace("adv_weight=min(cfg['lambda_ceiling'],cfg['rho']*n0/(na+cfg['gradient_eps'])) if discriminator is not None and requested_weight is None else (requested_weight or 0.)","adv_weight=(cfg['rho']*n0/(na+cfg['gradient_eps']) if min(n0,na)>cfg['gradient_zero_threshold'] else 0.) if discriminator is not None and requested_weight is None else (requested_weight or 0.)")
s=s.replace("ceiling_triggered=float(discriminator is not None and adv_weight==cfg['lambda_ceiling'])","ceiling_triggered=0.").replace("n=len(list(m.parameters()))","n=len([p for p in m.parameters() if p.requires_grad])")
s=s.replace("        if mode!='calibrate':","        sums['gradient_control_zero_safety']=float(min(n0,na)<=cfg['gradient_zero_threshold']) if discriminator is not None else 0.\n        if mode!='calibrate':")
p.write_text(s)
p=R/'train.py';s=p.read_text().replace('params=[p for m in models.values() for p in m.parameters()]','params=[p for m in models.values() for p in m.parameters() if p.requires_grad]').replace("for k,m in models.items()],weight_decay=oc['weight_decay']","for k,m in models.items() if k!='wrapper'],weight_decay=oc['weight_decay']").replace("'gradient_tools.py','projected.py'","'gradient_tools.py','projected.py','illm_runtime.py'").replace('from projected import create','from illm_runtime import create_for').replace("disc=create(torch.device('cuda:0'))","disc=create_for(a.branch,torch.device('cuda:0'))")
s=s.replace("assert all(g['finite'] and g['effective_parameters']>0 for g in gs.values())","assert all(g['finite'] for g in gs.values()) and all(gs[k]['effective_parameters']>0 for k in ('bridge','generator')) and gs['wrapper']['norm']==0")
s=s.replace("assert core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality]","assert core==v4.compression_hash(im,pm) and qh==[v4.module_hash(m) for m in quality] and initial['wrapper']==v4.module_hash(models['wrapper'])")
s=s.replace("frozen_compression_unchanged=True,quality_frozen_unchanged=True","wrapper_frozen_unchanged=True,frozen_compression_unchanged=True,quality_frozen_unchanged=True")
p.write_text(s)
# Evaluation config contains only B1 + validated v625 historical models initially.
e=json.loads((V25/'evaluation/config.json').read_text());e.update(schema='v626',methods=c['methods'],expected_points=800);e['checkpoints']={k:v for k,v in e['checkpoints'].items() if k=='B1_initial'};e['deployment_receiver_hashes']={k:v for k,v in e['deployment_receiver_hashes'].items() if k=='B1_initial'}
old=json.loads((V25/'evaluation/config.json').read_text())
for k in ('A_continue_1000','B_capped_projected_1000','C_capped_conditional_projected_1000'):
 e['checkpoints']['v625_'+k]=old['checkpoints'][k];e['deployment_receiver_hashes']['v625_'+k]=old['deployment_receiver_hashes'][k]
save(R/'evaluation/config.json',e)
# Depend only on immutable historical inputs, then freeze the new protocol.
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
deps=json.loads((V25/'audits/dependencies.json').read_text())
for name in ('io21.py','adapter.py','window_training.py','config.json','protocol.json','plans/shared.json'):deps[str(V25/name)]=sha(V25/name)
for p in (R/'vendor').rglob('*.py'):deps[str(p)]=sha(p)
deps[str(R/'weights/vqvae_xcit_p8_ch64_cb1024_h8.ckpt')]=sha(R/'weights/vqvae_xcit_p8_ch64_cb1024_h8.ckpt');save(R/'audits/dependencies.json',deps)
files=['config.json','protocol.json','plans/shared.json',*[str(p.relative_to(R)) for p in (R/'manifests').glob('*.json')],*[p.name for p in R.glob('train_manifest*.json')]];save(R/'audits/protocol_hashes.json',{n:sha(R/n) for n in files})
assert sha(Path(c['source_checkpoint']['path']))=='dc76f0e221dc7aaa5e5f45f648722584ce226e65828ce24151f9961985f3f6f7'
assert (R/'plans/shared.json').read_bytes()==(V25/'plans/shared.json').read_bytes() and sha(R/'plans/shared.json')=='e83011570849e027883fc9e34e7bf04907e79dabbcfb89382a50571f774902db'
save(R/'audits/preflight.json',dict(status='PASS',source_verified=True,plan_byte_equal=True,plan_sha256=sha(R/'plans/shared.json'),no_environment_upgrade=True))
for p in R.glob('*.py'):ast.parse(p.read_text())
print('ADAPTATION SETUP PASS')
