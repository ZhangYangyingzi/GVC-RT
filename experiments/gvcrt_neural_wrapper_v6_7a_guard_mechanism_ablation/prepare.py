"""Read-only anchors and exact training-sequence controls before any GPU work."""
import ast,random,math
from v67_io import *
def state_hash(state):
    h=hashlib.sha256()
    for k,t in state.items():h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    if (ROOT/'preflight_audit.json').exists():frozen();return
    import torch
    torch.set_num_threads(2)
    for d in ('audits','logs','parts','training_logs','results','evaluation'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    for b in BRANCHES:
        for d in ('checkpoints','logs','parts'):(ROOT/'branches'/b/d).mkdir(parents=True,exist_ok=True)
    assert load(V64/'final_integrity.json')['status']==load(V66/'final_integrity.json')['status']=='PASS'
    cfg=load(V64/'config.json');cfg66=load(V66/'config.json')
    common=('source_checkpoint','source_checkpoint_sha256','seed','sample_seed','qp_seed','clip_length','crop','batch','force_zero_thres','lambda_proxy','optimizer','updates','checkpoint_steps','external_qps')
    assert all(cfg[k]==cfg66[k] for k in common)
    assert sha(cfg['source_checkpoint'])==cfg['source_checkpoint_sha256']=='c185e1f69fd5f249a4aa3753ddee1dc23d4402d5d6f74d18ef6b51b559fbdb73'
    oldcps=load(V66/'checkpoint_hashes.json');anchors={A:oldcps[A],B:oldcps[OLD_B]}
    assert anchors[A]['sha256']=='1c7c9a3f89cbbe4abd4ba6f6f7c963d2fab22737e9928fe222f3aa4c8a8c75e1'
    for cp in anchors.values():
        assert sha(cp['path'])==cp['sha256'];s=torch.load(cp['path'],map_location='cpu',weights_only=True)
        assert s['step']==1000 and {k:state_hash(s[k]) for k in ('wrapper','bridge','generator')}==cp['module_hashes'];del s
    source=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=True)
    init=load(V64/'initialization_audit.json')['branches']['vimeo_only'];assert source['step']==20000
    assert {k:state_hash(source[k]) for k in ('wrapper','bridge','generator')}==init['module_hashes'];del source
    assert load(V66/'branches'/OLD_B/'initialization_audit.json')['module_hashes']==init['module_hashes']
    bcal=load(V66/'audits/structure_guard_calibration.json');assert bcal['status']=='PASS' and .095<=bcal['resulting_gradient_ratio']<=.105
    bcode=(V66/'objective.py').read_text();assert 'struct = 1 - ssim' in bcode and 'ms_ssim_rgb(proxy,frame)' in bcode and "components['loss'] = components['loss'] + structure_weight*struct" in bcode
    assert load(V66/'branches'/OLD_B/'smoke_audit.json')['status']=='PASS'
    # Differentiable MS-SSIM implementation and deterministic padding are unchanged from V6.6.
    def ssim_implementation(path):
        tree=ast.parse(Path(path).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bind');nodes=[]
        for n in fn.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and ast.unparse(t)=="context['ms_ssim_rgb']" for t in n.targets):break
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('reflect_pad','DeterministicPad','checked_ssim'):nodes.append(ast.dump(n,include_attributes=False))
        return nodes
    assert len(ssim_implementation(ROOT/'objective.py'))==3 and ssim_implementation(ROOT/'objective.py')==ssim_implementation(V66/'objective.py')
    cfg.update(schema='v67a_guard_mechanism_v1',branches=list(BRANCHES),anchors=anchors,guard_target_ratio=.1,calibration_clips=32,no_test_selection=True,gpu_policy='Physical GPU 4/5/6/7 sharing allowed by latest user instruction; sufficient free memory required; never terminate other experiments')
    dump(ROOT/'config.json',cfg)
    for name in ('source_manifest.json','qp_semantics_audit.json','qp_sequence_audit.json','qp_train_eval_semantics_audit.json'):dump(ROOT/name,load(V64/name))
    dump(ROOT/'audits/source_checkpoint_audit.json',dict(status='PASS',source_path=cfg['source_checkpoint'],source_sha256=cfg['source_checkpoint_sha256'],initialization=init,anchors=anchors,source_module_hashes_pass=True,fresh_optimizer_required=True))
    dump(ROOT/'audits/B_definition_audit.json',dict(status='PASS',checkpoint=anchors[B],lambda_B=bcal['lambda_struct'],calibration_gradient_ratio=bcal['resulting_gradient_ratio'],objective='L_main + 0.01*L1(P(x),x) + lambda_B*(1-MS_SSIM(P(x),x))',objective_source=str(V66/'objective.py'),objective_sha256=sha(V66/'objective.py'),calibration_sha256=sha(V66/'audits/structure_guard_calibration.json'),not_retrained=True,differentiable_msssim_AST_equal=True))
    arows=[json.loads(x) for x in (V64/'training_logs/vimeo_only.jsonl').read_text().splitlines()];brows=[json.loads(x) for x in (V66/'training_logs'/f'{OLD_B}.jsonl').read_text().splitlines()]
    assert len(arows)==len(brows)==1000
    manifest=load(V64/'manifests/vimeo_official_train.json');rng=random.Random(cfg['sample_seed']);qr=random.Random(cfg['qp_seed']);qps=load(V64/'qp_sequence_audit.json')['sequence'];plans=[]
    for i,(ar,br) in enumerate(zip(arows,brows)):
        plan=dict(update=i+1,sample_id=rng.choice(manifest['sequences']),start=rng.randrange(4),crop_x=rng.randrange(193),crop_y=rng.randrange(1),external_qp=qr.randrange(10))
        assert all(ar[k]==br[k]==z for k,z in plan.items()) and qps[i]==plan['external_qp']
        assert ar['frame_sha256']==br['frame_sha256'] and ar['actual_qps']==br['actual_qps'];plans.append(plan)
    calrows=read(V66/'audits/structure_guard_calibration_samples.csv');assert len(calrows)==32
    for r,ref in zip(calrows,arows):
        assert r['sample_id']==ref['sample_id'] and all(int(r[k])==ref[k] for k in ('start','crop_x','crop_y','external_qp'))
        assert ast.literal_eval(r['frame_sha256'])==ref['frame_sha256']
    dump(ROOT/'audits/training_sequence_equivalence.json',dict(status='PASS',updates=1000,branches=list(BRANCHES),reference_A=str(V64/'training_logs/vimeo_only.jsonl'),reference_B=str(V66/'training_logs'/f'{OLD_B}.jsonl'),anchors_equivalent=True,planned_sample_sequence_identical=True,planned_qp_sequence_identical=True,plans=plans,formal_logs_pass=False))
    deps=load(V66/'audits/dependencies.json')
    for folder in (V64,V66,V65,V62B,V4,ENGINE):
        for p in folder.glob('*.py'):deps[str(p)]=sha(p)
    for p in [Path(cfg['source_checkpoint']),*[Path(c['path']) for c in anchors.values()],V66/'checkpoint_hashes.json',V66/'audits/structure_guard_calibration.json',V66/'audits/structure_guard_calibration_samples.csv',V66/'training_logs'/f'{OLD_B}.jsonl',V66/'final_integrity.json',V64/'initialization_audit.json']:deps[str(p)]=sha(p)
    for p,h in deps.items():assert sha(p)==h
    dump(ROOT/'audits/dependencies.json',deps);dump(ROOT/'audits/old_inventory.json',old_inventory())
    dump(ROOT/'audits/write_scope.json',dict(allowed_output_root=str(ROOT),all_implementation_writes_confined=True,static_inventory='Completed reference experiments only; unrelated concurrently running experiments are not dependencies and are not touched'))
    dump(ROOT/'audits/local_protocol.json',{str(p.relative_to(ROOT)):sha(p) for p in [*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'source_manifest.json',ROOT/'qp_sequence_audit.json',ROOT/'qp_semantics_audit.json']})
    dump(ROOT/'preflight_audit.json',dict(status='PASS',anchor_checkpoint_and_module_hashes_pass=True,sample_sequence_1000_pass=True,calibration_plans_32_pass=True,shared_msssim_implementation_pass=True))
    dump(ROOT/'final_integrity.json',dict(status='PENDING'));print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()
