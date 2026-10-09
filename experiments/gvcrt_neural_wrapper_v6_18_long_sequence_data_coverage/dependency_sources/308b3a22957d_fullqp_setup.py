"""Frozen, hash-verified U-Long cohorts and real implementation QP audit."""
import ast
import collections
import concurrent.futures
import inspect
import math
import random
import shutil
import types
import zipfile
from fullqp_common import *

def preserve_previous():
    archive = ROOT / 'superseded_fixed_sweep_setup'
    archive.mkdir(exist_ok=True)
    for p in list(ROOT.glob('*.json')) + [ROOT / 'setup.py', ROOT / 'branch_train.py']:
        dest = archive / p.name
        if not dest.exists():
            shutil.copy2(p, dest)
    dump(ROOT / 'revision_audit.json', dict(revision=REVISION, legacy_branch_disposition='Retained in place, excluded from full-QP schedule experiment',
        invalidated_previous_claims=['Validation paths were empty', 'QP parity copied training values rather than exercising inference', 'Initial branch equality was not measured'],
        legacy_beta0='May finish its already-running job; never counted as a requested schedule branch',
        latest_request='/Huang_group/zyyz/home_dir/.codex/attachments/676e6fe7-eef6-441b-aa4c-8e8923e590db/已粘贴的文本.txt'))
    dump(ROOT / 'final_integrity.json', dict(status='PENDING', revision=REVISION, reason='Full-QP replacement requires new audits, five paired branches and real validation'))
    for name in ('preflight_audit', 'validation_split_integrity'):
        dump(ROOT / (name + '.json'), dict(status='PENDING', revision=REVISION))

def audit_qp():
    v4 = module('fullqp_v4', V4 / 'train.py')
    import gvc_hooks
    from src.models.video_model_gvcrt import DMC
    # Initialize the real CPU model so qp_shift is set by its actual constructor.
    model = DMC()
    engine_path = OLD_ENGINE / 'engine.py'
    tree = ast.parse(engine_path.read_text())
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'run')
    assignments = [n for n in ast.walk(run) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'q' for t in n.targets)]
    assert len(assignments) == 1
    expression = ast.Expression(assignments[0].value)
    expr = compile(ast.fix_missing_locations(expression), str(engine_path), 'eval')
    rows = []
    for q in range(10):
        train = [frame_qp(model, q, i) for i in range(4)]
        evaluation = [eval(expr, {'qp': q, 'i': i, 'pe': model, 'eco': types.SimpleNamespace(INDEX_MAP=gvc_hooks.INDEX_MAP)}) for i in range(4)]
        rows.append(dict(external_qp=q, I_QP=train[0], frame1_P_QP=train[1], frame2_P_QP=train[2], frame3_P_QP=train[3],
                         training=train, evaluation=evaluation, exact_match=train == evaluation))
    assert all(r['exact_match'] for r in rows)
    dump(ROOT / 'qp_train_eval_semantics_audit.json', dict(status='PASS', revision=REVISION, rows=rows,
        training_source=str(ROOT / 'fullqp_common.py'), shift_source=inspect.getfile(DMC.shift_qp),
        shift_source_sha256=sha(inspect.getfile(DMC.shift_qp)), index_source=gvc_hooks.__file__, INDEX_MAP=gvc_hooks.INDEX_MAP,
        evaluation_source=str(engine_path), evaluation_source_sha256=sha(engine_path),
        evaluation_expression=ast.unparse(expression), method='Execute real DMC.shift_qp and AST-extracted evaluator expression independently; no copied parity list',
        runtime_endpoint_smoke='Required per branch before optimizer updates'))

def cohort_selection(train, split):
    stats = read(V61 / 'training_source_statistics.csv')
    excluded = {v['sha256'] for v in train} | set(split['reserved_sha256']) | set(split['test_sha256']) | set(split['validation_sha256'])
    eligible = list({r['sha256']:r for r in stats if r['status']=='PASS' and r['sha256'] not in excluded}.values())
    population = [r for r in stats if r['status']=='PASS']
    weights = collections.Counter(int(r['stratum']) for r in population)
    quotas = {s: int(32 * n / len(population)) for s,n in weights.items()}
    remainder = sorted(weights, key=lambda s: (-(32*weights[s]/len(population)-quotas[s]), s))
    for s in remainder[:32-sum(quotas.values())]: quotas[s] += 1
    rng = random.Random(SEED)
    normal = []
    for s in sorted(quotas):
        group = sorted([r for r in eligible if int(r['stratum'])==s], key=lambda r:r['sha256'])
        normal.extend(dict(r, selection_category='proportional_stratum') for r in rng.sample(group, quotas[s]))
    remain = [r for r in eligible if r['sha256'] not in {x['sha256'] for x in normal}]
    fields = ('temporal_rgb_L1','temporal_rgb_MSE','sobel_edge_energy','laplacian_variance')
    ranks = {}
    for f in fields:
        ordered = sorted(remain, key=lambda r:(float(r[f]),r['sha256']))
        ranks[f] = {r['sha256']:(i+1)/len(ordered) for i,r in enumerate(ordered)}
    def score(r, category):
        z = [ranks[f][r['sha256']] for f in fields]
        return [z[0],z[1],(z[2]+z[3])/2,min(z[0],z[1],z[2])][category]
    hard = []; used = set()
    for category, label in enumerate(('temporal_L1','temporal_MSE','texture_Sobel_Laplacian','joint_temporal_texture')):
        choices = sorted([r for r in remain if r['sha256'] not in used], key=lambda r:(-score(r,category),r['sha256']))[:8]
        hard.extend(dict(r, selection_category=label, selection_score=score(r,category)) for r in choices)
        used.update(r['sha256'] for r in choices)
    dump(ROOT/'validation_selection_audit.json', dict(status='PASS', seed=SEED, eligible_count=len(eligible),
        population_count=len(population), population_strata=dict(weights), normal_strata=quotas,
        normal_rule='Largest-remainder proportional allocation to existing 4x4 temporal/texture strata, seeded uniform draw within each stratum',
        hard_rule='Eight disjoint sources each: temporal L1, temporal MSE, mean Sobel/Laplacian percentile, joint min(L1,MSE,Sobel) percentile',
        excluded_sha256=sorted(excluded), no_external_test_statistics=True))
    return {'normal':normal,'hard':hard}

def materialize(rows, split):
    out = []
    for i,r in enumerate(rows):
        dest = ROOT / 'validation' / 'sources' / (r['sha256']+'.mp4')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            with zipfile.ZipFile(r['archive']) as z:
                info = z.getinfo(r['filename'])
                assert info.file_size == int(r['bytes'])
                temp = dest.with_suffix('.partial')
                with z.open(info) as source, temp.open('wb') as target: shutil.copyfileobj(source,target,1<<20)
                assert sha(temp)==r['sha256']; temp.replace(dest)
        assert sha(dest)==r['sha256']
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate','-of','json',str(dest)]))['streams'][0]
        num,den = map(int,probe['avg_frame_rate'].split('/')); fps=num/den
        assert int(probe['width'])==int(r['width']) and int(probe['height'])==int(r['height'])
        assert int(probe['nb_frames'])==int(r['frames']) and int(r['frames'])>=32
        assert math.isclose(fps,float(r['fps']),rel_tol=1e-6)
        out.append(dict(dataset=split,video_index=i,name=Path(r['filename']).stem,path=str(dest),sha256=r['sha256'],
            archive=r['archive'],archive_member=r['filename'],frames=32,source_frames=int(r['frames']),
            width=int(r['width']),height=int(r['height']),fps=fps,rate_fps=fps,stratum=int(r['stratum']),
            selection_category=r['selection_category'],**{f:float(r[f]) for f in ('temporal_rgb_L1','temporal_rgb_MSE','sobel_edge_energy','laplacian_variance')}))
        print('EXTRACTED',split,i+1,flush=True)
    dump(ROOT/f'validation_{split}_manifest.json',dict(status='PASS',count=32,videos=out))
    return out

def main():
    if (ROOT/'fullqp_config.json').exists():
        check_gate(); print('SETUP ALREADY FROZEN'); return
    preserve_previous()
    source=V41/'checkpoints/beta_high/step_20000.pt'
    old={}
    for folder in (V4,V41,V52,V61,OLD_ENGINE,REPO/'src',REPO/'expericent_generation_input/expericent_interface_causal_controls_v9/src',ROOT.parent/'gvcrt_neural_wrapper_v2_joint'):
        for p in sorted(folder.rglob('*.py')): old[str(p)]=sha(p)
    for p in (source,V61/'train_manifest_1024.json',V61/'data_split_integrity.json',V61/'training_source_statistics.csv',V61/'checkpoint_hashes.json',REPO/'README.md'):
        old[str(p)]=sha(p)
    dump(ROOT/'fullqp_old_source_hashes.json',old)
    split=load(V61/'data_split_integrity.json'); manifest=V61/'train_manifest_1024.json'
    assert sha(manifest)==split['train_manifest_sha256']
    train=load(manifest)['videos']; assert len(train)==len({v['sha256'] for v in train})==1024
    def verify(v):
        actual=sha(v['path']); assert actual==v['sha256'], v['path']; return dict(path=v['path'],sha256=actual)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: verified=list(pool.map(verify,train))
    dump(ROOT/'training_source_hash_audit.json',dict(status='PASS',train_manifest_sha256=sha(manifest),videos=verified))
    print('TRAIN SOURCE HASHES PASS',flush=True)
    audit_qp()
    dump(ROOT/'gvc_lambda_schedule_audit.json',dict(status='PASS',whether_found_in_code=False,
        source='arXiv:2608.04891v1 section 6.1 + latest user specification section 4',
        source_url='https://arxiv.org/html/2608.04891v1#S6.SS1',
        evidence='Paper states QP 0 through 9 and interpolation between 0.08 and 0.9; it does not explicitly specify linear versus logarithmic interpolation. The linear rule here is the user-specified calibration protocol, NOT claimed to be recovered official code.',
        repository_evidence='README marks training code release incomplete; searched Python and project documentation, no official lambda schedule found',
        formula='0.08 + external_qp * (0.9 - 0.08) / 9',
        rows=[dict(external_qp=q,lambda_q=lambda_q(q),whether_found_in_code=False,source='paper endpoints + user linear formula') for q in range(10)]))
    selected=cohort_selection(train,split)
    cohorts={s:materialize(rows,s) for s,rows in selected.items()}
    sets={s:{r['sha256'] for r in rows} for s,rows in cohorts.items()}; sets['train']={r['sha256'] for r in train};sets['test']=set(split['test_sha256'])
    intersections={a+'_'+b:sorted(sets[a]&sets[b]) for a,b in (('train','normal'),('train','hard'),('normal','hard'),('normal','test'),('hard','test'))}
    assert not any(intersections.values())
    dump(ROOT/'validation_split_integrity.json',dict(status='PASS',counts={k:len(v) for k,v in sets.items()},intersections=intersections,all_sources_hashed=True,no_UVG=True,no_VIRAT=True))
    cfg=dict(revision=REVISION,seed=SEED,branches=list(BRANCHES),gpus=[4,5,6,7],source_checkpoint=str(source),
        source_checkpoint_sha256=sha(source),train_manifest=str(manifest),train_manifest_sha256=sha(manifest),
        updates=3000,checkpoint_steps=STEPS,validation_qps=QPS,clip_length=4,crop=256,batch=1,force_zero_thres=.12,lambda_proxy=.01,
        optimizer=dict(name='AdamW',wrapper_lr=5e-5,bridge_lr=3e-6,generator_lr=5e-7,weight_decay=1e-4,grad_clip=1.0),
        optional_beta0='Skipped: legacy beta0 retained as pilot, not certified by this revision; prioritize five required branches',
        validation_frames=32,validation_frames_basis='Reuse V6.1 validation protocol',comparison_anchor='original codec plus initial step0 secondary comparison',
        audit_hashes={f:sha(ROOT/f) for f in ('qp_train_eval_semantics_audit.json','gvc_lambda_schedule_audit.json','validation_normal_manifest.json','validation_hard_manifest.json','training_source_hash_audit.json','validation_split_integrity.json')})
    for name,scale in BRANCHES.items():
        br=ROOT/'branches'/name
        for d in ('checkpoints','logs','parts','results','validation'): (br/d).mkdir(parents=True,exist_ok=True)
        dump(br/'config.json',dict(**cfg,branch=name,scale=scale,fixed_or_schedule='fixed' if scale is None else 'schedule',fresh_optimizer=True))
        write(br/'qp_lambda_beta_mapping.csv',[dict(external_qp=q,lambda_q=lambda_q(q),beta_q=beta_q(name,q)) for q in range(10)])
    dump(ROOT/'fullqp_config.json',cfg)
    dump(ROOT/'paired_sampling_audit.json',dict(status='PENDING',seed=SEED,rule='Same seed, shared sample_clip then randrange(10); diagnostics use separate RNG; compare every completed per-update plan across branches'))
    dump(ROOT/'initialization_audit.json',dict(status='PENDING',reason='Await measured live initialization of all five branches'))
    assert_old_unchanged()
    dump(ROOT/'preflight_audit.json',dict(status='PASS',revision=REVISION,verified_training_files=1024,verified_validation_files=64,old_artifacts_unchanged=True))
    print('FULLQP SETUP PASS',flush=True)
if __name__=='__main__':
    try: main()
    except Exception as e:
        dump(ROOT/'fullqp_setup_failure.json',dict(status='FAIL',error=repr(e)));raise
