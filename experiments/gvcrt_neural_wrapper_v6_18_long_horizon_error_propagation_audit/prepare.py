"""Checkpoint, dataset, causal-code and historical SHA preflight; no training."""
import ast,traceback
from io18 import *
def main():
    import numpy as np,torch
    from PIL import Image
    torch.set_num_threads(2);command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    for d in ('audits','logs','manifests','evaluation/points','evaluation/source_artifacts','results','plots','parts'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    if (ROOT/'preflight_audit.json').exists() and load(ROOT/'preflight_audit.json')['status']=='PASS':frozen();print('PREFLIGHT ALREADY PASS');return
    before={}
    for folder in (V15,V16,V17):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.pid','.tmp'):before[str(p)]=sha(p)
        print('HISTORICAL SHA SNAPSHOT',folder.name,flush=True)
    deps=load(V17/'audits/dependencies.json')
    for p,h in deps.items():assert sha(p)==h,('historical dependency mismatch',p);before[p]=h
    for folder in (V15,V16,V17):assert load(folder/'final_integrity.json')['status']=='PASS'
    cfg=load(V17/'evaluation/config.json');assert tuple(cfg['methods'])==METHODS
    cp_audit={};canonical={}
    for m,folder in [('mixed_1000',V15),('dists_fixed_1000',V16),('dists_w05_1000',V17)]:
        idx=load(folder/'checkpoint_index.json');fidx=load(folder/'frozen_checkpoint_index.json');rec=idx['1000']
        assert fidx['status']=='PASS' and fidx['checkpoints']['1000']==rec
        if 'checkpoint_index_sha256' in fidx:assert fidx['checkpoint_index_sha256']==sha(folder/'checkpoint_index.json')
        assert rec['adaptation_step']==1000 and rec['compression_hash']==cfg['compression_hash']
        assert cfg['checkpoints'][m]==rec['inference'];canonical[m]=rec
    canonical['v62_initial']=dict(path=str(V62/'branches/schedule_s1p0/checkpoints/step_1000.pt'),**{k:v for k,v in cfg['checkpoints']['v62_initial'].items() if k!='path'})
    for m in METHODS[1:]:
        rec=canonical[m];runtime=cfg['checkpoints'][m]
        assert Path(rec['path']).is_absolute() and sha(rec['path'])==rec['sha256']
        payload=torch.load(rec['path'],map_location='cpu',weights_only=True)
        hashes={k:tensor_hash(payload[k]) for k in ('wrapper','bridge','generator')};assert hashes==rec['module_hashes']==runtime['module_hashes']
        if m=='v62_initial':assert payload['step']==1000
        else:assert payload['adaptation_step']==1000
        assert sha(runtime['path'])==runtime['sha256'];rs=torch.load(runtime['path'],map_location='cpu',weights_only=True)
        assert {k:tensor_hash(rs[k]) for k in hashes}==hashes
        deployed={k:tensor_hash({n:t.half() for n,t in rs[k].items()}) for k in ('bridge','generator')};assert deployed==cfg['deployment_receiver_hashes'][m]
        cp_audit[m]=dict(absolute_path=rec['path'],SHA256=rec['sha256'],**{k+'_hash':v for k,v in hashes.items()},compression_core_hash=cfg['compression_hash'],inference_export=runtime,FP16_receiver_hashes=deployed,index_source=str((V62/'branches/schedule_s1p0/config.json') if m=='v62_initial' else Path(rec['path']).parents[1]/'checkpoint_index.json'))
        before[rec['path']]=rec['sha256'];before[runtime['path']]=runtime['sha256'];del payload,rs
        print('CHECKPOINT VERIFIED',m,flush=True)
    cp_audit['original']=dict(absolute_path={'I':str(REPO/'checkpoints/GVC-RT_I.pt'),'P':str(REPO/'checkpoints/GVC-RT_P.pt')},SHA256={k:sha(REPO/f'checkpoints/GVC-RT_{k}.pt') for k in ('I','P')},wrapper_hash=None,wrapper='identity; no wrapper adaptation',bridge_hash=cfg['original_receiver_hashes']['bridge'],generator_hash=cfg['original_receiver_hashes']['generator'],compression_core_hash=cfg['compression_hash'],I_module_hash=cfg['I_hash'])
    dump(ROOT/'checkpoint_integrity.json',dict(status='PASS',methods=list(METHODS),checkpoints=cp_audit,full_step_checkpoint_and_inference_export_tensor_equality=True))
    cfg.update(schema='v618_long_horizon_error_propagation_audit',no_training=True,methods=list(METHODS),datasets=list(DATASETS),external_qps=list(QPS),frames_per_point=64,expected_points=240)
    cfg['checkpoints']={m:cfg['checkpoints'][m] for m in METHODS[1:]};cfg['deployment_receiver_hashes']={m:cfg['deployment_receiver_hashes'][m] for m in METHODS[1:]}
    dump(ROOT/'config.json',cfg);q=load(V17/'qp_semantics_audit.json');dump(ROOT/'qp_semantics_audit.json',q)
    dataset_audit=[]
    for d in DATASETS:
        man=load(V17/'manifests'/f'{d}.json');vs=man['videos']
        assert len(vs)=={'ulong':8,'uvg_holdout':2,'uvg_validation':1,'hevc_b':5}[d]
        for old in (V15,V16):assert vs==load(old/'manifests'/f'{d}.json')['videos'],('manifest mismatch',d,old)
        if d=='uvg_holdout':assert {v['name'] for v in vs}=={'HoneyBee','Jockey'}
        if d=='uvg_validation':assert vs[0]['name']=='Bosphorus'
        for v in vs:
            assert v['frames']==len(v['source_frame_indices'])==64 and (v['width'],v['height'])==(1920,1080)
            assert v['rate_accounting_fps']>0
            if d=='hevc_b':assert v['rate_accounting_fps']=={'BasketballDrive':50,'BQTerrace':60,'Cactus':50,'Kimono1':24,'ParkScene':24}[v['name']]
            sp=Path(v['source_path']);assert sp.is_absolute() and sha(sp)==v['source_sha256'];before[str(sp)]=v['source_sha256']
            digest=hashlib.sha256()
            for i in range(64):
                p=Path(v['input_dir'])/(f'im{i+1:05d}.png' if d=='hevc_b' else f'frame_{i:06d}.png')
                ph=sha(p);assert ph==v['frame_png_sha256' if d=='hevc_b' else 'frame_file_sha256'][i];before[str(p)]=ph
                with Image.open(p) as im:assert im.mode=='RGB' and im.size==(1920,1080);raw=np.asarray(im,dtype=np.uint8).tobytes()
                assert hashlib.sha256(raw).hexdigest()==v['frame_rgb_sha256'][i];digest.update(raw)
            assert digest.hexdigest()==v['rgb_sha256']
            dataset_audit.append(dict(dataset=d,sequence=v['name'],source_path=v['source_path'],source_SHA256=v['source_sha256'],source_frame_indices=v['source_frame_indices'],frames=64,rate_accounting_fps=v['rate_accounting_fps'],source_fps=v.get('source_fps_metadata',v['rate_accounting_fps']),RGB_SHA256=v['rgb_sha256'],canvas=[1920,1088],crop=[1920,1080],padding='replicate to multiples of 64',color_conversion='historical canonical RGB; FP16 -> replicate pad -> 2*x-1',verified_against=[str(x/'manifests'/f'{d}.json') for x in (V15,V16,V17)]))
            print('SOURCE FRAMES VERIFIED',d,v['name'],flush=True)
        dump(ROOT/'manifests'/f'{d}.json',man)
    dump(ROOT/'dataset_integrity.json',dict(status='PASS',sequences=16,source_frames=1024,records=dataset_audit,historical_manifest_equality=True))
    engine_source=(ENGINE/'engine.py').read_text();tree=ast.parse(engine_source);runtime=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Runtime');run=next(n for n in runtime.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    # Two sequence loops: encoding and persisted-stream independent decoding.
    loops=[n for n in run.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and getattr(n.iter.func,'id',None)=='enumerate' and ast.unparse(n.iter.args[0])=='frames'];assert len(loops)==2
    for loop in loops:
        for n in ast.walk(loop):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('clear_dpb','set_curr_poc'):
                enclosing=next((i for i in ast.walk(loop) if isinstance(i,ast.If) and any(x is n for x in ast.walk(i))),None)
                assert enclosing is not None and ast.unparse(enclosing.test)=='i == 0'
    assert "pa.decompress(bits,current,q)" in engine_source and "pdc.decompress(encoded['bit_stream'],sps,q)" in engine_source
    assert "pe.add_ref_frame(None,encoded['x_hat'])" in engine_source and "pdc.add_ref_frame(None,decoded['x_hat'])" in engine_source
    dump(ROOT/'audits/causal_state_code.json',dict(status='PASS',source=str(ENGINE/'engine.py'),SHA256=sha(ENGINE/'engine.py'),whole_sequence_encoding=True,persisted_stream_independent_decode=True,DPB_clear_only_initialization_or_I_frame=True,reconstructed_reference_used=True,teacher_forcing=False,AST_run_SHA256=hashlib.sha256(ast.dump(run).encode()).hexdigest()))
    dump(ROOT/'protocol.json',dict(no_training=True,comparison='diagnostic same-external-QP; not equal-rate',methods=list(METHODS),datasets=list(DATASETS),external_qps=list(QPS),frames_per_point=64,frame_index='one-based 1..64; historical artifacts zero-based 0..63',actual_QP='historical shift_qp mapping; inherited qp_semantics_audit.json',rate='complete persisted real RANS stream bytes * 8 / (64 * 1920 * 1080)',frame_bits='complete stream accounting; first frame includes SPS header; cumulative sum of real_bits',causal_path='unchanged historical Runtime.run; DPB retained across P frames; decoded reference only',independent_decode='fresh decoder over persisted RANS stream; reconstruction SHA equality and state sync required',codec_dtype='FP16',wrapper_dtype='FP32',force_zero_thres=.12,color_conversion='inherited canonical RGB; half -> replicate pad -> 2*x-1',canvas=[1920,1088],metric_crop=[1920,1080],FloLPIPS='historical transition from frame i-1 to frame i, assigned to target frame; frame1 undefined NaN',evaluation_DISTS='historical value-only evaluation; no backward',windows=['1-8','9-16','17-32','33-48','49-64'],std='population standard deviation ddof=0; undefined FloLPIPS frame1 excluded',drift='mean frames49-64 minus mean frames1-8',excess_drift='method drift minus original drift at same external QP',slope='ordinary least squares with intercept over frames2-64 for raw and method-original delta',aggregation='equal sequence weights within each dataset; datasets remain separate',reuse='exact method/export checkpoint hash, canonical frames, actual QPs, real bytes, code, metric versions, decode and synchronization artifacts verified'))
    for folder in (V15,V16,V17):
        for p in folder.glob('*.py'):deps[str(p)]=sha(p)
    deps.update({str(ENGINE/'engine.py'):sha(ENGINE/'engine.py'),str(BASE/'evaluate.py'):sha(BASE/'evaluate.py')})
    dump(ROOT/'audits/dependencies.json',deps);dump(ROOT/'audits/historical_sha256_before.json',before)
    dump(ROOT/'preflight_audit.json',dict(status='PASS',no_training=True,expected_points=240,checkpoints_verified=True,datasets_verified=True,causal_code_verified=True,historical_SHA_files=len(before),GPU_audit_pending=True))
    seal();print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(ROOT/'preflight_audit.json',dict(status='FAIL',error=traceback.format_exc()));raise
