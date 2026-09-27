"""Freeze inputs and audit official QP semantics before any new evaluation."""
import ast
import json
import re
import shutil
import subprocess
from audit_utils import *

def literal_assignment(tree,name):
    return next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))

def main():
    assert not (ROOT/'config.json').exists(),'Preparation already frozen; do not overwrite'
    assert load(V5/'final_integrity.json')['status']=='PASS'
    paths={};evidence={}
    for method in METHODS:
        r=load(V5/f'parts/final/{method}/video_0_qp0.json')
        path=Path(r['checkpoint']) if r['checkpoint'] else None
        if path:assert path.is_file() and sha(path)==r['checkpoint_sha256']
        paths[method]={'path':str(path) if path else '', 'sha256':sha(path) if path else ''}
        evidence[method]=str(V5/f'parts/final/{method}/video_0_qp0.json')
    assert paths['v41_20000']['path']==load(V5/'config.json')['source_checkpoint']
    assert paths['clip4_control']['path'].endswith('/step_7000.pt') and paths['clip8']['path'].endswith('/step_3000.pt')
    common=REPO/'src/models/common_model.py';video=REPO/'src/models/video_model_gvcrt.py';official=REPO/'test_video_gvcrt_1088.py'
    tree=ast.parse(common.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_qp_num')
    qps=ast.literal_eval(next(n.value for n in ast.walk(fn) if isinstance(n,ast.Return)))
    shift=literal_assignment(ast.parse(video.read_text()),'qp_shift')
    index=literal_assignment(ast.parse(official.read_text()),'index_map')
    assert qps==10 and shift==[0,2,1] and index==[0,1,0,2,0,2,0,2]
    for scale in ('q_scale_enc','q_scale_dec','q_scale_feature','q_scale_recon'):
        assert re.search(r'self\.'+scale+r' = nn.Parameter\(torch.ones\(\(self.get_qp_num\(\) \+ extra_qp',video.read_text())
    assert "curr_qp = args['qp_i']" in official.read_text()
    assert "curr_qp = p_frame_net.shift_qp(args['qp_p'], fa_idx)" in official.read_text()
    core=V4/'eval_core.py';text=core.read_text()
    assert 'actual_qp = qp if frame_index == 0 else p_encoder.shift_qp(qp, INDEX_MAP[frame_index % 8])' in text
    hooks=REPO/'expericent_generation_input/expericent_interface_causal_controls_v9/src/gvc_hooks.py'
    assert literal_assignment(ast.parse(hooks.read_text()),'INDEX_MAP')==index
    dump('qp_semantics_audit.json',dict(status='PASS',get_qp_num=qps,supported_external_qps=list(range(qps)),qp_shift=shift,
        internal_qp_tensor_length={s:qps+max(shift) for s in ('q_scale_enc','q_scale_dec','q_scale_feature','q_scale_recon')},
        index_map=index,official_i_frame_qp_behavior='actual I QP = external qp_i; first frame only in this experiment',
        official_p_frame_qp_behavior='actual P QP = external qp_p + qp_shift[index_map[frame_index % 8]]',
        external_i_and_p_base_equal=True,actual_internal_qp_range=[0,11],no_periodic_I_reset=True,adaptive_reset_disabled=True,
        evaluation_core=str(core),source_hashes={str(p):sha(p) for p in (common,video,official,core,hooks)},
        note='index_map cycles do not reset the causal DPB; internal QP10/11 are not external rate points'))
    for name in ('test_manifest.json','metric_implementation_audit.json'):
        shutil.copyfile(V5/name,ROOT/name)
    inventory=[];found=[];excluded=[]
    for path in sorted(UVG.rglob('*')):
        if not path.is_file():continue
        inventory.append({'path':str(path),'bytes':path.stat().st_size})
        if path.suffix.lower()!='.yuv':
            excluded.append({'path':str(path),'reason':'archive or non-frame auxiliary file; no downloads/extraction performed'});continue
        match=re.fullmatch(r'(.+)_(\d+)x(\d+)_(\d+)fps_420_8bit_YUV',path.stem)
        assert match,('unrecognized raw format',str(path))
        name,width,height,fps=match.groups();width=int(width);height=int(height);fps=int(fps)
        if (width,height)!=(1920,1080):excluded.append({'path':str(path),'reason':'not 1920x1080'});continue
        size=width*height*3//2;assert path.stat().st_size%size==0
        count=path.stat().st_size//size;assert count>=2
        actual=min(96,count)
        probe=['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pixel_format','yuv420p','-video_size',f'{width}x{height}',
               '-framerate',str(fps),'-i',str(path),'-frames:v',str(actual),'-f','null','-']
        subprocess.run(probe,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        found.append(dict(sequence_name=name,path=str(path),width=width,height=height,fps=fps,num_frames=count,
            frames_evaluated=actual,source_format='raw planar YUV420p 8-bit',pixel_format='yuv420p',sha256=sha(path),
            metadata_source='explicit filename format plus exact file-size divisibility; ffmpeg successfully reads evaluation prefix',
            file_bytes=path.stat().st_size))
    assert found and len({v['sequence_name'] for v in found})==len(found)
    repo_manifest=load(REPO/'UVG_rgb.json');repo_names=list(repo_manifest['test_classes']['UVG']['sequences'])
    dump('uvg_available_manifest.json',dict(status='PASS',root=str(UVG),num_available_uvg_sequences=len(found),
        sequence_names=[v['sequence_name'] for v in found],videos=found,inventory=inventory,excluded_files=excluded,
        total_evaluated_frames=sum(v['frames_evaluated'] for v in found),repo_UVG_rgb_sha256=sha(REPO/'UVG_rgb.json'),
        repo_sequence_names=repo_names,repo_manifest_is_complete=set(repo_names)=={v['sequence_name'] for v in found},
        frozen_before_evaluation=True))
    train=load(V5/'train_manifest.json');serialized=json.dumps(train).lower()
    assert 'uvg' not in serialized
    dump('uvg_zero_shot_audit.json',dict(status='PASS',training_used_uvg=False,checkpoint_selection_used_uvg=False,
        hyperparameter_tuning_used_uvg=False,scope='current wrapper fine-tuning and this evaluation-only experiment; no claim about original pretrained backbone datasets',
        checkpoint_paths_frozen=paths,training_manifest_sha256=sha(V5/'train_manifest.json'),
        uvg_manifest_sha256=sha(ROOT/'uvg_available_manifest.json')))
    protocol_files=[V4/'eval_core.py',V41/'metric_runtime.py',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py',
        V5/'evaluate.py',V5/'test_manifest.json',V5/'config.json',V5/'final_rd_points.csv',V5/'final_qp_summary.csv',V5/'final_integrity.json']
    dump('config.json',dict(schema='V5-A.1 evaluation only',evaluation_only=True,gpus=[4,5,6,7],external_qps=list(range(10)),
        checkpoints=paths,checkpoint_path_evidence=evidence,source_hashes={str(p):sha(p) for p in protocol_files},
        ulong_manifest_sha256=sha(ROOT/'test_manifest.json'),uvg_manifest_sha256=sha(ROOT/'uvg_available_manifest.json'),
        metric_audit_sha256=sha(ROOT/'metric_implementation_audit.json'),training_allowed=False,
        processing_canvas=[1920,1088],metric_crop=[1920,1080],FID_pooling='dataset x method x external QP, all actual source frames',
        SSIM_definition='unchanged V4/V4.1 global RGB SSIM implementation; not windowed SSIM',
        numeric_tolerance=2e-6))
    for d in ('logs','parts','bitstreams','features','rd_curves/ulong','rd_curves/uvg'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    dump('pipeline_status.json',dict(status='PREPARED',phase='QP0-3 compatibility pending',evaluation_only=True))
    print('PREPARE PASS',len(found),'UVG sequences',sum(v['frames_evaluated'] for v in found),'frames',flush=True)

if __name__=='__main__':main()
