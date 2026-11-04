"""Freeze evaluation-only addendum; no checkpoint or training-source migration."""
from eval_parallel_common import *
import ast

def main():
    verify_sources();assert load(ROOT/'run_status.json')['status']=='CONTROL_PAUSED'
    assert load(EP/'safe_pause.json')['status']=='PASS'
    assert not (EP/'source_manifest.json').exists(),'Already prepared; verify and resume rather than overwrite'
    parent=load(ROOT/'development_baseline_v3.json');assert parent['checkpoint_sha256']==PARENT_SHA
    development=sorted(selected_videos(dict(kind='development'),'development'),key=lambda v:v['name']);names=[v['name'] for v in development[:8]]
    p=dict(addendum=ADDENDUM,scope='Evaluation scheduling and process count only',user_authorization='可以，那我觉得可以修改，请你修改',parent_protocol_sha256=sha(ROOT/'protocol_v3.json'),parent_source_manifest_sha256=sha(ROOT/'source_manifest_v3.json'),requested_workers_per_GPU=2,physical_GPUs=[4,5,6,7],training_implementation_changed=False,training_effective_batch=4,model_loss_precision_frames_QP_splits_and_stopping_unchanged=True,input_decode_worker_cap=4,worker_partition='Video index modulo total workers; all checkpoint/QP points for a video stay in one worker',worker_to_GPU='4 + floor(worker/workers_per_GPU)',GTFlow_setup_audit='Each worker uses its own output directory to avoid fixed audit-name collisions; fixed PWC source and weights unchanged',engineering_workload=dict(split='development',checkpoint='parent5000',sequences=names,QP=[0,5,9],frames=64,points=24),engineering_order=['4 workers reference','8 workers candidate','4 workers warmed reference'],metric_max_absolute_tolerance=1e-6,bitstream_reconstruction_source_and_per_frame_CSV_require_exact=True,minimum_speedup_to_adopt=1.05,throughput_comparison='Candidate must be at least1.05x faster than the fastest of both four-worker references; otherwise retain four workers',failed_correctness='Do not adopt; retain error; existing checkpoint and decisions untouched',disk_free_reserve_GiB=100,large_artifact_cap_GiB=20,private_media_cache_cap_GiB=80,early_stop_check_counts_unchanged=True,probe_does_not_generate_model_selection_decisions=True)
    dump(EP/'protocol.json',p)
    for label in ('reference_before','candidate','reference_after'):
        batch=dict(id='parallel_probe_'+label,kind='engineering',splits=['development'],states=[dict(id='parent5000',checkpoint_path=str(P/'checkpoints/step_5000.pt'),checkpoint_filename='step_5000.pt',checkpoint_sha256=PARENT_SHA,additional_step=0,absolute_step=5000,roles=['parent5000'])],protocol_sha256=sha(ROOT/'protocol_v3.json'),source_manifest_sha256=sha(ROOT/'source_manifest_v3.json'),evaluation_rows_sha256=load(ROOT/'protocol.json')['evaluation_rows_sha256'],QP=[0,5,9],fresh_encode=True,independent_decode=True,selection_sha256=None,engineering_sequences=names)
        dump(EP/f'probe_manifests/{label}.json',batch)
    generated=worker_source();ast.parse(generated);(EP/'generated_worker.py').write_text(generated)
    names=sorted(f.name for f in ROOT.glob('eval_parallel*.py'))+['launch_eval_parallel.py']
    for name in names:assert (ROOT/name).is_file();ast.parse((ROOT/name).read_text())
    dump(EP/'source_manifest.json',dict(addendum=ADDENDUM,protocol_v3_sha256=sha(ROOT/'protocol_v3.json'),source_manifest_v3_sha256=sha(ROOT/'source_manifest_v3.json'),addendum_sha256=sha(EP/'protocol.json'),code={name:sha(ROOT/name) for name in names},generated_worker_sha256=hashlib.sha256(generated.encode()).hexdigest(),frozen_before_benchmark=True))
    dump(EP/'freeze.json',dict(status='FROZEN',frozen_unix=time.time(),source_manifest_sha256=sha(EP/'source_manifest.json'),protocol_sha256=sha(EP/'protocol.json'),probe_quality_not_used_for_research_selection=True))
    print('Evaluation-only addendum and generated numerical-body adapter frozen.')

if __name__=='__main__':main()
