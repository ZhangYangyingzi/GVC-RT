"""Read-only baseline checks; an exact SHA mismatch prohibits model loading/training."""
from pathlib import Path
import hashlib,json,os,shutil,time

E=Path(__file__).resolve().parent
PROJECT=Path('/Huang_group/zyyz/Projects/GVC-RT')
PREVIOUS=PROJECT/'GVC-RT_worktrees/tcg_stage2_history_gate_joint_20261106/experiments/gvcrt_tcg_stage2_history_gate_joint_20261106'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(name,data):
    path=E/name;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)

def main():
    previous=json.loads((PREVIOUS/'protocol.json').read_text())
    requested=json.loads((E/'requested_protocol.json').read_text())
    pilot=PROJECT/'experiments/gvcrt_tcg_stage1_pilot/checkpoints/adaptation_step_0.pt'
    baseline=Path(previous['baseline_checkpoint'])
    checks=[]
    for label,path,expected in [('pilot_receiver',pilot,requested['pilot_sha256']),('B1500_TCG',baseline,requested['baseline_sha256']),
        ('I_codec',PROJECT/'checkpoints/GVC-RT_I.pt','c8a840415cc028307131c66f68076b7e9f6fcc4db7fed423c62e8b202f83a22d'),
        ('P_codec',PROJECT/'checkpoints/GVC-RT_P.pt','7a89bd342a1484b3da7dfc3f03834ff81e0e4180e02f8a286dec0d1025c99f4e')]:
        actual=sha(path) if path.is_file() else None
        checks.append(dict(component=label,path=str(path),size_bytes=path.stat().st_size if path.exists() else None,
            expected_sha256=expected,actual_sha256=actual,status='PASS' if actual==expected else 'FAILED'))
    failures=[c for c in checks if c['status']!='PASS']
    resume=f'/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/bin/python -B -u {E}/preflight.py'
    result=dict(status='BLOCKED' if failures else 'PREFLIGHT_SHA_PASS',checks=checks,time=time.time(),
        previous_protocol_sha256=sha(PREVIOUS/'protocol.json'),previous_protocol_pilot_sha256=previous['receiver_initialization']['sha256'],
        formal_model_loaded=False,formal_training_started=False,committed_optimizer_updates=0,
        blocker='PILOT_SHA_MISMATCH' if failures else None,
        resume_precondition='User confirms a correction of the requested pilot SHA, or supplies the exact matching file; do not bypass the check' if failures else 'Further implementation/engineering checks required before training',
        resume_command=resume,resume_command_scope='Read-only preflight only; no training process is launched',disk_free_bytes=shutil.disk_usage(E).free)
    write('preflight.json',result)
    write('status.json',dict(**result,experiment_directory=str(E),current_stage='baseline_SHA_verification',
        expected_test_points=135,completed_test_points=0,best_dev_step=None,experiment_complete=False,runner_running=False))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures:raise SystemExit(2)

if __name__=='__main__':main()
