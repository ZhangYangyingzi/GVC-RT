"""Existing allowlist publication, plus independent verification of the evaluator addendum."""
from eval_parallel_common import *
import traceback

def main():
    verify_parallel_sources();import v3_publish
    assert load(EP/'benchmark.json')['correctness_status']=='PASS'
    v3_publish.main()
    phase=sys.argv[sys.argv.index('--phase')+1];state=load(ROOT/'publication_status_v3.json')
    commit=state['code_metadata_commit'] if phase=='code' else state['results_commit']
    pub=Path(load(ROOT/'audits/publication_v3_worktree.local.json')['path']);assert v3_publish.remote_head()==commit
    names=['eval_parallel.py','eval_parallel_controller.py','launch_eval_parallel.py','migrations/eval_parallel_v1/protocol.json','migrations/eval_parallel_v1/source_manifest.json','migrations/eval_parallel_v1/benchmark.json']
    for name in names:
        blob=subprocess.check_output(['git','show',f'{commit}:{v3_publish.REL}/{name}'],cwd=pub);assert hashlib.sha256(blob).hexdigest()==sha(ROOT/name)
    dump(EP/f'publication_{phase}.json',dict(status='PUBLISHED',phase=phase,branch=v3_publish.BRANCH,verified_commit=commit,commit_url=f'https://github.com/ZhangYangyingzi/GVC-RT/commit/{commit}',remote_verified=True,addendum_core_blobs_verified=True,core_files=names))

if __name__=='__main__':
    try:main()
    except Exception:
        phase=sys.argv[sys.argv.index('--phase')+1];dump(EP/f'publication_{phase}.json',dict(status='FAILED',error=traceback.format_exc(),remote_verified=False));traceback.print_exc();sys.exit(1)
