"""Append objective evaluator runtime facts to the unchanged numerical report."""
from eval_parallel_common import *

def main():
    verify_parallel_sources();import v3_finalize
    v3_finalize.main();benchmark=load(EP/'benchmark.json')
    facts=dict(addendum=ADDENDUM,selected_workers_per_GPU=benchmark['selected_workers_per_gpu'],evaluator_processes=benchmark['selected_evaluator_processes'],engineering_speedup_vs_fastest_reference=benchmark['speedup_vs_fastest_reference'],engineering_workload='Parent5000, eight already evaluated development sequences, QP0/5/9,64 frames;4/8/4 processes',training_implementation_changed=False,checkpoint_migration_required=False,source_manifest_sha256=sha(EP/'source_manifest.json'),protocol_sha256=sha(EP/'protocol.json'))
    report=ROOT/'REPORT.md';report.write_text(report.read_text()+'\nEvaluation scheduling addendum (engineering workload timings only):\n\n```json\n'+json.dumps(facts,indent=2)+'\n```\n')

if __name__=='__main__':main()
