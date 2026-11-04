"""Read-only active-v3 checkpoint, validation, and publication status."""
from v3_common import *
def main():
 r=load(ROOT/'run_status.json');takeover=load(MIG/'takeover.json');runtime=load(MIG/'runtime_selection.json') if (MIG/'runtime_selection.json').exists() else None
 arms={}
 for arm in ARMS:
  n=committed(arm) if latest_path(arm).exists() else takeover['frozen_checkpoints'][arm]['additional_step'];s=early_state(arm)
  arms[arm]=dict(committed_additional_updates=n,absolute_optimizer_step=5000+n,stop_reason=s['stop_reason'],best_dev_step=s['best_raw']['step'],best_dev_M=s['best_raw']['M'],bad_check_count=s['bad_check_count'],next_validation_step=None if s['stop_reason'] else min((s['last_validation_step']//500+1)*500,12710))
 selected=load(ROOT/'selected_models_v3.json') if (ROOT/'selected_models_v3.json').exists() else None
 batches=[]
 for f in (ROOT/'evaluation_v3').glob('*/run_status.json'):
  data=load(f)
  if data.get('status')=='PASS':batches.append(dict(id=f.parent.name,points=data['completed_points']))
 output=dict(status=r['status'],current_stage=r.get('stage'),protocol_revision=V3,experiment_directory=str(ROOT),data_plan_frozen=True,N_A=923,N_B=8470,max_additional_updates=12710,arms=arms,runtime_selection=runtime,K_common=selected['K_common'] if selected else None,completed_evaluation_batches=batches,final_expected_points=selected['expected_final_points'] if selected else None,publication=load(ROOT/'publication_status_v3.json') if (ROOT/'publication_status_v3.json').exists() else dict(status='PENDING'),report_path=str(ROOT/'REPORT.md'),resume_command=f'{PYTHON} -B -u {ROOT}/launch_v3.py --resume')
 print(json.dumps(output,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
