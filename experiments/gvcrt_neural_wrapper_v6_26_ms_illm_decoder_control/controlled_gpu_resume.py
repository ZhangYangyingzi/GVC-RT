"""Resume only this experiment's own C worker at an atomic checkpoint boundary."""
import pathlib,json,os,signal,time,subprocess
R=pathlib.Path(__file__).resolve().parent
branch='C_decoder_illm'
def main():
 deadline=time.time()+180
 while time.time()<deadline:
  s=json.loads((R/'pipeline_status.json').read_text());w=s.get('active',{}).get('train_'+branch);t=json.loads((R/'branches'/branch/'training_status.json').read_text())
  if w and w['gpu']==6 and t.get('updated_unix') and time.time()-t['updated_unix']<1.5 and t['adaptation_step']>=130 and t['adaptation_step']%10==0:
   pid=w['PID'];proc=pathlib.Path('/proc')/str(pid);identity=(proc/'stat').read_text().rsplit(') ',1)[1].split()[19];argv=(proc/'cmdline').read_bytes().split(b'\0');assert identity==w['identity'] and str(R/'train.py').encode() in argv and branch.encode() in argv
   free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);values={int(a):int(b) for a,b in (x.split(',') for x in free.splitlines())};assert values[4]>=w['required_MiB']
   latest=R/'branches'/branch/'checkpoints/latest.pt';rows=(R/'branches'/branch/'training_logs/train.jsonl').read_text().splitlines()
   if latest.with_suffix('.tmp').exists() or json.loads(rows[-1])['adaptation_step']!=t['adaptation_step']:continue
   if time.time()-t['updated_unix']>=2:continue
   audit=dict(status='CONTROLLED_RESUME',branch=branch,previous_PID=pid,process_identity=identity,old_physical_gpu=6,reason='GPU6 shared B/C while GPU4 has sufficient free memory; move via persistent queue',checkpoint_boundary=t['adaptation_step'],formal_plan_unchanged=True,numerical_settings_unchanged=True,optimizer_RNG_and_cursor_in_latest=True,scheduler_will_resume=True,unix=time.time());(R/'audits/C_gpu_migration.json').write_text(json.dumps(audit,indent=2)+'\n');os.kill(pid,signal.SIGTERM);print('CONTROLLED RESUME',pid,t['adaptation_step'],flush=True);return
  time.sleep(.15)
 print('No safe boundary; original trainer preserved.',flush=True)
if __name__=='__main__':main()
