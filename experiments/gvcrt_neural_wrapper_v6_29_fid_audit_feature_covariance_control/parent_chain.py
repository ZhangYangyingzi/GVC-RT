from io21 import *
def main():
 import torch
 path=Path(load(ROOT/'config.json')['source_checkpoint']['path']);chain=[];seen=set()
 while path.exists() and path not in seen:
  seen.add(path);s=torch.load(path,map_location='cpu',weights_only=True);experiment=next((p for p in path.parents if p.parent==ROOT.parent),None);cfg=load(experiment/'config.json') if experiment and (experiment/'config.json').exists() else {};parent=cfg.get('source_checkpoint',cfg.get('initial_checkpoint'))
  rec=dict(path=str(path),sha256=sha(path),source_step=s.get('source_step'),adaptation_step=s.get('adaptation_step',s.get('step')),source_checkpoint_sha256=s.get('source_checkpoint_sha256'),source_experiment=s.get('source_experiment'),config_path=str(experiment/'config.json') if experiment else None,loss=cfg.get('objective'),source_definition=parent)
  chain.append(rec);del s
  nextpath=parent.get('path') if isinstance(parent,dict) else parent if isinstance(parent,str) else None
  if not nextpath:break
  p=Path(nextpath)
  if not p.is_absolute():p=REPO/p
  if rec['source_checkpoint_sha256'] and p.exists():assert sha(p)==rec['source_checkpoint_sha256']
  path=p
 dump(ROOT/'audits/actual_checkpoint_parent_chain.json',dict(status='PASS',records=chain,first_gap_stage='尚未定位首次出现阶段',evidence_limit='No claim inferred from experiment numbering. Intermediate pooled feature evidence is not established.'))
if __name__=='__main__':main()
