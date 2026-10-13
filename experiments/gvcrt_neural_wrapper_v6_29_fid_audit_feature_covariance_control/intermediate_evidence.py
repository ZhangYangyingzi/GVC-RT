from io21 import *
from fid_statistics import lowrank_unclamped
import numpy as np,statistics

def main():
 rows=[];missing=[]
 for root,m in ((V20,'native_i_adapt_1000'),(V21,'B_1000'),(V22,'B1_reconstruction_1000')):
  cfg=load(root/'evaluation/config.json');cp=cfg['checkpoints'][m];assert sha(cp['path'])==cp['sha256']
  for d in ('ulong','hevc_b'):
   for q in (0,5,9):
    gt=[];rec=[];refs=[]
    for v in sources(d):
     candidates=[root/'parts/ip_-1'/d/m/f'video_{v["video_index"]:02d}_qp{q}.json',root/'parts'/d/m/f'video_{v["video_index"]:02d}_qp{q}.json']
     p=next((p for p in candidates if p.exists()),None)
     if p is None:missing.append(dict(stage=root.name,sequence=v['name'],QP=q,expected=[str(x) for x in candidates]));break
     r=load(p);assert r['checkpoint_sha256']==cp['sha256'];assert r['source_frame_indices']==v['source_frame_indices'] and r['source_rgb_sha256']==v['rgb_sha256'];assert r['real_RANS'] and r['independent_decode_pass'] and r['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(q)][:64]
     with np.load(r['feature_path']) as f:gt.append(f['real']);rec.append(f['reconstruction'])
     refs.append(dict(path=str(p),sha256=sha(p),features_path=r['feature_path'],features_sha256=sha(r['feature_path'])))
    else:
     g=np.concatenate(gt);b=np.concatenate(rec);dm=float(np.square(g.astype(np.float64).mean(0)-b.astype(np.float64).mean(0)).sum());fid=lowrank_unclamped(g,b)
     rows.append(dict(stage=root.name,method=m,dataset=d,QP=q,FID=fid,D_mu=dm,D_cov=fid-dm,samples=len(g),checkpoint_sha256=cp['sha256'],point_sources=refs,comparison='same external QP stage evidence; not equal rate; does not identify first historical onset'))
 write(ROOT/'results/FID_audit/intermediate_stages.csv',rows or [dict(status='missing')]);dump(ROOT/'audits/intermediate_stage_evidence.json',dict(status='PASS',available=len(rows),missing=missing,first_onset='尚未定位首次出现阶段',scope='Only protocol-matched available native-I stages; no rerun of historical experiments'))
if __name__=='__main__':main()
