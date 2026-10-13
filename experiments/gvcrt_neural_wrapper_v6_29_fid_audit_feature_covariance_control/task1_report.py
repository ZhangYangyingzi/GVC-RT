"""Independent audit completion and factual gap tables, separate from training benefit."""
from io21 import *
import numpy as np
from fid_statistics import stats
from audit_statistics import historical_point

def main():
 for n in ('FID_statistics_integrity','FID_historical_reproduction','FID_invariances_integrity','FID_gap_integrity'):
  assert load(ROOT/'audits'/f'{n}.json')['status']=='PASS'
 files=list((ROOT/'audits/features').glob('*/*/qp*.json'));assert len(files)==18 and all(load(p)['status']=='PASS' for p in files)
 raw=read(ROOT/'results/FID_audit/fid_decomposition_raw.csv');groups=read(ROOT/'results/FID_audit/frame_group_statistics.csv');lv=read(ROOT/'results/FID_audit/leave_one_video_out.csv');offsets=[];text='# Independent FID audit\n\nExecution:PASS. Features18/18; explicit covariance statistics18/18; full GT self-distance/symmetry and analytic Gaussian checks passed. This does not constitute a statistical-estimator reliability guarantee.\n\n## Same-QP matched pooled differences\n\nDeltas=B1−original on identical frame UIDs. These are diagnostic same-QP comparisons, not equal-rate gain claims.\n\n|Dataset|QP|Delta FID|Delta D_mu|Delta D_cov|\n|---|---:|---:|---:|---:|\n'
 for d in ('ulong','hevc_b','mcljcv'):
  for q in (0,5,9):
   a=next(x for x in raw if x['dataset']==d and x['QP']==str(q) and x['method']=='original');b=next(x for x in raw if x['dataset']==d and x['QP']==str(q) and x['method']=='B1_initial')
   text+=f'|{d}|{q}|'+ '|'.join(f"{float(b[k])-float(a[k]):.9g}" for k in ('FID','D_mu','D_cov'))+'|\n'
   pools={m:[] for m in ('original','B1_initial')}
   for v in sources(d):
    for m in pools:
     r=load(historical_point(d,m,v['video_index'],q))
     with np.load(r['feature_path']) as f:pools[m].append(f['reconstruction'].astype(np.float64))
    ma,ca=stats(pools['original'][-1]);mb,cb=stats(pools['B1_initial'][-1]);diff=cb-ca;diag=np.diag(diff)
    offsets.append(dict(dataset=d,QP=q,scope=v['name'],samples=64,mean_difference_L2=float(np.linalg.norm(ma-mb)),covariance_trace_difference=float(np.trace(cb-ca)),diagonal_squared_difference=float(np.square(diag).sum()),off_diagonal_squared_difference=float(np.square(diff).sum()-np.square(diag).sum()),comparison='direct B1 versus original descriptor difference; not a decomposition of covariance FID'))
   ma,ca=stats(np.concatenate(pools['original']));mb,cb=stats(np.concatenate(pools['B1_initial']));diff=cb-ca;diag=np.diag(diff);offsets.append(dict(dataset=d,QP=q,scope='pooled_dataset',samples=64*len(sources(d)),mean_difference_L2=float(np.linalg.norm(ma-mb)),covariance_trace_difference=float(np.trace(diff)),diagonal_squared_difference=float(np.square(diag).sum()),off_diagonal_squared_difference=float(np.square(diff).sum()-np.square(diag).sum()),comparison='direct B1 versus original descriptor difference; not a decomposition of covariance FID'))
 write(ROOT/'results/FID_audit/direct_original_B1_offsets.csv',offsets)
 text+='\n## Fixed frame-group differences\n\nOnly equal-identity, equal-count original/B1 pairs within a group are compared. Different group sample counts do not by themselves establish temporal drift.\n\n|Dataset|QP|Group|Samples|Delta FID|Delta D_mu|Delta D_cov|\n|---|---:|---|---:|---:|---:|---:|\n'
 for d in ('ulong','hevc_b','mcljcv'):
  for q in (0,5,9):
   for label in ('I0','P1_16','P17_32','P33_48','P49_63'):
    a=next(x for x in groups if (x['dataset'],x['QP'],x['group'],x['method'])==(d,str(q),label,'original'));b=next(x for x in groups if (x['dataset'],x['QP'],x['group'],x['method'])==(d,str(q),label,'B1_initial'))
    text+=f"|{d}|{q}|{label}|{a['samples']}|"+'|'.join(f"{float(b[k])-float(a[k]):.9g}" for k in ('FID','D_mu','D_cov'))+'|\n'
 text+='\n## Leave-one-video-out sensitivity\n\nNames below identify the extremes of the matched-QP pooled gap after one sequence is removed. This is sensitivity evidence, not an independent significance test or a gain comparison at equal rate.\n\n|Dataset|QP|Smallest remaining gap|Excluded sequence|Largest remaining gap|Excluded sequence|\n|---|---:|---:|---|---:|---|\n'
 for d in ('ulong','hevc_b','mcljcv'):
  for q in (0,5,9):
   ds=[]
   for v in sources(d):
    a=next(x for x in lv if (x['dataset'],x['QP'],x['excluded_sequence'],x['method'])==(d,str(q),v['name'],'original'));b=next(x for x in lv if (x['dataset'],x['QP'],x['excluded_sequence'],x['method'])==(d,str(q),v['name'],'B1_initial'));ds.append((float(b['FID'])-float(a['FID']),v['name']))
   lo,hi=min(ds),max(ds);text+=f'|{d}|{q}|{lo[0]:.9g}|{lo[1]}|{hi[0]:.9g}|{hi[1]}|\n'
 text+='\n## Structure and provenance\n\nSpectra, GT-relative and direct original/B1 diagonal/off-diagonal differences, per-video mean shifts and within/between covariance closure are saved as separate numeric tables. Covariance additivity is not an additive attribution of covariance FID. The actual nine-checkpoint source chain is recorded, and18 available native-I intermediate points from V6.20/V6.21/V6.22 were matched. 首次出现阶段尚未定位：更早全部阶段的同输入、同 native-I 协议证据未建立，不能从版本号推断原因。\n\nBoth feature extractors received the identical floating RGB tensor, including directly decoded reconstruction (no reconstruction PNG quantization substitution). Same canonical GT sources and fixed feature weights were retained. No test data were used to select statistical layers, coefficients or budgets.\n'
 (ROOT/'FID_AUDIT_REPORT.md').write_text(text)
 dump(ROOT/'audits/task1_integrity.json',dict(status='PASS',features=18,explicit_covariance_points=18,GT_self_and_symmetry=True,gap_tables_complete=True,no_training=True,no_discriminator=True,first_onset='尚未定位首次出现阶段'))
 print('TASK1 FID AUDIT PASS',flush=True)
if __name__=='__main__':main()
