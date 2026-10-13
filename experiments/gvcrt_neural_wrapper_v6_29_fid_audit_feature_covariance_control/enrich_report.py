from io21 import *
import statistics

def main():
 table=read(ROOT/'results/four_metric_comparisons.csv');lines=['\n## Required questions\n','\n### Historical FID implementation\n\nThe18 same-floating-RGB feature checks,18 explicit sample-covariance/eigendecomposition comparisons, complete GT self-distance and symmetry checks, and analytic Gaussian parameter checks passed the frozen tolerances. No unresolved implementation mismatch was found in these tested points. This is an implementation audit, not a claim that finite-sample FID is an unbiased estimator.\n']
 gap=read(ROOT/'results/FID_audit/fid_decomposition_raw.csv');groups=read(ROOT/'results/FID_audit/frame_group_statistics.csv');lv=read(ROOT/'results/FID_audit/leave_one_video_out.csv')
 lines.append('\n### Original to B1: matched-QP statistical gap\n\nThis table diagnoses identical samples at the same external QP and is not an equal-rate gain table.\n\n|Dataset|QP|Delta FID|Delta mean term|Delta covariance term|Covariance trace change|\n|---|---:|---:|---:|---:|---:|\n')
 for d in ('ulong','hevc_b','mcljcv'):
  for q in (0,5,9):
   a=next(x for x in gap if x['dataset']==d and x['QP']==str(q) and x['method']=='original');b=next(x for x in gap if x['dataset']==d and x['QP']==str(q) and x['method']=='B1_initial')
   vals=[float(b[k])-float(a[k]) for k in ('FID','D_mu','D_cov','covariance_trace')];lines.append(f'|{d}|{q}|'+ '|'.join(f'{x:.9g}' for x in vals)+'|\n')
 lines.append('\nVideo-level mean/covariance offsets and diagonal/off-diagonal squared errors are in per_video_statistics.csv; spectra are in covariance_spectrum.csv. Within/between covariance matrices close numerically; their additivity does not imply an additive split of covariance FID distance. I0 reconstruction tensor hashes are compared in the independent feature audit. Frame-group differences only compare equal sample identities within each group.\n')
 lines.append('\n### First historical appearance\n\nThe actual nine-checkpoint parent chain is recorded. Compatible available V6.20/V6.21/V6.22 points were evaluated numerically from existing features (intermediate_stages.csv). 首次出现阶段尚未定位；未建立更早全部阶段的相同 native-I 与输入协议证据，不能从版本号推断原因。\n')
 lines.append('\n### Covariance target versus matched continuation\n\nThe following are mechanical four-metric decisions at fixed1000 updates. “Lower in all four” is a numerical comparison and is not a statistical significance test.\n\n|Dataset|Reference|LPIPS delta %|DISTS delta %|FloLPIPS delta %|FID delta %|All four non-increasing|\n|---|---|---:|---:|---:|---:|---|\n')
 for d in (*DATASETS,'mcljcv_exclude_overlap29'):
  for ref in ('A_continue_1000','B1_initial','original'):
   rs=[r for r in table if r['dataset']==d and r['method']=='B_covstats_1000' and r['reference']==ref];assert len(rs)==4
   vals=[];ok=True
   for k in ('LPIPS','DISTS','FloLPIPS','FID'):
    r=next(r for r in rs if r['metric']==k);val=float(r['relative_change_percent']) if r['status']=='available' and r.get('relative_change_percent') else None;vals.append('unavailable' if val is None else f'{val:.8g}');ok=ok and val is not None and val<=0
   lines.append('|'+d+'|'+ref+'|'+'|'.join(vals)+'|'+str(ok)+'|\n')
 lines.append('\nThe main loss comparison is B versus A; A versus B1 isolates the extra training budget, while B versus B1 reports net change. Positive metric deltas are costs. Actual same-QP bitrate changes are recorded separately in same_QP_rate_quality_differences.csv; same-QP quality changes are not equal-rate gains.\n')
 lines.append('\n### Rate bands and leave-one-video-out limits\n\nMain low/mid/high bands use equal-width ln(bpp) thirds of each four-model measured common interval and PCHIP without extrapolation. New leave-one-video-out curves are a three-QP sensitivity diagnostic, clearly separated from the main ten-QP benchmark. Their variation can identify concentration in specific sequences but does not create independent repeats or a significance test. No claim of significant small gains is made.\n')
 lines.append('\n### Feature covariance diagnosis\n\nD_mu/D_cov diagnostic interpolation applies the same linear ln(bpp) weights to mean vectors and covariance matrices, then computes the resulting distance. It is not main PCHIP FID and is not required to equal that curve. The per-frame DISTS spatial covariance target is distinct from pooled Inception covariance. A decrease in its training loss is not evidence of an FID gain.\n')
 lines.append('\n### Remaining scope\n\nOne fixed initialization, sampling plan, coefficient and1000-update budget were used. MCL30 and exclusion29 are separately matched pools; Kimono1/videoSRC17 is cross-test duplication, not demonstrated training leakage. Unknown external pretraining exposure was not ruled out. Fixed64-frame inference and one run cannot prove universal generalization. No alternative statistical target or coefficient was selected from HEVC-B/MCL-JCV.\n')
 lines.append('\n## Absolute values on the primary common-rate grid\n\nAll values below use the main ten-QP,100-point PCHIP protocol, not same-QP measurements or covariance diagnostic interpolation. Perceptual metrics use equal video weights; FID uses each fixed pooled feature set.\n\n|Dataset|Method|Reference|Metric|Method mean|Reference mean|Delta|Relative %|\n|---|---|---|---|---:|---:|---:|---:|\n')
 def number(x):return 'unavailable' if x in ('',None) else f'{float(x):.9g}'
 for r in table:
  lines.append('|'+ '|'.join([r['dataset'],r['method'],r['reference'],r['metric']]+[number(r.get(k)) for k in ('method_mean','reference_mean','delta','relative_change_percent')])+'|\n')
 lines.append('\n## Low/mid/high: added covariance loss versus matched continuation\n\nEach band is one equal-width ln(bpp) third of the measured four-model common interval. The complete four-comparison tables and exact intervals are in results/equal_rate_bands_summary.csv and equal_rate_bands_per_sequence.csv. Percent=100*(method mean−reference mean)/reference mean; not BD-rate.\n\n|Dataset|Band|Metric|B mean|A mean|Delta|Relative %|\n|---|---|---|---:|---:|---:|---:|\n')
 for r in read(ROOT/'results/equal_rate_bands_summary.csv'):
  if r['method']=='B_covstats_1000' and r['reference']=='A_continue_1000':
   lines.append('|'+ '|'.join([r['dataset'],r['band'],r['metric']]+[number(r.get(k)) for k in ('method_mean','reference_mean','delta','relative_change_percent')])+'|\n')
 lines.append('\n## Leave-one-video-out numerical sensitivity\n\nThese are three-QP diagnostic curves on each remaining pool, not independent repetitions, confidence intervals, or the main ten-QP effect. Excluded-video names identify sensitivity extremes, not additive video contributions.\n\n|Dataset|Available exclusions|Negative B−A FID changes|Minimum %|Excluded at minimum|Maximum %|Excluded at maximum|\n|---|---:|---:|---:|---|---:|---|\n')
 for d in (*DATASETS,'mcljcv_exclude_overlap29'):
  folder=ROOT/'results/new_leave_one_video_out'/d;p=folder/'equal_rate_diagnostic.csv'
  if not p.exists():lines.append(f'|{d}|0|unavailable|unavailable|single-sequence empty pool|unavailable|single-sequence empty pool|\n');continue
  rs=[r for r in read(p) if r['method']=='B_covstats_1000' and r['reference']=='A_continue_1000' and r['status']=='available' and r.get('relative_change_percent')]
  if not rs:lines.append(f'|{d}|0|unavailable|unavailable|unavailable|unavailable|unavailable|\n');continue
  low=min(rs,key=lambda r:float(r['relative_change_percent']));high=max(rs,key=lambda r:float(r['relative_change_percent']));negative=sum(float(r['relative_change_percent'])<0 for r in rs)
  lines.append(f"|{d}|{len(rs)}|{negative}|{number(low['relative_change_percent'])}|{low['excluded_sequence']}|{number(high['relative_change_percent'])}|{high['excluded_sequence']}|\n")
 (ROOT/'REPORT.md').write_text((ROOT/'REPORT.md').read_text()+''.join(lines));(ROOT/'RESULTS_SUMMARY.md').write_text((ROOT/'REPORT.md').read_text())
if __name__=='__main__':main()
