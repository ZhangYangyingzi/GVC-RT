"""Fact-only numerical tables and dataset-separated fixed-point figures."""
from util import *
import math,statistics,collections
GROUPS=['A_zero_base','B_flow_guided']
METRICS=['PSNR','LPIPS','DISTS','FloLPIPS','MC_RGB','MC_Sobel','edge_strength','GT_edge_strength','HF_energy','GT_HF_energy','HF_GT_error']
def rows():
 out=[]
 for p in sorted((ROOT/'evaluation/points').glob('*.json')):
  s=dict(load(p)['summary']);s['RWE_status']='RWE_NOT_VERIFIED';s['sigma']=s.get('sigma',0.);out.append(s)
 return out

def aggregate(rs):
 buckets=collections.defaultdict(list)
 for r in rs:buckets[tuple(r[k] for k in ['dataset','label','step','qp','beta','sigma'])].append(r)
 result=[]
 for key,vs in buckets.items():
  d=dict(zip(['dataset','label','step','qp','beta','sigma'],key));d['source_count']=len(vs);d['source_ids']='|'.join(sorted(v['id'] for v in vs));d['bpp']=statistics.mean(v['bpp'] for v in vs)
  for m in METRICS:d[m]=statistics.mean(v[m] for v in vs)
  d['RWE']=None;d['RWE_status']='RWE_NOT_VERIFIED';result.append(d)
 return result

def tables(rs):
 csvwrite(ROOT/'metrics/per_video_annotated.csv',rs);agg=aggregate(rs);csvwrite(ROOT/'metrics/aggregate_by_dataset.csv',agg);paired=[]
 for r in rs:
  if r['label']=='B300':continue
  b=next(x for x in rs if x['label']=='B300' and x['id']==r['id'] and x['qp']==r['qp']);d={k:r[k] for k in ['dataset','id','label','step','beta','qp','sigma']};d['reference']='B300'
  for k in METRICS:d['delta_'+k]=r[k]-b[k]
  paired.append(d)
  if r['label']=='B_flow_guided':
   a=next(x for x in rs if x['label']=='A_zero_base' and x['step']==r['step'] and x['beta']==r['beta'] and x['id']==r['id'] and x['qp']==r['qp']);d=dict(d,reference='A_zero_base')
   for k in METRICS:d['delta_'+k]=r[k]-a[k]
   paired.append(d)
 csvwrite(ROOT/'metrics/paired_differences.csv',paired)
 dst=ROOT/'metrics/all_per_frame.csv';header=None;count=0;allrows=[]
 for p in sorted((ROOT/'metrics/frames').glob('*.csv')):
  rr=list(csv.DictReader(p.open()));assert len(rr)==64;allrows.extend(rr);count+=len(rr)
 csvwrite(dst,allrows);assert count==9600
 repeats=collections.Counter(s['id'] for batch in plan()['joint'] for s in batch);csvwrite(ROOT/'metrics/training_coverage.csv',[dict(id=r['id'],source_group=r['source_group'],start=r['start'],crop=json.dumps(r['crop']),external_QP=r['external_qp'],segment_repeats_per_group=repeats[r['id']]) for r in training_rows()])
 traces={}
 for line in (ROOT/'logs/events.jsonl').open():
  event=json.loads(line)
  if event.get('event')!='UPDATE_COMMITTED':continue
  flat={k:v for k,v in event.items() if k not in ['gradient_groups','activation']};flat.update({'grad_'+k:v for k,v in event['gradient_groups'].items()});flat.update({'activation_'+k:v for k,v in event['activation'].items()});key=(event['group'],event['step']);assert key not in traces;traces[key]=flat
 assert len(traces)==600 and all({s for g,s in traces if g==group}==set(range(1,301)) for group in GROUPS)
 csvwrite(ROOT/'metrics/training_updates.csv',[traces[key] for key in sorted(traces)]);return agg,paired

def nondominated(rs,xkey,ykey):
 return [r for r in rs if not any(q[xkey]<=r[xkey] and q[ykey]<=r[ykey] and (q[xkey]<r[xkey] or q[ykey]<r[ykey]) for q in rs if q is not r)]
def plots(rs):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 figdir=ROOT/'figures';figdir.mkdir(exist_ok=True);made=[];fixed=load(ROOT/'evaluation/fixed_points.json')['fixed_points'];curves=[('Original',0),('B300',0)]+[(g,s) for s in fixed for g in GROUPS];colors={'Original':'#666666','B300':'#1f77b4','A_zero_base':'#df8800','B_flow_guided':'#bf2943'}
 for dataset in ['ulong','uvg']:
  data=[r for r in rs if r['dataset']==dataset and r['label']!='B300_blur' and r['beta']==1. and ((r['label'],r['step']) in curves)];expected={r['id'] for r in devrows() if r['domain']==dataset}
  for label,step in curves:
   for qp in [0,5,9]:assert {r['id'] for r in data if r['label']==label and r['step']==step and r['qp']==qp}==expected
  for metric in ['PSNR','LPIPS','DISTS','FloLPIPS']:
   fig,ax=plt.subplots(figsize=(7,4.8))
   for label,step in curves:
    pts=[]
    for qp in [0,5,9]:
     a=[r for r in data if r['label']==label and r['step']==step and r['qp']==qp];pts.append((statistics.mean(r['bpp'] for r in a),statistics.mean(r[metric] for r in a)))
    pts.sort();ax.plot([v[0] for v in pts],[v[1] for v in pts],marker='o',linestyle='--' if step==100 else '-',color=colors[label],label=label+(f' s{step}' if label in GROUPS else ''))
   ax.set(xlabel='Actual64-frame bitstream bpp',ylabel=metric,title=f'{dataset}: '+('four ULong development sources' if dataset=='ulong' else 'Bosphorus validation sequence only'));ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();p=figdir/f'{dataset}_RD_{metric}.png';fig.savefig(p,dpi=170);fig.savefig(p.with_suffix('.svg'));plt.close(fig);made.append(str(p))
 for row in devrows()[:4]:
  subset=[r for r in rs if r['id']==row['id'] and r['qp']==5 and (r['label'] in ['Original','B300','B300_blur'] or r['step'] in fixed) and r['beta']!=0]
  for metric in ['LPIPS','DISTS']:
   fig,ax=plt.subplots(figsize=(10,6));nd=nondominated(subset,metric,'MC_Sobel')
   for r in subset:
    label=r['label'];lab=label
    if label in GROUPS:lab+=f" s{r['step']} beta{r['beta']:g}"
    if label=='B300_blur':lab+=f" sigma{r['sigma']:g}"
    marker='^' if r['step']==100 else 'o';color=colors.get(label,'#55aa66');ax.scatter(r[metric],r['MC_Sobel'],s=45 if label not in GROUPS else 35+40*r['beta'],color=color,marker=marker,label=lab)
    if r in nd:ax.scatter(r[metric],r['MC_Sobel'],s=135,facecolors='none',edgecolors='black',linewidths=.9)
   ax.set(xlabel=metric+' (lower)',ylabel='GT-referenced MC-Sobel (lower)',title=row['id']+' / QP5 /64frames');ax.grid(alpha=.2);ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),fontsize=7);fig.text(.02,.015,'Black ring: nondominated in this displayed2D pair only. RWE_NOT_VERIFIED. Blur is diagnostic, not a trained candidate.',fontsize=8);fig.tight_layout(rect=(0,.04,1,1));p=figdir/f'{row["id"]}_{metric}_MC_Sobel_tradeoff.png';fig.savefig(p,dpi=160);plt.close(fig);made.append(str(p))
 for metric in ['PSNR','LPIPS','DISTS','FloLPIPS','MC_Sobel']:
  fig,ax=plt.subplots(figsize=(6,4))
  for group in GROUPS:
   pts=[]
   for step in [0,50,100,200,300]:pts.append(statistics.mean(r[metric] for r in rs if r['dataset']=='ulong' and r['label']==group and r['step']==step and r['qp']==5 and r['beta']==1))
   ax.plot([0,50,100,200,300],pts,'o-',color=colors[group],label=group)
  b=statistics.mean(r[metric] for r in rs if r['dataset']=='ulong' and r['label']=='B300' and r['qp']==5);ax.axhline(b,color=colors['B300'],linestyle='--',label='B300');ax.set(xlabel='Committed optimizer updates',ylabel=metric,title='Fixed four-source ULong/QP5/beta1/64frames');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();p=figdir/f'ULong_trajectory_{metric}.png';fig.savefig(p,dpi=170);plt.close(fig);made.append(str(p))
 return made

def write_report(rs,agg,paired,figs):
 p=load(ROOT/'PROTOCOL.json');pre=load(ROOT/'audits/preflight.json');completions={g:load(ROOT/'checkpoints'/g/'completion.json') for g in GROUPS};perf=load(ROOT/'audits/performance.json');fixed=load(ROOT/'evaluation/fixed_points.json')['fixed_points'];fmt=lambda x:f'{x:.9g}' if isinstance(x,(int,float)) else str(x)
 lines=['# Flow-guided deformable A/B：事实执行报告','', '状态：本轮训练、150个64帧评测点、强度扫描、图表、媒体和本地完整性检查已完成。仅执行预定实验，不作方法有效性、统计显著性或收敛结论。GitHub发布状态另见 publication_status.json。','', '## 基座与训练边界','', f"起点为PREV已提交HEAD `{p['base_commit']}`。完整B300恢复链：Original→pilot→B1500 TCG→A2000 LoRA(s=0.5)→最后覆盖B300增量。合成参数SHA256：`{pre['base_composite_parameter_SHA256']}`。codec/entropy/Bridge/de-tokenizer/现有TCG/LoRA权重全部冻结，输入梯度和4帧段内TCG跨帧梯度保留。没有从旧T100/T300初始化，也没有重训旧S/U/T。",'', 'A_zero_base基础位移为0，局部offset/mask/DCN/q/fusion仍可学习；B_flow_guided基础位移来自冻结RAFT在独立Original解码参考RGB上的预测。A/B证据张量相同，A仍包含运动证据，不是完全无对齐或无运动信息的组。每组新增258,372个参数，同seed20261113与相同初始化、样本顺序、crop、QP及四帧短段。','', '## 结构和状态','', 'Bridge18通道输出后接单deform-group的3×3 modulated DCN，18输入/18输出。offset输入58通道，Conv3×3 58→64→64→27；18个offset采用1.0*tanh，单位latent网格；其余9个logit经sigmoid。offset/mask末层全零；DCN中心核2*I、其余核与bias0，使初始mask0.5时中心内容恒等。这是本实验初始化，不是论文标准初始化声明。','', '最终DCN只采样原始未warp历史latent；粗warp只供offset预测。每点独立几何越界屏蔽，全局v采用中心采样有效性，全部无效时填回当前latent。q接口59→32→1，残差融合54→64→两个残差块→18零输出；l_prime=l+beta*q*delta。训练beta1，输出同时进入原conv_in和AdaGN。buffer只保存上一帧原始Bridge latent及元数据，修正latent与候选RGB不进入codec DPB。各候选独立reset与TCG滚动。','', '## 数据、缓存和目标','', '沿用32个隔离ULong训练来源组、128个不重叠16帧窗口、固定256×256原始RGB裁剪、QP0/5/9共384个实例；先RGB裁剪再冻结codec/Bridge。两组沿用前轮前300个joint计划，每更新4个4帧段等权，microbatch1；按各自当前权重从窗口I帧reset因果预热。段内保留跨帧梯度，段外detach。来源、完整计划与重复次数见data_manifest.json、training_plan.json及metrics/training_coverage.csv。600条正式更新的loss、分组梯度和激活记录见metrics/training_updates.csv；零初始化及后续固定样本诊断见checkpoints/各组/diagnostic_*.json。','', '旧训练缓存original仅有I帧。本轮补齐所有P帧FP32 Original参考RGB，并对每种QP抽样逐帧与真实bitstream decompress比对，结果严格相等，见audits/original_reference_replay_q*.json。参考流来自已审计RAFT权重/transform、双向各12次迭代，只读取解码参考，不读取GT、GT-flow、候选输出或未来帧。photometric与FB残差在首8个训练实例上固定RMS归一化，latent/QP固定统计沿用合法训练统计。','', '磁盘压力下使用可逆字节重排压缩；后续流/证据保存为新分支实际消费的latent网格，向量按真实宽高比缩单位，Original参考RGB仍完整FP32保存。旧RGB网格条目保持只读兼容，已核对同尺寸二次resize数值完全相同，详见audits/cache_storage_revision.json。','', 'L_sp=LPIPS_Alex(normalize=True)+0.5*DISTS+0.1383589248954108*RGB_L1；L_temp为GT固定流/有效区下的MC-RGB变化残差+0.5*有符号归一化/8 Sobel变化残差。L_total=L_sp+4.63349060748*L_temp。明确不含旧0.1*L_align，没有新增GAN、滤波训练目标、teacher-latent或HF旁路。AdamW lr5e-5, betas0.9/0.999, eps1e-8, weight_decay1e-4, global_clip1.0，无调度。重建、分支、loss为FP32参考路径；codec沿用FP16，Original P参考FP32、I沿用冻结image codec精度。','', '## 实际步数与停止规则','', '|组|已提交更新|停止原因|更新计算秒数|','|---|---:|---|---:|']
 for g,r in completions.items():lines.append(f"|{g}|{r['committed_updates']}|{r['stop_reason']}|{r['training_seconds']:.6f}|")
 lines+=['', '保存0/1/10/50/100/200/300及latest，包含optimizer/RNG/计划游标/协议及实现哈希。checkpoint0/50/100/200/300均等待自己的64帧评测，实际领先评测0个间隔。按首评测前冻结的规则，仅连续两个检查点被同一早期点在PSNR↑、LPIPS/DISTS/FloLPIPS/MC-Sobel↓五指标Pareto支配且至少一项严格更好才停，容差为1e-6相对。所有检查点机械判定均为继续，两组最终达到300上限，没有自动续训。','', '## 评测范围与原值','', '150个视频×模型×step×QP×beta/blur点，每点64连续原分辨率帧，总9600条逐帧记录。4个ULong合法开发源，UVG仅Bosphorus合法validation单序列，Bosphorus原生120fps；2个可视案例ULong原生30fps。全序列从相同I帧/GOP起点独立reset/rollout，无重复帧补齐。所有候选共享对应64帧真实码流，未混用旧16帧bpp/指标，也未使用最终135点测试。','', '固定多QP对照点为'+str(fixed)+'；没有按质量选择其他step。两个固定案例分别beta0/0.25/0.5/1独立滚动，beta0经64帧与B300精确相等后显示去重。B300轻微Gaussian sigma0.25/0.5/0.75仅诊断，核半径ceil(3sigma)，离散归一化核，replicate边界，不进入训练。','', 'VideoGigaGAN RWE：已核对[原文Eq4与Sec4.1](https://arxiv.org/html/2404.12388v2#S4.SS1)、[官方项目](https://videogigagan.github.io/)及[补充材料](https://videogigagan.github.io/assets/supp.pdf)。其核心为用候选估计的forward RAFT运动warp GT，再计算有效像素上的平方RGB向量误差。官方材料未给出可复核的精确non-occlusion mask实现/阈值与通道归约代码，本轮按任务允许的回退标记RWE_NOT_VERIFIED，不生成公式变体冒充原RWE或填0。停止规则首评测前已固定为其余5个指标；取舍图用已核实MC-Sobel轴，不重命名为RWE。','', '|ULong四源等权，QP5，beta1|step|PSNR|LPIPS|DISTS|FloLPIPS|MC-RGB|MC-Sobel|边缘强度|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in sorted([r for r in agg if r['dataset']=='ulong' and r['qp']==5 and r['beta']==1 and r['label']!='B300_blur'],key=lambda r:(r['label'],r['step'])):lines.append('|'+ '|'.join(fmt(r[k]) for k in ['label','step','PSNR','LPIPS','DISTS','FloLPIPS','MC_RGB','MC_Sobel','edge_strength'])+'|')
 lines+=['', '最终复核发现3个新GT监督缓存源的有效掩码在1088高补齐域计算后，只裁回1080而未与原生采样域相交。已对相关72个评测点修正固定掩码并仅重算MC-RGB/MC-Sobel；训练、模型、参考流、候选输出和其他指标均未改变。原值保留在audits/native_validity_correction/before，逐帧差异和输出MSE核对见其points目录。四项未变的停止指标已排除可能的连续支配，修正后重新判定仍全部继续；没有基于结果更换结构、loss或权重。', '', '逐帧/逐视频×QP×step×beta原始数值、逐数据集等权汇总与B300/A配对差值分别见metrics/all_per_frame.csv、per_video_annotated.csv、aggregate_by_dataset.csv、paired_differences.csv。原始per_video_metrics.csv中的RWE_status因通用数值归约为空，annotated表只将该显示字段明确为RWE_NOT_VERIFIED，未改任何数值。','', '取舍图分别逐视频绘制LPIPS/DISTS—MC-Sobel，明确step、beta和模糊参照；黑圈仅表示当前显示二维指标的非支配点，不代表最佳模型。R-D分别绘制ULong和UVG单源，包含独立Original、冻结B300、A/B100与A/B300，不跨数据集合并、不外推3个QP点。HEVC/MCL未虚构结果。','', '## 性能与资源','', '|组|codec|独立参考RGB|双向RAFT及证据|DCN/offset|fusion/q|完整最终decoder|非缓存端到端 ms/P帧|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for g,r in perf['results'].items():
  a=r['mean_ms'];lines.append('|'+g+'|'+'|'.join(fmt(a[k]) for k in ['codec','independent_reference_RGB','bidirectional_RAFT_and_evidence','DCN_and_offset','fusion','final_decoder','end_to_end'])+'|')
 lines+=['', '实际RAM码流非缓存测量完整P1..P63，包含独立Original参考重建、双向RAFT、Bridge、DCN、融合及完整最终decoder，哈希传输也包含在总计时，GT/loss/离线质量指标不在该计时。分支命中缓存速度不是部署速度；零新增码流不代表实时。完整样本计时与累计峰值在audits/performance.json；逐组件峰值/驻留与增量显存见audits/performance_components.json。主性能报告峰值为进程内累计，未伪装成隔离组件峰值。','', '只使用物理GPU0–3，未终止其他任务。GPU2按计划生产缓存，A/B按自身下个样本等待；GPU0/1在组等待评测时运行相应评测，GPU3处理队列。早期CUDA OOM进入等待/重试，所有最终评测点均从reset完成并PASS，没有降分辨率、删帧、改loss或填缺失指标。缓存格式调整只涉及本任务自己的生产器重启；训练源/计划/数值目标未变。CUDA grid_sample与DCN反向使用deterministic warn_only，固定seed不等于bitwise确定性声明。','', '## 媒体、完整性与限制','', '30个原分辨率FFV1与12个5栏原生ROI FFV1，均连续64帧，附MP4播放副本；每栏ROI256×256，标题独立48像素，不遮挡或缩放内容。模型步数、beta、QP、真实码率、帧范围、FPS与SHA256见VIDEO_INDEX.md及audits/video_delivery.json。新增beta1视频在1/32/63固定帧的浮点MSE与对应已完成评测逐值一致。','', 'B300加载/冻结、step0/beta0连续64帧等价、DCN坐标/梯度/初始化、独立TCG状态、codec bits/features、全部预期点/帧数/重复行/非有限值、checkpoint与媒体完整性在final_integrity.json复核。未增加第二seed、第三训练组或下一轮；未让执行端替用户做科学结论。限制包括单seed、32训练来源、固定开发来源、UVG单源、仅3个QP、RWE未完成精确核验，以及crop/full-frame分布差异。','', '最终指标修正期间，旧ULong源目录变为不可访问，相关任务立即停止。随后从既有只读ULong ZIP定向恢复相同4个开发视频，逐文件SHA256与冻结协议全部相同，写入本任务cache/recovered_sources并记录source_path_migrations.json；未替换样本或改写旧路径。已完成修正点幂等保留后恢复任务。', '', '所有新增代码、缓存、日志和临时文件均在本独立worktree目录。大模型/数据/缓存/码流/大型视频不进普通Git，large_artifacts.json给出本地绝对路径、大小、SHA256和复现入口；发布范围与远端核验见publication_status.json。','']
 (ROOT/'REPORT.md').write_text('\n'.join(lines));return lines
