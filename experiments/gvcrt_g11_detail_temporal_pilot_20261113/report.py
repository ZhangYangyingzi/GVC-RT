"""Dataset-separated factual tables, QP5 scatter plots and training traces."""
from util import *
import statistics
METRICS=['PSNR','LPIPS','DISTS','FloLPIPS','MC_RGB','MC_Sobel','highpass_signed_error_MSE','signed_Sobel_L1','highpass_energy','GT_highpass_energy','highpass_cosine','valid_fraction']

def tables():
 points=[load(p)['metrics'] for p in sorted((ROOT/'metrics/points').glob('*.json'))];csv_write(ROOT/'metrics/per_source.csv',points);keys=['dataset','label','step'];buckets=collections.defaultdict(list)
 for r in points:buckets[tuple(r[k] for k in keys)].append(r)
 agg=[]
 for k,rs in sorted(buckets.items()):
  d=dict(zip(keys,k));d.update(sources=len(rs),source_ids='|'.join(sorted(r['sequence'] for r in rs)),bpp=statistics.mean(r['bpp'] for r in rs),QP=5)
  for m in METRICS:
   vals=[r[m] for r in rs if r.get(m) is not None];d[m]=statistics.mean(vals) if vals else None
  agg.append(d)
 csv_write(ROOT/'metrics/per_dataset.csv',agg);paired=[]
 for r in points:
  for baseline in ['C_ref','B300']:
   if r['label']==baseline:continue
   b=next(x for x in points if x['label']==baseline and x['sequence']==r['sequence']);d={k:r[k] for k in ['dataset','sequence','label','step','bpp']};d['reference']=baseline
   for m in METRICS:
    if r.get(m) is not None and b.get(m) is not None:d['delta_'+m]=r[m]-b[m];d['relative_'+m]=r[m]/max(abs(b[m]),1e-12)-1
   paired.append(d)
 csv_write(ROOT/'metrics/paired_differences.csv',paired)
 import csv
 for directory,name in [('frames','all_per_frame.csv'),('transitions','all_transitions.csv')]:
  rows=[]
  for p in sorted((ROOT/'metrics'/directory).glob('*.csv')):rows.extend(csv.DictReader(p.open()))
  csv_write(ROOT/'metrics'/name,rows)
 updates=[]
 for line in (ROOT/'logs/train.jsonl').open():
  r=json.loads(line);d={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
  for k in ['errors','reference','gradient_groups','activation']:
   for n,v in r[k].items():d[k+'_'+n]=v
  d.update(mu=json.dumps(r['mu']),g=json.dumps(r['g']),LR=json.dumps(r['LR']),ROI=json.dumps(r['ROI']),valid_counts=json.dumps(r['valid_counts']));updates.append(d)
 csv_write(ROOT/'metrics/training_updates.csv',updates);assert len(updates)==load(ROOT/'completion.json')['committed_updates']
 import gzip
 (ROOT/'metrics/training_log.jsonl.gz').write_bytes(gzip.compress((ROOT/'logs/train.jsonl').read_bytes(),mtime=0))
 used=load(ROOT/'sample_schedule.json')['updates'][:len(updates)];coverage=[]
 for row in videos('train'):
  clips=[c for c in used if c['video']==row['sequence']];coverage.append(dict(sequence=row['sequence'],source_group=row['source_group'],formal_clips=len(clips),formal_supervised_frame_slots=4*len(clips),unique_temporal_windows=len({c['start'] for c in clips}),unique_spatial_ROIs=len({(c['x'],c['y']) for c in clips}),unique_space_time_clips=len({(c['start'],c['x'],c['y']) for c in clips})))
 csv_write(ROOT/'metrics/training_coverage.csv',coverage);dump(ROOT/'audits/actual_training_coverage.json',dict(formal_source_groups=len({c['source_group'] for c in used}),formal_updates=len(used),formal_supervised_frame_slots=4*len(used),unique_temporal_windows=len({(c['video'],c['start']) for c in used}),unique_spatial_ROIs=len({(c['video'],c['x'],c['y']) for c in used}),unique_space_time_clips=len({(c['video'],c['start'],c['x'],c['y']) for c in used}),calibration_and_smoke_updates_excluded=True))
 return points,agg,paired

def plots(points,agg):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 out=[];colors={'Original':'#666666','B300':'#1f77b4','C_ref':'#228833','G11':'#b52344'};markers={0:'o',50:'s',100:'^',200:'D',300:'X'}
 for dataset in sorted({r['dataset'] for r in agg}):
  for metric in ['PSNR','LPIPS','DISTS','FloLPIPS','MC_Sobel','highpass_signed_error_MSE']:
   fig,ax=plt.subplots(figsize=(9,5))
   for r in [a for a in agg if a['dataset']==dataset]:ax.scatter(r['bpp'],r[metric],color=colors[r['label']],marker=markers[r['step']],s=70,label=r['label']+(f" s{r['step']}" if r['label']=='G11' else ''))
   ax.set(xlabel='Actual64-frame bitstream bpp (QP5 only)',ylabel=metric,title=dataset+' validation6 / scored49..63 / single-QP scatter');ax.grid(alpha=.2);ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=8);fig.tight_layout();p=ROOT/'figures'/f'{dataset}_QP5_scatter_{metric}.png';fig.savefig(p,dpi=160);plt.close(fig);out.append(str(p))
   fig,ax=plt.subplots(figsize=(7,4.5));rs=sorted([r for r in agg if r['dataset']==dataset and r['label']=='G11'],key=lambda r:r['step']);ax.plot([r['step'] for r in rs],[r[metric] for r in rs],'o-',color=colors['G11'],label='G11')
   for label in ['Original','B300','C_ref']:
    b=next(r for r in agg if r['dataset']==dataset and r['label']==label);ax.axhline(b[metric],color=colors[label],linestyle='--',label=label)
   ax.set(xlabel='Committed optimizer updates',ylabel=metric,title=dataset+' validation / QP5 /15frame short segment');ax.grid(alpha=.2);ax.legend();fig.tight_layout();p=ROOT/'figures'/f'{dataset}_trajectory_{metric}.png';fig.savefig(p,dpi=160);plt.close(fig);out.append(str(p))
 return out

def write(points,agg,figures):
 done=load(ROOT/'completion.json');pre=load(ROOT/'audits/preflight.json');cp=load(ROOT/'weight_manifest.json');steps=sorted({r['step'] for r in points if r['label']=='G11'});mean=load(ROOT/'audits/budget_calibration.json');lam=load(ROOT/'audits/lambda_calibration.json') if (ROOT/'audits/lambda_calibration.json').exists() else None
 lines=['# G11单组先导试验：事实记录','',f"本轮已完成{done['committed_updates']}个正式optimizer updates，停止原因`{done['stop_reason']}`。只执行G11、seed20261112，未启动G00/G10/G01、第二种子、多QP或下一轮。科研解释由用户与分析端处理。",'', '## 固定参考与参数边界','', 'C_ref加载链为Original→pilot→B1500 TCG→A2000 LoRA(s=0.5)→旧B300最后overlay→D12-S@900 seed20261112。旧B300不是flow-guided B组step300。D12-S完整S前向、condition=S及共享metadata/validity保持原样，S仍有codec/TCG历史。', '', f"D12-S增量文件SHA256：`{cp['D12_S_900']['sha256']}`；完整C_ref(I+P+S、按命名张量顺序)SHA256：`{pre['C_ref_complete_sha256']}`。权威预期来自父checkpoint_integrity/weight_manifest，实际文件逐项核验。本轮重算C_ref六源的LPIPS/DISTS/RGB_MSE/signedHF_MSE/SobelL1与父同协议结果逐项差值均为0，见audits/C_ref_parent_replay.json。父已审计P-model状态SHA与本次实测一致：`{pre['C_ref_base_state_hashes']['pm']}`。这里状态哈希与其他文档仅参数哈希口径不同，不混作同一数值。",'', '只训练新增对齐/投影/融合/输出1,412,251参数，及stage3.blocks[0]原生卷积988,800参数，共2,401,051。精确12个接收块张量名和模块结构见audits/runtime_inspection.json；已有LoRA因子、TCG、S、GN/AdaGN、Bridge、codec、其他stage3/stage4与head均冻结。冻结hash排除有意训练的接收块；suffix保留输入梯度。','', '## 实现、数据和精度','', 'G11在tcg_mid后先经过完整S，再以h_star=h_s+v*Delta_h进入stage3。读取冻结S.E的128通道当前/原始过去内容；128通道3×3 modulated DCN从raw过去一次采样。粗warp仅给388→128→27 predictor提供线索；残余dy/dx offset=tanh限±1个H/8格，mask初始0.5，DCN中心2I、其他0、无bias。P_h512→128；514→128+SiLU、两个父定义残差块、128→512零输出头无bias，无额外零scalar。','', 'v是父协议FB/native-center及实际采样点几何有效性代理，不是真实遮挡或可靠度标签。采用像素中心8x+3.5、align_corners=False，向量单位除8。推理只读独立Original当前/过去解码RGB上RAFT的父metadata；GT-flow仅进入监督/评价，不进候选forward。raw c_prev不包含G11输出；每片段候选和参考独立2帧suffix预热、独立TCG_out状态，四帧内保留前后输出梯度，段间/reset断开。stage3/4 checkpoint只包纯计算块。','', '严格使用父12个train与6个validation来源，均ULong。final_test六源没有加入本地数据清单或被本轮评测。固定300步schedule每12步seeded shuffle覆盖12源各一次，start49..60、连续4帧全部49..63、256×256 ROI在native有效范围的8格上，同clip同ROI；预热start-2/start-1。首4个不同训练源对应片段用于全部校准。来源组隔离和全部ROI/schedule/SHA已冻结。','', '完整padded后缀执行GroupNorm，只有损失裁ROI；高通、Sobel与warp在native整图计算再裁ROI。分支/decoder/loss/grad为FP32，TF32关闭，codec沿用FP16。对照Original用独立原版代码和原始I/P，FP32重建与本轮评分相配；父对齐metadata仍保持其已审计Original FP16参考，不把二者混成同一缓存。缓存仅复用父码流/codec feature/GT/独立Original，另缓存本任务冻结prefix/S/E及固定GT-RAFT监督。所有新写入留在独立worktree。','', '## 固定目标和正式训练','', 'L_sp=LPIPS_Alex(normalize=True)+0.5*DISTS(require_grad=True)+0.1383589248954108*RGB_L1。L_temp=GT参考固定运动补偿RGB变化误差+0.5*有符号/8 Sobel变化误差；只纳有效同reset段相邻对，双端梯度保留。', '', '预算E_j为LPIPS、未加权DISTS、对GT的signed5×5 binomial highpass MSE。m_j为首4个训练校准clip的C_ref平均误差；g_j=(E_j(candidate)-1.01*E_j(C_ref))/max(m_j,1e-8)。L_budget=sum(mu_j*relu(g_j))；mu初始0.1，正式更新后mu=clip(mu+0.01*detached(g),0,1)，validation不更新mu。C_ref只提供误差基准，不是RGB蒸馏目标；这是软预算，没有逐样本或泛化保证。', '', f"训练固定m_j：`{mean['means']}`。",'', 'step1..50仅L_sp+L_budget；step50已提交后，在固定4训练clip上对接收块原生参数计算L_sp/L_temp梯度范数、比值和余弦，无权重更新。lambda_star=0.2*median(norm_sp/(norm_temp+1e-12))，51..100线性升至lambda_star，101..300固定。']
 if lam:lines += ['',f"实际lambda_star=`{lam['lambda_star']}`；逐片段范数/比值/余弦见audits/lambda_calibration.json。"]
 lines += ['', 'AdamW新分支peak_lr5e-5、接收块5e-6，前20步线性warmup、其后恒定，betas0.9/0.999、eps1e-8、weight_decay1e-4、全局clip1。每步一个4帧clip。训练完整loss/误差/乘子/LR/梯度/激活/ROI见metrics/training_updates.csv及本地logs/train.jsonl。checkpoint含branch、native_receiver、optimizer、RNG、mu、lambda、schedule游标和协议SHA，未遗漏接收块。', '', f"纯训练更新计算累计{done['training_seconds']:.6f}秒，缓存和验证等待不计入该数。固定观察点实际完成{steps}，每点六源验证全部结束后才推进。预设预算停止规则与逐源证据见evaluation/decision_*.json，未因单次指标波动或PSNR下降自行改方向。",'', '## 验证与原始指标','', f"共{len(points)}个模型×checkpoint×validation-source点，每点47/48独立suffix预热、49..63计分15帧。跨reset时间对被排除，最大14相邻对。所有候选共享对应64帧真实码流、实际QP、feature、native/padding约定。bytes/bpp口径是完整64帧真实码流，未拿15帧长度重新冒充码流。",'', '|dataset|model|step|sources|bpp|PSNR|LPIPS|DISTS|FloLPIPS|MC_RGB|MC_Sobel|signed HF MSE|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in agg:lines.append('|'+ '|'.join(str(r[k]) if k in ['dataset','label','step','sources'] else f"{r[k]:.9g}" if r.get(k) is not None else 'N/A' for k in ['dataset','label','step','sources','bpp','PSNR','LPIPS','DISTS','FloLPIPS','MC_RGB','MC_Sobel','highpass_signed_error_MSE'])+'|')
 lines += ['', '逐帧、逐时间对、逐源、逐数据集等权汇总与分别相对C_ref/B300差值见metrics/。整体均值没有替代逐数据集结果。本轮合法来源只有ULong，其他数据集未评估；QP5图为bpp—质量散点及更新轨迹，不是完整R-D曲线。保留Original，未做码率外推或选择最终测试权重。','', '## 预检、性能、媒体及限制','', f"step0对完整C_ref的RGB max_abs_diff={pre['step0_RGB_max_abs']}，阈值1e-6。DCN恒等与正负/分数平移、越界回退、独立参考无可变别名、四帧状态梯度、双端时间梯度、冻结白名单检查均通过。smoke3步不计正式预算，完成后已恢复初始branch/receiver/optimizer/RNG及状态。见audits/preflight.json。",'', '首次正式更新在backward阶段被Original初始化恢复的strict deterministic模式阻断，未提交optimizer更新。已保存当时step0/initial与原始凭据，修复setup_torch在Original初始化后恢复协议声明的warn_only，新增该路径的反向预检，并从同一初始seed/权重/schedule重做预检与step0。结构、loss、预算、数据和阈值未改变，详见audits/pre_first_update_determinism_fix。', '', '新增分支/接收块、codec/Bridge冻结审计、相同bits/bpp/feature、非有限值与点数完整性见final_integrity.json。grid_sample/DCN反向采用deterministic warn_only，固定seed不承诺CUDA训练bitwise复现。','', '原分辨率FFV1为full-range RGB8，float→8bit使用round/clamp，逐字节无损回读；各视频统一source frame49..63及各自真实FPS。GT/Original/B300/C_ref/G11明确标识，G11保存step100及终点（达到时），附MP4播放副本。15帧短段不代表长时稳定性；循环预览不作长序列证据。完整媒体路径/哈希/身份见VIDEO_INDEX.md与audits/video_delivery.json。','', '独立Original参考+双向RAFT及G11分支/后缀耗时范围与峰值见audits/performance.json，分别标明实际父FP16参考路径（与缓存RGB核对）和额外FP32重建计时；输入codec feature来自已审计缓存，此计时不冒充完整entropy端到端解码速度，不宣称实时。','', 'step0验证期间GPU2被其他进程占用导致临时OOM，未提交正式更新。已保留凭据，在用户授权GPU0–3范围按28000MiB可用显存排队/迁移，只有一个trainer，未终止其他进程、缩小后缀或更换数据/loss。实际GPU、PID、恢复次数和日志见completion.device、logs/runner_events.jsonl、audits/resource_recoveries。验证阶段恢复RNG，0步运行元数据迁移时已核对branch/native/optimizer未改变。', '', '单seed、单训练候选、12训练源、6开发源、QP5和短验证段是本轮范围。没有把MC_Sobel百分比改写成人眼闪烁百分比，也没有把高频能量增加称为真实细节增加；单组结果不能独立归因于历史/DCN/接收块/软预算。用户树叶样例未纳入主训练/校准/停止/评价。','', '所有大型缓存、权重、数据和视频只在本地项目内，以large_artifacts.json路径/大小/SHA清单交付，不上传普通Git；代码、协议、审计、CSV、小图、报告和索引提交origin独立分支，远端核验见publication_status.json。']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
