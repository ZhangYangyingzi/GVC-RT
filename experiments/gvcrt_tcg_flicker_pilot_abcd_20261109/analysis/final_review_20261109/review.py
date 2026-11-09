from pathlib import Path
import csv,json,gzip,hashlib,math,statistics,datetime,collections
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PRIMARY='MCResidual_Sobel_RAFT_pixelcenter_v1_PtoP'
RGB='MCResidual_RGB_RAFT_pixelcenter_v1_PtoP'
METRICS=['PSNR','LPIPS','DISTS','FloLPIPS',PRIMARY,RGB,'WarpingError_GTFlow_PWC_PtoP']
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def readcsv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
audit={'scope':'Independent CPU review of saved records; no model inference or training; original status files not overwritten','time_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'errors':[]}
errors=audit['errors']
def check(ok,message):
    if not ok:errors.append(message)
manifest=load(ROOT/'source_manifest.json')
for section in ['code','immutable_manifests']:
    for rel,expected in manifest[section].items():check(sha(ROOT/rel)==expected,'Frozen source mismatch: '+rel)
check(sha(ROOT/'protocol.json')==manifest['protocol_sha256'],'Protocol mismatch')
with gzip.open(ROOT/'plans/train.json.gz','rt') as f:plan=json.load(f)
dev=load(ROOT/'manifests/quick_development.json');dev_by_seq={x['sequence']:x for x in dev}
expected_keys={(x['sequence'],q) for x in dev for q in [0,5,9]}
check(not ({x['source_group'] for x in plan}&{x['source_group'] for x in dev}),'Train/development source overlap')
check(len({x['identity'] for x in plan})==1200,'Training identities not unique')
training={}
for group in 'ABCD':
    completion=load(ROOT/group/'training/completion.json')
    receipts=[load(p) for p in sorted((ROOT/group/'training/receipts').glob('*.json'))]
    check(completion['status']=='PASS' and completion['committed_updates']==300 and len(receipts)==300,group+': incomplete training')
    check([r['additional_step'] for r in receipts]==list(range(1,301)),group+': receipt order')
    check(len({r['frozen_state_SHA256'] for r in receipts})==1,group+': frozen state changed')
    for i,r in enumerate(receipts):
        check(r['sample_keys']==[x['sample_key'] for x in plan[4*i:4*i+4]],group+': plan mismatch '+str(i+1))
        check(r['frozen_exact_bytes_checked'] and r['buffers_reset'] and r['gradient_norm']>0 and r['post_clip_norm']<=1.00001,group+': training audit '+str(i+1))
        check(all(s['frames_supervised']==15 and s['segment_lengths']==[4,4,4,3] for s in r['samples']),group+': BPTT schedule '+str(i+1))
        check(all(s==i+1 for s in r['optimizer_steps']),group+': optimizer step mismatch')
    for step in range(0,301,50):check((ROOT/group/f'training/checkpoints/step_{step:04d}.pt').is_file(),group+': checkpoint missing '+str(step))
    training[group]={'committed_updates':len(receipts),'unique_training_videos':1200,'unique_source_groups':len({x['source_group'] for x in plan}),'summed_update_seconds':sum(x['seconds'] for x in receipts),'stop_reason':completion['stop_reason']}
audit['training']=training
df=pd.read_csv(ROOT/'development_metrics.csv');check(len(df)==648,'Expected648 metric records')
base=df[(df.model=='A2000')&(df.step==0)].set_index(['sequence','QP'])
for (group,step),rows in df.groupby(['model','step']):
    check(len(rows)==36 and set(zip(rows.sequence,rows.QP))==expected_keys,f'{group}/{step}: coverage')
    for _,r in rows.iterrows():
        b=base.loc[(r.sequence,r.QP)]
        check(all(r[k]==b[k] for k in ['real_bytes','bpp','bitstream_SHA256','codec_trace_SHA256','source_RGB_SHA256']),f'{group}/{step}/{r.sequence}/{r.QP}: stream mismatch')
audit.update(evaluation_records=len(df),distinct_development_points=len(base),final_test_points=0,training_QP_counts=dict(collections.Counter(x['external_qp'] for x in plan)))
gt_hashes={};record_files=0
for point in sorted((ROOT/'evaluation').glob('*/step_*/points/*.json')):
    d=load(point);r=d['result']
    check(d['status']=='PASS' and d['codec_features_match_original'] and d['codec_references_forced_to_original'] and d['frozen_components_unchanged'],'Point proof failed: '+str(point))
    for filename,h in d['record_SHA256'].items():check(sha(ROOT/filename)==h,'Record hash mismatch: '+filename);record_files+=1
    ff=readcsv(point.parents[1]/'frames'/(point.stem+'.csv'));tt=readcsv(point.parents[1]/'transitions'/(point.stem+'.csv'))
    check(len(ff)==64 and len(tt)==63,'Frame/transition counts: '+str(point))
    check(sum(int(x['real_bits']) for x in ff)==r['real_bytes']*8 and d['bytes_consumed']==r['real_bytes'],'Rate bytes mismatch: '+str(point))
    native=dev_by_seq[r['sequence']]['row'];pixels=64*native['height']*native['width']
    check(r['bpp']==r['real_bytes']*8/pixels,'bpp recomputation failed: '+str(point))
    psnr=-10*math.log10(max(sum(float(x['pixel_SSE']) for x in ff)/(pixels*3),1e-15))
    check(abs(psnr-r['PSNR'])<1e-9,'PSNR recomputation failed: '+str(point))
    for metric in ['LPIPS','DISTS','MS_SSIM']:check(abs(statistics.mean(float(x[metric]) for x in ff)-r[metric])<1e-9,'Spatial mean failed: '+str(point)+' '+metric)
    for metric in ['FloLPIPS','MCResidual_RGB_RAFT_pixelcenter_v1','MCResidual_Sobel_RAFT_pixelcenter_v1']:
        for records,suffix in [(tt,''),(tt[1:],'_PtoP')]:
            vals=[float(x[metric]) for x in records if x[metric]!='']
            check(bool(vals) and r[metric+suffix] is not None and abs(statistics.mean(vals)-r[metric+suffix])<1e-9,'Temporal mean failed: '+str(point)+' '+metric+suffix)
    for x in tt:
        key=(r['sequence'],int(x['to_frame']));v=(x['GT_backward_flow_SHA256'],x['GT_valid_mask_SHA256'])
        if key in gt_hashes:check(v==gt_hashes[key],'GT flow/mask changed across QP/models')
        else:gt_hashes[key]=v
audit.update(record_files_hash_verified=record_files,gt_flow_mask_pairs_verified=len(gt_hashes))
checkpoints=load(ROOT/'checkpoint_manifest.json')['new_checkpoints']
for item in checkpoints:check(sha(item['local_path'])==item['sha256'],'Checkpoint hash mismatch: '+item['group']+' '+item['filename'])
audit['checkpoint_files_hash_verified']=len(checkpoints)
vids=[]
for p in sorted((ROOT/'videos/manifests').glob('*.json')):
    d=load(p);check(d['status']=='PASS' and d['frames']==128,'Video manifest incomplete: '+str(p))
    for path,key in [('path','sha256'),('preview_path','preview_sha256')]:check(sha(d[path])==d[key],'Video hash mismatch: '+d[path])
    vids.append({k:d[k] for k in ['sequence','model','step','frames','path','preview_path','stream_SHA256','stream_bytes']})
audit.update(individual_128frame_videos_verified=len(vids),synchronized_video_manifest_present=(ROOT/'videos/manifest.json').is_file(),pipeline_status=load(ROOT/'status.json')['status'],publication_phase=load(ROOT/'publication_status.json')['phase'])
audit['review_status']='PASS_SAVED_TRAINING_AND_METRICS' if not errors else 'FAIL'
audit['pipeline_complete']=False
audit['missing_deliverables']=['Four synchronized comparison videos / final montage manifest','Fresh final_integrity.json','Final-results publication receipt']
audit['inputs']={n:sha(ROOT/n) for n in ['development_metrics.csv','temporal_metrics_by_frame.csv','training_history.csv','protocol.json','source_manifest.json']}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False));pd.DataFrame(vids).to_csv(OUT/'individual_video_index.csv',index=False)
summary=[]
for (group,step),rr in df.groupby(['model','step']):
    for q in [0,5,9,'all']:
        a=rr if q=='all' else rr[rr.QP==q];b=base if q=='all' else base.xs(q,level='QP')
        row={'model':group,'step':step,'QP':q,'points':len(a),'bpp':a.bpp.mean(),'kbps':a.kbps.mean()}
        for k in METRICS:row[k]=a[k].mean();row[k+'_delta_vs_A2000']=a[k].mean()-b[k].mean();row[k+'_percent_vs_A2000']=100*(a[k].mean()/b[k].mean()-1)
        summary.append(row)
s=pd.DataFrame(summary);s.to_csv(OUT/'summary_by_model_step_qp.csv',index=False)
final=df[((df.model.isin(list('ABCD')))&(df.step==300))|df.model.isin(['Original','A2000'])].copy()
for k in METRICS:final[k+'_percent_vs_A2000']=[100*(r[k]/base.loc[(r.sequence,r.QP),k]-1) for _,r in final.iterrows()]
final.to_csv(OUT/'final_point_deltas.csv',index=False)
tf=pd.read_csv(ROOT/'temporal_metrics_by_frame.csv');tf=tf[tf.transition_type=='P_to_P']
edge=tf.groupby(['model','step'])[['GT_Sobel_strength','reconstruction_Sobel_strength']].mean()
edge['recon_over_GT']=edge.reconstruction_Sobel_strength/edge.GT_Sobel_strength
edge['percent_vs_A2000']=100*(edge.reconstruction_Sobel_strength/edge.loc[('A2000',0),'reconstruction_Sobel_strength']-1)
edge.to_csv(OUT/'edge_strength.csv')
pe=tf.groupby(['model','step','sequence','QP'])[['GT_Sobel_strength','reconstruction_Sobel_strength']].mean().reset_index();pe['recon_over_GT']=pe.reconstruction_Sobel_strength/pe.GT_Sobel_strength;pe.to_csv(OUT/'edge_strength_by_video_qp.csv',index=False)
colors={'Original':'#777777','A2000':'#111111','A':'#1565c0','B':'#ef6c00','C':'#2e7d32','D':'#ad1457'}
names={'Original':'Original GVC-RT','A2000':'A2000 s=0.5','A':'A +300','B':'B +300','C':'C +300','D':'D +300'}
fig,axes=plt.subplots(2,3,figsize=(15,8))
for ax,k,label in zip(axes.flat,['PSNR','LPIPS','DISTS','FloLPIPS',PRIMARY,RGB],['PSNR (dB), higher better','LPIPS, lower better','DISTS, lower better','FloLPIPS, lower better','Sobel MC residual, lower better','RGB MC residual, lower better']):
    for group in ['Original','A2000','A','B','C','D']:
        z=final[final.model==group].groupby('QP')[[k,'bpp']].mean().sort_values('bpp');ax.plot(z.bpp,z[k],'-o',label=names[group],color=colors[group],linewidth=1.5,markersize=4)
    ax.set(xlabel='Real RANS bpp (equal video weights)',ylabel=label);ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8);fig.suptitle('ULong development subset: 12 videos, QP 0/5/9, 64 frames; matched bitstreams');fig.tight_layout();fig.savefig(OUT/'rate_quality_ULong_dev12.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,k,label in zip(axes,[PRIMARY,'FloLPIPS','DISTS'],['Sobel MC residual change (%)','FloLPIPS change (%)','DISTS change (%)']):
    for g in 'ABCD':
        z=s[(s.model==g)&(s.QP=='all')].sort_values('step');ax.plot(z.step,z[k+'_percent_vs_A2000'],'-o',label=g,color=colors[g])
    ax.axhline(0,color='black',linestyle='--');ax.set(xlabel='Additional optimizer updates',ylabel=label);ax.grid(alpha=.2)
axes[0].legend();fig.suptitle('ULong dev12: ratio of equal-point means vs fixed A2000 s=0.5; lower is better');fig.tight_layout();fig.savefig(OUT/'training_tradeoff.png',dpi=160);plt.close(fig)
lines=['# ABCD 300 步独立结果复核','','范围：12 个 ULong 开发视频 × QP 0/5/9，每点64帧。不是完整开发集或最终测试集。','四组均从 A2000（s=0.5）起步：TCG+LoRA联合训练、4帧分段反传、新增RGB/Sobel运动补偿残差损失、RAFT权重0.05。B/D额外开放Bridge mlp[3]；C/D额外对齐历史特征。','码流、字节数、bpp、压缩特征轨迹和输入源哈希按点完全一致。百分比为候选均值/参考均值-1，负值表示误差减少，不是码率节省。','','|方案|Sobel时序误差%|RGB时序误差%|LPIPS%|DISTS%|FloLPIPS%|PSNR变化dB|','|---|---:|---:|---:|---:|---:|---:|']
for g in 'ABCD':
    r=s[(s.model==g)&(s.step==300)&(s.QP=='all')].iloc[0];lines.append('|'+g+'|'+'|'.join(f'{r[k+"_percent_vs_A2000"]:+.3f}' for k in [PRIMARY,RGB,'LPIPS','DISTS','FloLPIPS'])+f'|{r["PSNR_delta_vs_A2000"]:+.4f}|')
lines+=['','## 分QP结果','','|方案|QP|Sobel%|LPIPS%|DISTS%|FloLPIPS%|','|---|---:|---:|---:|---:|---:|']
for g in 'ABCD':
    for q in [0,5,9]:
        r=s[(s.model==g)&(s.step==300)&(s.QP==q)].iloc[0];lines.append('|'+g+'|'+str(q)+'|'+'|'.join(f'{r[k+"_percent_vs_A2000"]:+.3f}' for k in [PRIMARY,'LPIPS','DISTS','FloLPIPS'])+'|')
lines+=['','## 判断与边界','','- A/B更均衡。B对A额外收益很小，单次运行不能宣称稳定胜出；C/D对齐未显示净收益，FloLPIPS比A2000更差。','- 四组36/36点的Sobel和RGB时序误差都下降，step300都通过预注册的连续两次数值筛选。筛选用逐视频跨QP比值的中位数，和上表的均值比不是同一统计量。','- 重建Sobel强度比A2000下降约8%，均值已低于GT：可能兼有伪纹理抑制和真实细节平滑，需原分辨率视频/ROI确认，不能直接称为消除闪烁。','- 四组共享新时序损失、RAFT权重变化、联合训练和4帧反传，无法单独归因于BPTT或某项loss。','- B/D只开放Bridge最后7524参数，原始de-tokenizer仍冻结。此次不能判断解冻主干是否更好。','- 最初person_dog视频dce599d3-d196-4d87-b2b1-698acdbbb3d4不在此开发子集，该具体案例尚未用新权重验证。','- 300步是预设上限，200→300仍有小幅改进，不代表收敛，也不能据此直接扩大训练。','','## 交付状态','','四组训练300/300；step0/100/200/300开发评测和Original/A2000基线齐全，共648条记录。28个原分辨率128帧单模型视频及预览已保存，四个同步拼接视频未完成。','FFmpeg xstack不支持fill导致拼接失败；自动恢复又进入工程检查，因训练缓存按协议清理而失败。这不等于正式训练失败。原final_integrity.json是约31步的旧快照，GitHub回执仍为preparation，不能宣称最终审计/结果发布完成。','本目录audit.json独立复核保存的记录、哈希和聚合；不覆盖状态或启动训练。','','## 下一步建议','','先同步比较A300/B300与A2000原分辨率纹理区域，确认稳定性改善并非主要来自变软，补齐视频和最终审计。若肉眼改善有限，再以A300作对照验证只开放de-tokenizer stage4+head，其余目标和数据一致。不要把子集结果外推为所有数据集有效。','','图：rate_quality_ULong_dev12.png包含原版GVC-RT、A2000及四组；training_tradeoff.png展示训练趋势。','表：summary_by_model_step_qp.csv、final_point_deltas.csv、edge_strength_by_video_qp.csv。视频路径见individual_video_index.csv。','']
(OUT/'REPORT.md').write_text(chr(10).join(lines))
print(json.dumps({'review_status':audit['review_status'],'errors':errors,'points':len(df),'record_files_verified':record_files,'checkpoint_files_verified':len(checkpoints),'videos_verified':len(vids),'output':str(OUT)},ensure_ascii=False))
