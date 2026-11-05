"""Validate every independently measured point, then add the original R-D line."""
from common import *
import math
import numpy as np


def read_csv(path):
    return list(csv.DictReader(Path(path).open(newline='')))


def finalize():
    protocol=verify_protocol();videos=load(ROOT/'videos.json')
    previous=read_csv(TCG/'metrics_v3.csv')
    baseline={(r['split'],r['domain'],r['sequence'],int(r['QP'])):r for r in previous if r['model_state']=='parent0'}
    old_points=TCG/'evaluation_v3'/load(TCG/'final_evaluation_v3.json')['batch_id']/'points'
    original=[];comparisons=[]
    for v in videos:
        for qp in protocol['QP']:
            key=f'GVC_RT_original_{v["domain"]}_{v["sequence"]}_qp{qp}'
            p=load(ROOT/f'points/{key}.json');r=p['result']
            assert p['status']=='PASS' and p['fresh_encode'] and p['independent_decode'] and p['real_RANS']
            assert not p['input_stream_reused'] and not p['TCG_modules_present'] and not p['adapted_checkpoint_loaded']
            assert p['original_codec_modules_only'] and p['model_hashes_unchanged']
            assert p['protocol_sha256']==sha(ROOT/'protocol.json') and r['original_commit']==COMMIT
            assert (r['split'],r['domain'],r['sequence'],r['QP'])==(v['split'],v['domain'],v['sequence'],qp)
            stream=ROOT/'bitstreams'/r['bitstream_file']
            assert sha(stream)==r['bitstream_sha256'] and stream.stat().st_size==p['bytes_consumed']==r['real_bytes']
            assert all(math.isfinite(r[m]) for m in METRICS)
            frames=read_csv(ROOT/f'frames/{key}.csv');trans=read_csv(ROOT/f'transitions/{key}.csv')
            assert len(frames)==64 and len(trans)==63
            assert [int(f['frame']) for f in frames]==list(range(64))
            assert [int(f['to_frame']) for f in trans]==list(range(1,64))
            assert [f['source_frame_RGB_sha256'] for f in frames]==v['frame_rgb_sha256']
            assert sum(int(f['real_bits']) for f in frames)==8*r['real_bytes']
            assert abs(r['bpp']-8*r['real_bytes']/(64*v['height']*v['width']))<1e-12
            assert abs(r['kbps']-8*r['real_bytes']*v['fps']/64/1000)<1e-10
            mse=sum(float(f['pixel_SSE']) for f in frames)/(64*v['height']*v['width']*3)
            assert abs(r['PSNR']+10*math.log10(max(mse,1e-15)))<1e-10
            for m in ('MS_SSIM','LPIPS','DISTS'):
                assert abs(r[m]-sum(float(f[m]) for f in frames)/64)<1e-10
            for m in ('FloLPIPS','WarpingError_ReconFlow_PWC','WarpingError_GTFlow_PWC'):
                assert abs(r[m]-sum(float(t[m]) for t in trans)/63)<1e-10
            reconstruction=hashlib.sha256(''.join(f['reconstruction_tensor_sha256'] for f in frames).encode()).hexdigest()
            assert reconstruction==p['reconstruction_sha256']
            for module,record in p['source_imports'].items():
                path=Path(record['path']);assert path.is_relative_to(ORIGINAL)
                assert record['sha256']==protocol['original_source_sha256'][str(path.relative_to(ORIGINAL))]
            old=baseline[(v['split'],v['domain'],v['sequence'],qp)]
            oldp=load(old_points/f'parent0_{v["domain"]}_{v["sequence"]}_qp{qp}.json')
            assert p['actual_qps']==oldp['actual_qps'] and p['source_frame_indices']==oldp['source_frame_indices']
            assert r['source_RGB_sha256']==old['source_RGB_sha256'] and r['source_sha256']==old['source_sha256']
            oldtrans=read_csv(old_points.parent/f'transitions/parent0_{v["domain"]}_{v["sequence"]}_qp{qp}.csv')
            assert [t['GT_flow_sha256'] for t in trans]==[t['GT_flow_sha256'] for t in oldtrans], 'Metric GT flow changed'
            comparisons.append(dict(domain=r['domain'],split=r['split'],sequence=r['sequence'],QP=qp,
                                    identical_bitstream=r['bitstream_sha256']==old['bitstream_sha256'],
                                    identical_reconstruction=reconstruction==oldp['reconstruction_sha256'],
                                    bpp_difference=r['bpp']-float(old['bpp']),
                                    **{m+'_difference':r[m]-float(old[m]) for m in METRICS}))
            original.append(r)
    assert len(original)==135
    source_audit=load(ROOT/'source_frame_audit.json')
    assert source_audit['status']=='PASS' and source_audit['protocol_sha256']==sha(ROOT/'protocol.json')
    write_csv(ROOT/'metrics_original.csv',original)
    write_csv(ROOT/'original_vs_parent0.csv',comparisons)
    write_csv(ROOT/'metrics_with_original.csv',previous+original)
    plot(previous+original)
    summary=dict(status='PASS',expected_points=135,completed_points=135,previous_points=675,combined_points=810,
                 original_commit=COMMIT,original_codec_no_TCG=True,fresh_encode=True,independent_decode=True,
                 identical_streams_to_parent0=sum(c['identical_bitstream'] for c in comparisons),
                 identical_reconstructions_to_parent0=sum(c['identical_reconstruction'] for c in comparisons),
                 max_absolute_metric_differences={m:max(abs(c[m+'_difference']) for c in comparisons) for m in METRICS},
                 source_frames_matched=True,GT_flow_hashes_matched=True,rate_and_metric_aggregation_verified=True,
                 protocol_sha256=sha(ROOT/'protocol.json'),metrics_original_sha256=sha(ROOT/'metrics_original.csv'),
                 previous_metrics_unchanged=sha(TCG/'metrics_v3.csv')==protocol['previous_metrics_sha256'])
    dump(ROOT/'final_integrity.json',summary)
    report=['# GVC-RT 原版独立评测', '',
            f'已完成 45 个视频 × 3 个 QP = 135 个原版实测点。原版代码固定于 `{COMMIT}`，直接加载项目 checkpoints 中的原版 I/P 权重，不包含 TCG，不加载 v6 实验检查点。', '',
            '每点均重新编码为实际 RANS 码流，并用新建解码器独立解码。视频、64 帧窗口、QP 调度、精度、预处理和指标计算与已有 TCG 实验对齐。这里是同协议下的原版编解码实现评测，不声称复现论文全部默认测试设置。', '',
            f'与 parent0 比较：{summary["identical_streams_to_parent0"]}/135 点码流完全相同，{summary["identical_reconstructions_to_parent0"]}/135 点重建哈希完全相同。若两条曲线重合，仍保留独立结果和图例，不复制 parent0 数值代替原版测试。', '',
            '每张图有六条线：GVC-RT original、parent0（零残差实验基线）、TCG parent5000、A+1500、A 最优+500、B 最优+1500。', '',
            '| 数据集/划分 | 视频数 | 图 |', '|---|---:|---|',
            '| ULong 新测试集 | 32 | [R-D](plots/ulong_new_test_rate_quality.png) |',
            '| ULong 历史子集 | 6 | [R-D](plots/ulong_historical_rate_quality.png) |',
            '| HEVC-B 历史子集 | 5 | [R-D](plots/hevc_b_rate_quality.png) |',
            '| UVG 历史子集（HoneyBee、Jockey） | 2 | [R-D](plots/uvg_holdout_rate_quality.png) |', '',
            '数据集图只在各自子集内部平均；逐视频图完全不跨视频平均。各图只有 QP 0、5、9 三个实测点，未估算 BD-rate。', '',
            '[原版逐点指标](metrics_original.csv) · [与 parent0 逐点比较](original_vs_parent0.csv) · [完整性检查](final_integrity.json)', '',
            '## 逐视频曲线', '']
    for v in videos:
        group=GROUP_BY_DOMAIN[v['domain']]
        report.append(f'- {v["domain"]}: [{v["sequence"]}](plots/per_video/{group}/{v["sequence"]}.png)')
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary,indent=2))


GROUP_BY_DOMAIN=dict(ulong_new='ulong_new_test',ulong='ulong_historical',hevc_b='hevc_b',uvg_holdout='uvg_holdout')


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    out=ROOT/'plots';out.mkdir(exist_ok=True)
    states=['GVC_RT_original','parent0','parent5000','A_common','A_best_dev','B_common']
    labels=['GVC-RT original (independent)','parent0 (zero-TCG baseline)','TCG parent5000','A +1500 (common)','A +500 (dev-best)','B +1500 (dev-best)']
    colors=['#111827','#94a3b8','#2563eb','#d97706','#0d9488','#c026d3']
    styles=['-','--','--',':','-.','-']
    metrics=['LPIPS','DISTS','FloLPIPS','PSNR','MS_SSIM','WarpingError_GTFlow_PWC']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.spines.top':False,'axes.spines.right':False})
    def draw(points,title,path,count,individual):
        fig,axes=plt.subplots(2,3,figsize=(14.4,8.8))
        vals={(r['model_state'],int(r['QP'])):r for r in points}
        assert len(vals)==18
        for ax,m in zip(axes.flat,metrics):
            for state,label,color,style in zip(states,labels,colors,styles):
                p=sorted([vals[state,q] for q in (0,5,9)],key=lambda r:float(r['bpp']))
                ax.plot([float(r['bpp']) for r in p],[float(r[m]) for r in p],label=label,color=color,linestyle=style,
                        marker='s' if state=='GVC_RT_original' else 'o',markerfacecolor='none' if state=='GVC_RT_original' else color,
                        markersize=7 if state=='GVC_RT_original' else 4,linewidth=2 if state in ('GVC_RT_original','B_common') else 1.4,
                        zorder=5 if state=='GVC_RT_original' else 3)
            display={'MS_SSIM':'MS-SSIM','WarpingError_GTFlow_PWC':'GT-flow warp error'}.get(m,m)
            direction='higher' if m in ('PSNR','MS_SSIM') else 'lower'
            ax.set_title(f'{display} ({direction} is better)',loc='left',fontsize=11,fontweight='bold')
            ax.set_xlabel('Measured rate (bpp)' if individual else 'Mean measured rate within this subset (bpp)')
            ax.set_ylabel('dB' if m=='PSNR' else display);ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(True,color='#e2e8f0');ax.set_axisbelow(True)
        fig.suptitle(title,fontsize=16,fontweight='bold',y=.985)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91 if individual else .947),ncol=3,frameon=False,fontsize=9.5)
        note=('Single-video results: no averaging across videos.' if individual else f'Equal-video means within this {count}-video subset only; no cross-dataset averaging.')
        note+='\nGVC-RT original: independently encoded and decoded with original code and weights, with no TCG modules.\n'
        note+='Three measured QPs (0, 5, 9); lines connect measurements only. Each model uses its own measured stream rate.\nGT-flow error also includes reconstruction/flow/occlusion error; it is not an isolated flicker measure.'
        fig.text(.045,.018,note,fontsize=8.8,color='#475569',va='bottom')
        fig.tight_layout(rect=[.015,.12,.995,.885],h_pad=2,w_pad=2.1)
        path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150,facecolor='white');plt.close(fig)
    aggregate=[]
    for domain,group in GROUP_BY_DOMAIN.items():
        subset=[r for r in rows if r['domain']==domain];names=sorted({r['sequence'] for r in subset})
        count={'ulong_new':32,'ulong':6,'hevc_b':5,'uvg_holdout':2}[domain]
        assert len(names)==count and len(subset)==count*18
        records=[]
        for state in states:
            for qp in (0,5,9):
                selected=[r for r in subset if r['model_state']==state and int(r['QP'])==qp]
                assert len(selected)==count
                records.append(dict(dataset_group=group,model_state=state,QP=qp,video_count=count,
                                    **{m:float(np.mean([float(r[m]) for r in selected])) for m in ['bpp','kbps']+metrics}))
        aggregate.extend(records)
        title={'ulong_new':'ULong: new test subset','ulong':'ULong: historical subset','hevc_b':'HEVC-B: historical subset','uvg_holdout':'UVG: held-out historical subset'}[domain]
        draw(records,f'{title} | {count} videos',out/f'{group}_rate_quality.png',count,False)
        for name in names:
            draw([r for r in subset if r['sequence']==name],f'{title}\n{name}',out/f'per_video/{group}/{name}.png',1,True)
    write_csv(ROOT/'rd_points_by_dataset.csv',aggregate)


if __name__=='__main__':
    finalize()
