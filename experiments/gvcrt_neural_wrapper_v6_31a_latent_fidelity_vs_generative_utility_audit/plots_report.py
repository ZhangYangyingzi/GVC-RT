"""Deterministic figures and conclusions conditional on the shared proxy teacher."""
import numpy as np
import pandas as pd
from audit_io import *


def finish(points,primary,inversions,summaries,tolerance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'savefig.bbox':'tight','figure.dpi':150})
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{name}.{ext}')
        plt.close(fig)
    marks={'Original':'o','B1':'^','B_covstats':'s'}
    for metric in ('LPIPS','DISTS','FloLPIPS'):
        fig,ax=plt.subplots(figsize=(6,4))
        for method,g in points.groupby('method'):
            sizes=8+40*g.actual_bpp/points.actual_bpp.max()
            ax.scatter(g.P_cosine,g['P_'+metric],s=sizes,marker=marks[method],alpha=.55,label=method)
        ax.set(xlabel='Continuous proxy latent cosine distance (lower is better)',ylabel=f'P-frame {metric} (lower is better)')
        ax.legend();save(fig,'latent_vs_'+metric.lower())
        fig,ax=plt.subplots(figsize=(6,4))
        for method,g in primary.groupby('method'):
            ax.scatter(g.delta_P_cosine,g['delta_P_'+metric],marker=marks[method],label=method)
        ax.axvline(0,c='0.5',lw=.8);ax.axhline(0,c='0.5',lw=.8)
        ax.set(xlabel='Delta proxy latent cosine distance (method - Original)',ylabel=f'Delta P-frame {metric} (method - Original)')
        ax.text(.98,.02,'latent worse / output better',transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
        if len(primary):ax.legend()
        save(fig,'delta_latent_vs_delta_'+metric.lower())
    fig,ax=plt.subplots(figsize=(8,4))
    selected=summaries[(summaries.population=='primary_1pct')&(summaries.frame_scope=='P')&(summaries.latent_metric=='cosine')&(summaries['groupby']=='dataset')]
    ds=sorted(points.dataset.unique());x=np.arange(len(ds));width=.25
    for i,metric in enumerate(('LPIPS','DISTS','FloLPIPS')):
        values=dict(zip(selected[selected.quality_metric==metric]['group'],selected[selected.quality_metric==metric].inversion_fraction))
        ax.bar(x+(i-1)*width,[values.get(d,np.nan) for d in ds],width,label=metric)
    ax.set_xticks(x,ds,rotation=20);ax.set(ylabel='Inversion fraction (primary <=1% pairs)',ylim=(0,1));ax.legend()
    save(fig,'inversion_fraction_by_dataset')
    examples=inversions[(inversions.latent_metric=='cosine')&(inversions.quality_metric=='LPIPS')&(inversions.frame_scope=='P')&inversions.is_inversion].copy()
    if len(examples):
        examples['inversion_magnitude']=examples.delta_P_cosine*(-examples.delta_P_LPIPS)
        examples=examples.sort_values(['rate_gap','inversion_magnitude','dataset','video','method'],ascending=[True,False,True,True,True]).head(4)
    examples.to_csv(ROOT/'figures/selected_counterexamples/selection.csv',index=False)
    dump('counterexample_selection.json',dict(count=len(examples),frame=32,rule='rate gap ascending, then delta_cosine*(-delta_LPIPS) descending; deterministic lexical final ties',
         no_human_selection=True,render_pending=bool(len(examples))))
    # Quality effect estimates: each primary pair corresponds to one video/method.
    utility=[];rng=np.random.default_rng(20261014)
    for method,g in primary.groupby('method'):
        for metric in ('LPIPS','DISTS','FloLPIPS'):
            vals=g['delta_P_'+metric].to_numpy(float)
            draws=np.mean(rng.choice(vals,(2000,len(vals)),replace=True),axis=1) if len(vals)>1 else []
            utility.append(dict(method=method,metric=metric,n=len(vals),mean_delta=float(vals.mean()),
                                improved_count=int((vals<0).sum()),ci_low=float(np.quantile(draws,.025)) if len(draws) else None,
                                ci_high=float(np.quantile(draws,.975)) if len(draws) else None))
    pd.DataFrame(utility).to_csv(ROOT/'matched_quality_summary.csv',index=False)
    b1=primary[primary.method=='B1']
    bad=b1.delta_P_cosine>tolerance['cosine']
    same=b1.delta_P_cosine.abs()<=tolerance['cosine']
    benefits=sum((b1['delta_P_'+m]<0).astype(int) for m in ('LPIPS','DISTS','FloLPIPS'))
    reliable=b1[(bad|same)&(benefits>=2)]
    strict=b1[bad&(benefits>=2)]
    # Pre-registered conservative rule: B1 alone drives the primary conclusion.
    strong=(reliable.dataset.nunique()>=2 and reliable.video.nunique()>=4 and len(reliable)>=4
            and len(set(zip(reliable.original_QP,reliable.method_QP)))>=2)
    case='Not assessable' if not len(b1) else 'Strong' if strong else 'Partial' if len(reliable) else 'Not supported'
    wording={
      'Not assessable':'No primary <=1% B1 matched-rate pair is available; the hypothesis is not assessed by this audit.',
      'Strong':'Relative to the fixed paper-compatible proxy teacher, improved tokenizer-latent fidelity is not a necessary condition for improved generative reconstruction at matched actual rate.',
      'Partial':'Relative to the fixed paper-compatible proxy teacher, latent fidelity and generative utility can decouple in some finite-rate comparisons. The evidence does not establish a general rule.',
      'Not supported':'The current proxy-teacher evidence does not support the claim that latent fidelity differs from generative utility. Where observed gains coincide with better proxy fidelity, they are consistent with more rate-efficient preservation of the proxy target latent; this does not identify the unpublished original target.'}[case]
    answers=[]
    for method in ('B1','B_covstats'):
        g=primary[primary.method==method]
        means={m:float(g['delta_P_'+m].mean()) if len(g) else None for m in ('LPIPS','DISTS','FloLPIPS','cosine')}
        answers.append(dict(method=method,pairs=len(g),datasets=g.dataset.nunique(),videos=g.video.nunique(),mean_deltas=means))
    dump('conclusion.json',dict(case=case,scope='fixed paper-compatible proxy teacher only',primary_method='B1',
         reliable_B1_counterexamples=len(reliable),strict_B1_inversions=len(strict),datasets=int(reliable.dataset.nunique()),
         videos=int(reliable.video.nunique()),QP_pairs=len(set(zip(reliable.original_QP,reliable.method_QP))),
         exact_original_training_reproduction=False,wording=wording,answers=answers))
    header='''# v6.31a paper-faithful proxy audit

1. **Is final perceptual quality better at the same rate?** The table below gives method-minus-Original P-frame means over each method's primary <=1% matched pairs. Negative means improvement; LPIPS, DISTS and FloLPIPS are assessed separately. Coverage can differ between methods.
2. **Is actual generator input closer to the LFQ target?** The cosine delta is relative to the same fixed continuous pre-LFQ proxy teacher. Negative means closer. This is not a verified original GVC-RT training teacher.
3. **Do real matched-rate worse/same-latent, better-output counterexamples exist?** Counts below require worse/same primary proxy cosine fidelity and improvement in at least two principal quality metrics; all individual metric tests are also saved.
4. **How broad is the phenomenon?** Dataset/video/QP-pair counts below use primary B1 pairs only. B_covstats is secondary; MCL29 is a sensitivity subset, not an independent dataset.
5. **Decision:** {case}, conditional on this proxy and the measured coverage.
6. **What does this say about P?** {p_interpretation} P and receiver adaptation are evaluated jointly; their individual causal contributions are not isolated by this comparison.

'''.format(case=case,p_interpretation=('The observed counterexamples show that the P+B system can improve output without improving this proxy fidelity.' if len(reliable) else 'No reliable B1 counterexample was found by the pre-registered rule. Any output gains accompanied by better proxy fidelity are compatible with improved retention of that proxy representation.'))
    text=header+'| Method | Primary pairs | LPIPS delta | DISTS delta | FloLPIPS delta | Proxy cosine delta |\n|---|---:|---:|---:|---:|---:|\n'
    for a in answers:
        vals=[a['mean_deltas'][m] for m in ('LPIPS','DISTS','FloLPIPS','cosine')]
        text+='| '+a['method']+' | '+str(a['pairs'])+' | '+' | '.join('N/A' if v is None else f'{v:.6g}' for v in vals)+' |\n'
    text+=f'\nPrimary B1 reliable counterexamples: {len(reliable)}; strict worse-latent subset: {len(strict)}; datasets: {reliable.dataset.nunique()}; videos: {reliable.video.nunique()}; distinct QP pairs: {len(set(zip(reliable.original_QP,reliable.method_QP)))}.\n\n'
    text+='**Observation:** '+wording+'\n\n**Evidence:** persisted RANS bytes and independent decode, exact decoder-input hooks, fixed source sampling, pre-registered matching independent of quality, 10 points decoded twice for numerical tolerance, and video-cluster intervals. See `matched_quality_summary.csv`, `inversion_summary.csv`, `bootstrap_summary.csv`, `numerical_noise_floor.json` and figures. Quantized-target distances are separate diagnostics; no actual transmitted LFQ token-match claim is made.\n\n'
    text+='**Interpretation and limits:** All conclusions are conditional on the fixed paper-compatible proxy. Neither exact original teacher identity nor the full unpublished `0.2 L_cos + 0.02 L_margin` implementation is reproduced. Cosine is an explicitly defined proxy distance, not an official primary metric. Correlation and residual correlation are associations, not causal effects. Unmatched videos contribute no primary pair. The 2% neighborhood is secondary and cannot substantiate an exact same-rate claim. P-frame fidelity is primary; I0 and all-frame results are separately retained. FID is pooled distribution evidence only. MCL29 excludes historical overlap source17 and is not independently counted.\n\n'
    text+='**P diagnostics:** `wrapper_diagnostics.csv` records source-to-P RGB residual and fixed frequency-band energy changes; `latent_fidelity_per_frame.csv` records actual frame bits and payload bytes remain in per-point quality parts. The coder exposes combined entropy streams rather than uniquely attributable channel/spatial bit budgets, so no invented allocation breakdown or differentiable rate estimate is reported. No new entropy estimator was introduced.\n\n'
    text+='Optional equal-norm perturbation probe was not executed; natural coding evidence remains the basis of this audit. Counterexample visualizations use fixed frame32 and automatically selected smallest-rate-gap pairs, then inversion magnitude. No counterexample is fabricated when the selection is empty.\n'
    (ROOT/'REPORT.md').write_text(text)
