"""Render measurements with explicit rate and inference limitations."""
import csv
import json
from pathlib import Path

P=Path(__file__).resolve().parent


def rows(name):
    with (P/name).open() as f:return list(csv.DictReader(f))


def f(x,n=5):
    try:return f'{float(x):.{n}f}'
    except (TypeError,ValueError):return '—'


def table(headers,data):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in line)+' |' for line in data])


def main():
    evals=rows('evaluation.csv');matches=rows('matched_rate_comparisons.csv')
    search=rows('rate_match_search.csv');selection=rows('rate_match_selection.csv');sensitivity=rows('interpolation_sensitivity.csv')
    audit=json.loads((P/'delivery_audit.json').read_text())
    pre=json.loads((P/'preflight.json').read_text())
    linear_flips=sum(x.get('dense_vs_sparse_linear_sign_agrees')=='False' for x in sensitivity)
    log_flips=sum(x.get('dense_vs_sparse_log_sign_agrees')=='False' for x in sensitivity)
    parts=['# v3 rate and spatial multiplier ablation','',
           'Fixed step 300 inference only. All reported rates include the 16-byte v3 experiment header. '
           'No training or weight update was performed. Lower LPIPS, DISTS and FloLPIPS is better; higher PSNR is better.','',
           '## Execution and accounting','',
           f'- Preflight: {pre["status"]}; delivery audit: {audit["status"]}.',
           f'- Protocol result records: {audit["protocol_tasks"]}; 16 native-stream anchors and 5 existing QP9 supplement points were rewrapped from v2, '
           f'{audit["new_full_tasks"]} full protocol evaluations were newly run.',
           f'- Rate-only candidates: {len([x for x in search if x["rate_only"]=="True"])}; '
           f'new selected-alpha full evaluations: {len([x for x in selection if x["alpha_index"]!="2"])}.',
           '- Encoder/decoder consistency for newly encoded tasks checks every frame reference and every P-frame latent and qmap. '
           'All 21 reused streams also pass fresh v2/v3 decoder comparisons with exact RGB, reference, latent and qmap hashes on all 64 frames. '
           'A and IDENTITY match native streams and all codec states at all four stage-2 source/QP combinations.',
           '- Two additional alpha=0.8 rate-only encodes were performed for the preflight synchronization checks, '
           'not for candidate selection. They are separately audited in `delivery_audit.json`.',
           '- Preflight also contains three full metric runs (A/MAP/IDENTITY on UVal0 QP2). '
           'The whole round therefore used 39 full metric runs including preflight, within the cap of 44. '
           'The final receiver audits only decode existing streams and do not recompute quality metrics.',
           '- `evaluation.csv` gives total bytes, codec payload, native header, experiment header, P payload, P-only estimated y/z bits, '
           'runtime and peak memory. Interleaved RANS does not expose exact y/z payload byte partitions.','',
           '## A. A versus MAP: measured rates and bracketed interpolation','',
           '### Same external QP: measured, with rate differences','',
           'These rows compare the same external QP, not equal rate. Each delta is MAP minus A.','']
    same_qp=[x for x in matches if x['comparison']=='A_vs_MAP_same_QP']
    parts += [table(['video','QP','A bpp','MAP bpp','rate gap bpp','delta LPIPS','delta DISTS','delta FloLPIPS','delta PSNR'],
        [[source,qp,f(next(x for x in same_qp if x['sequence']==source and x['external_qp']==qp)['reference_actual_bpp'],6),
          f(next(x for x in same_qp if x['sequence']==source and x['external_qp']==qp)['method_actual_bpp'],6),
          f(next(x for x in same_qp if x['sequence']==source and x['external_qp']==qp)['rate_difference_bpp'],6),
          *[f(next(x for x in same_qp if x['sequence']==source and x['external_qp']==qp and x['metric']==metric)['measured_delta_method_minus_reference'])
            for metric in ('LPIPS','DISTS','FloLPIPS','PSNR')]]
         for source,qp in dict.fromkeys((x['sequence'],x['external_qp']) for x in same_qp)]),'',
           '### Nearest measured A point and bracketed estimates','',
           'Each target below is the fixed measured MAP QP2 or QP7 rate. Nearest-point rows are measured but retain a rate gap. '
           'Interpolation uses only measured A points bracketing that fixed target; missing brackets are left empty. '
           'The interpolated A quality is an estimate, not a measurement.','']
    for metric in ('LPIPS','DISTS','FloLPIPS','PSNR'):
        subset=[x for x in matches if x['comparison']=='A_vs_MAP' and x['metric']==metric]
        parts += [f'### {metric}','',table(
            ['video','MAP QP','MAP bpp','MAP measured','nearest A QP / bpp','rate gap bpp','nearest A measured','MAP−A nearest','MAP−A linear','MAP−A log','same sign'],
            [[x['sequence'],x['MAP_QP'],f(x['MAP_actual_bpp'],6),f(x['MAP_measured_quality']),
              f'{x["nearest_A_QP"]} / {f(x["nearest_A_actual_bpp"],6)}',f(x['nearest_rate_difference_bpp'],6),
              f(x['nearest_A_measured_quality']),f(x['nearest_measured_delta_MAP_minus_A']),
              f(x.get('delta_MAP_minus_A_linear')),f(x.get('delta_MAP_minus_A_log')),
              x.get('interpolation_delta_sign_agrees','no bracket')] for x in subset]),'']
    parts += ['Interpolation sensitivity is tabulated in `interpolation_sensitivity.csv`; there is no extrapolation or BD-rate estimate. '
              'Sparse estimates use only the existing A QP2/7/9 measurements. Dense estimates add A QP3/4/5/6; '
              'all comparisons keep exactly the same MAP QP2/7 target rates.','',
              f'Densifying A changes the sign of {linear_flips}/{len(sensitivity)} linear-bpp contrasts and '
              f'{log_flips}/{len(sensitivity)} log-bpp contrasts relative to the sparse measurements. '
              'After densification, linear-bpp and log-bpp signs agree in all 32 metric/target contrasts. '
              'At the four MAP QP2 target rates, MAP has worse interpolated comparisons on all four metrics. '
              'At the four MAP QP7 target rates, MAP has lower LPIPS in 4/4, lower DISTS in 3/4, '
              'lower FloLPIPS in 3/4, and higher PSNR in 4/4. These are local interpolated comparisons; '
              'the nearest measured points and their rate gaps remain the direct measurement evidence.','',
              table(['video','MAP QP','metric','sparse linear delta','dense linear delta','sparse log delta','dense log delta','linear sign unchanged','log sign unchanged'],
                    [[x['sequence'],x['MAP_QP'],x['metric'],f(x.get('sparse_linear_delta')),f(x['linear_delta']),
                      f(x.get('sparse_log_delta')),f(x['log_delta']),x.get('dense_vs_sparse_linear_sign_agrees','no bracket'),
                      x.get('dense_vs_sparse_log_sign_agrees','no bracket')] for x in sensitivity]),'',
              '## B. A versus IDENTITY: same native codec stream, different generator','',
              'IDENTITY uses H11 generator weights with H multiplier set to one. The native stream SHA is compared with A at each source/QP.','']
    subset=[x for x in matches if x['comparison']=='same_native_stream_generator']
    parts += [table(['video','QP','metric','A bpp','IDENTITY bpp','IDENTITY−A measured','native SHA equal'],
        [[x['sequence'],x['external_qp'],x['metric'],f(x['reference_actual_bpp'],6),f(x['method_actual_bpp'],6),
          f(x['measured_delta_method_minus_reference']),x['native_stream_sha_equal']] for x in subset]),'',
          'With identical native streams and exact codec states, IDENTITY has higher PSNR in 4/4 cases. '
          'LPIPS, DISTS and FloLPIPS are each lower in 2/4 cases. Generator weight changes therefore have '
          'metric-dependent effects even when compression is held fixed.','']
    for mode,section in [('MEAN','C'),('ROLL','D')]:
        parts += [f'## {section}. MAP versus {mode}','',
            'The same-QP table uses alpha 1; the near-rate table uses the candidate selected only by total bytes. '
            'Every near-rate row retains its actual rate gap and match status. '
            'All four alpha=1 cases already fall within the fixed 2% tolerance, so the near-rate table '
            'reuses those same full evaluations and no further alpha candidates were encoded.','']
        for kind in (f'same_QP_spatial_{mode.lower()}',f'near_rate_{mode}'):
            subset=[x for x in matches if x['comparison']==kind]
            parts += [f'### {kind}','',table(['video','QP','metric','alpha','MAP bpp','variant bpp','rate gap %','variant−MAP measured','rate status'],
                [[x['sequence'],x['external_qp'],x['metric'],x.get('selected_alpha') or '1.0',
                  f(x['reference_actual_bpp'],6),f(x['method_actual_bpp'],6),
                  f(x.get('rate_difference_percent') or 100*float(x['rate_difference_bpp'])/float(x['reference_actual_bpp']),2),
                  f(x['measured_delta_method_minus_reference']),x.get('rate_match_status') or 'same QP']
                 for x in subset]),'']
        if mode=='MEAN':
            parts += ['MEAN has lower LPIPS in 2/4, lower DISTS in 2/4, lower FloLPIPS in 4/4, and higher PSNR in 2/4 cases. '
                      'The fixed-weight replacement does not show a consistent advantage for MAP across these cases and metrics. '
                      'MEAN averages only H\'s added multiplier; q_base retains its own spatial/channel variation.','']
        else:
            parts += ['ROLL has lower LPIPS, lower DISTS, lower FloLPIPS and higher PSNR in 2/4 cases each. '
                      'Its multiplier value multiset is exactly preserved on every P frame, but coupling to q_base '
                      'and the subsequent decoded state can change. These mixed observations do not establish '
                      'a final contribution of spatial correspondence.','']
    summaries=rows('quant_summary.csv')
    diagnostics=[x for x in summaries if x['segment']=='all_P' and x['sequence'] in ('UVal1','Jockey')
                 and x['external_qp'] in ('2','7') and x['mode'] in ('MAP','MEAN','ROLL','IDENTITY')]
    parts += ['## Receiver diagnostics','',
        'Values below average each P frame\'s spatial statistics. Skip fractions refer to the symbols '
        'in each coded pass; clamp fractions count H-adjusted q entries below the v2 lower bound before clamping. '
        'At QP2, about 88% of MAP q entries trigger that lower bound, so effective rho can differ substantially '
        'from H\'s multiplier. The full per-frame quantiles are in `quant_diagnostics.csv`.','',
        table(['video','QP','mode','m mean','m spatial std','rho mean','rho spatial/channel std','q floor %','pass0 skip %','pass1 skip %'],
              [[x['sequence'],x['external_qp'],x['mode'],f(x['m_mean']),f(x['m_std']),f(x['rho_mean']),f(x['rho_std']),
                f(100*float(x['q_clamp_fraction']),2),f(100*float(x['pass0_skip_fraction']),2),
                f(100*float(x['pass1_skip_fraction']),2)] for x in diagnostics]),'',
        '## Fixed outputs and limitations','',
        '- `rate_match_search.csv` records every encoded candidate, target bytes, rate error, and selected flag. Selection used no quality metric.',
        '- `per_frame.csv`, `per_transition.csv`, and `frame_segment_summaries.csv` separate individual frames and the first 15 versus remaining 49 frames.',
        '- `figures/RD_*.png` shows measured A/MAP rate–quality points separately for all four metrics. `figures/fixed_previews/` fixes frames 15 and 47 and crop rows 200:456, columns 400:656. `figures/quant_maps/` shows full latent fields using fixed 0.5–2.0 scales for multiplier/rho and 0.5–16 for mean-channel q across all methods.',
        '- `task_manifest.csv` and `configs/` bind each task to source, weight, runtime and configuration SHAs. Original storage names are retained as aliases for resume compatibility. `quant_diagnostics.csv` records every P-frame receiver distribution and both skip fractions; `quant_summary.csv` aggregates their frame statistics. `resource_summary.csv` separates original evaluation cost from the additional receiver audits.',
        '- MEAN removes only the spatial variation of the added H multiplier; base codec q may still vary over space and channels.',
        '- Inference substitution can shift generator input distributions. These observations do not establish equal-training-budget superiority, convergence, statistical significance, or a proven causal mechanism.',
        '- This experiment covers four fixed 64-frame sources and two anchor QPs. Existing checkpoint training exposure follows v1/v2; no strict unseen-data claim is made. FID was not computed.','']
    (P/'REPORT.md').write_text('\n'.join(parts))


if __name__=='__main__':main()
