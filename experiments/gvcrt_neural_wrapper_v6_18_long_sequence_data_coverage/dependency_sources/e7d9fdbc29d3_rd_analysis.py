"""Shared validation/final RD implementation; lower distortion is better."""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator

METRICS = ('LPIPS', 'DISTS', 'FloLPIPS')

def points(rows, metric):
    ordered = sorted(rows, key=lambda r: float(r['kbps']))
    x = np.asarray([float(r['kbps']) for r in ordered])
    y = np.asarray([float(r[metric]) for r in ordered])
    if len(x) != 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('requires four finite QP points')
    if np.any(x <= 0) or np.any(np.diff(np.log(x)) <= 1e-10):
        raise ValueError('nonpositive or duplicate/unstable bitrate')
    return np.log(x), y

def compare(anchor, candidate, metric, anchor_name='original', method='candidate'):
    base = dict(anchor=anchor_name, method=method, metric=metric,
                rate_axis='natural log(kbps)', interpolation='PCHIP, extrapolate=False',
                common_rate_min_kbps='', common_rate_max_kbps='')
    eq = dict(base, status='invalid', reason='', grid_points=100,
              grid_spacing='uniform natural log(kbps), inclusive endpoints',
              mean_equal_rate_delta='', fraction_of_common_rate_range_better='',
              anchor_monotone='', candidate_monotone='')
    bd = dict(base, status='invalid', reason='', BD_rate_percent='',
              common_metric_min='', common_metric_max='',
              integration='exact integral of piecewise cubic PCHIP over common metric interval',
              formula='100 * expm1(mean(log(rate_candidate)-log(rate_anchor)))')
    try:
        ax, ay = points(anchor, metric); cx, cy = points(candidate, metric)
    except ValueError as exc:
        eq['reason'] = bd['reason'] = str(exc)
        return eq, bd
    lo, hi = max(ax.min(), cx.min()), min(ax.max(), cx.max())
    for row in (eq, bd):
        row['common_rate_min_kbps'] = float(np.exp(lo))
        row['common_rate_max_kbps'] = float(np.exp(hi))
    eq['anchor_monotone'] = bool(np.all(np.diff(ay) < 0))
    eq['candidate_monotone'] = bool(np.all(np.diff(cy) < 0))
    if hi - lo <= 1e-10:
        eq['reason'] = 'empty or numerically negligible common rate interval'
    else:
        grid = np.linspace(lo, hi, 100)
        delta = PchipInterpolator(cx, cy, extrapolate=False)(grid) - PchipInterpolator(ax, ay, extrapolate=False)(grid)
        if np.isfinite(delta).all():
            eq.update(status='valid', mean_equal_rate_delta=float(delta.mean()),
                      fraction_of_common_rate_range_better=float((delta < 0).mean()),
                      nonmonotone_policy='preserve measured local shape; no sorting of metric values or smoothing')
        else: eq['reason'] = 'nonfinite interpolated values'
    if not eq['anchor_monotone'] or not eq['candidate_monotone']:
        bd['reason'] = 'metric must strictly decrease as bitrate increases; no monotonicization performed'
        return eq, bd
    low, high = max(ay.min(), cy.min()), min(ay.max(), cy.max())
    bd.update(common_metric_min=float(low), common_metric_max=float(high))
    if high-low <= max(1e-12, 1e-8*max(abs(low), abs(high))):
        bd['reason'] = 'empty or numerically negligible common metric interval'
        return eq, bd
    if min(np.abs(np.diff(ay)).min(), np.abs(np.diff(cy)).min()) <= 1e-12:
        bd['reason'] = 'duplicate or numerically unstable metric samples'
        return eq, bd
    ia = PchipInterpolator(ay[::-1], ax[::-1], extrapolate=False)
    ic = PchipInterpolator(cy[::-1], cx[::-1], extrapolate=False)
    delta_log = float((ic.integrate(low, high)-ia.integrate(low, high))/(high-low))
    try: value = math.expm1(delta_log)*100
    except OverflowError: value = float('inf')
    if not math.isfinite(value): bd['reason'] = 'nonfinite BD-rate'
    else: bd.update(status='valid', BD_rate_percent=value)
    return eq, bd

def gate(equal_rows, bd_rows):
    reasons = []
    for metric in ('LPIPS', 'DISTS'):
        eq = next(r for r in equal_rows if r['metric'] == metric)
        bd = next(r for r in bd_rows if r['metric'] == metric)
        if eq['status'] != 'valid' or float(eq['mean_equal_rate_delta']) >= 0:
            reasons.append(metric + ' equal-rate gate failed')
        if bd['status'] == 'valid' and float(bd['BD_rate_percent']) >= 0:
            reasons.append(metric + ' valid BD-rate gate failed')
    return not reasons, '; '.join(reasons)
