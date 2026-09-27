"""Generalize audited PCHIP analysis to 4, 6, or 10 measured rate points."""
import numpy as np
from audit_utils import V41,module
base=module('v51_original_rd_analysis',V41/'rd_analysis.py')

def points(rows,metric):
    ordered=sorted(rows,key=lambda r:float(r['kbps']))
    x=np.asarray([float(r['kbps']) for r in ordered]);y=np.asarray([float(r[metric]) for r in ordered])
    if len(x) not in (4,6,10) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('requires 4, 6, or 10 finite measured external QP points')
    if np.any(x<=0) or np.any(np.diff(np.log(x))<=1e-10):raise ValueError('nonpositive or duplicate/unstable bitrate')
    return np.log(x),y

base.points=points

def compare(a,c,metric,anchor,method):
    eq,bd=base.compare(a,c,metric,anchor,method)
    for rows in (a,c):
        ordered=sorted(rows,key=lambda r:int(r['external_qp']))
        if any(float(y['kbps'])<=float(x['kbps']) for x,y in zip(ordered,ordered[1:])):
            bd.update(status='invalid',BD_rate_percent='',reason='bitrate does not strictly increase with external QP; no points removed')
    return eq,bd
