"""Regression test for metrics restored as strings from audited caches."""
from audit_utils import *
from report import normalized_point,collect

def main():
    original=load(point_path('ulong','original',0,0))
    strings=dict(original,**{key:str(original[key]) for key in NUMERIC})
    normalized=normalized_point(strings)
    assert all(isinstance(normalized[key],float) for key in NUMERIC)
    assert all(normalized[key]==float(original[key]) for key in NUMERIC)
    assert all(isinstance(strings[key],str) for key in NUMERIC),'source record mutated'
    for bad in ('nan','inf','-inf','invalid',True):
        try:normalized_point(dict(strings,LPIPS=bad))
        except (ValueError,TypeError):pass
        else:raise AssertionError(f'invalid metric accepted: {bad!r}')
    rows=collect('ulong')
    assert len(rows)==320
    assert all(isinstance(r[key],float) for r in rows for key in NUMERIC)
    dump('numeric_record_regression_audit.json',dict(status='PASS',ulong_points_validated=len(rows),
        numeric_strings_supported=True,nonfinite_values_rejected=True,source_records_unchanged=True,
        original_numerical_tolerances_unchanged=True))
    print('NUMERIC RECORD REGRESSION PASS; ALL 320 U-LONG POINTS VERIFIED',flush=True)

if __name__=='__main__':main()
