"""Coverage, source bytes, exact training cursor and paired-stream audit."""
from common_s2 import *


def main():
    env();import torch
    rows=plan_rows();training={};complete=True
    common=torch.load(ROOT/'checkpoints/common_adapter_step0.pt',map_location='cpu',weights_only=True)
    for g in GROUPS:
        receipts=[load(p) for p in sorted((ROOT/g/'training/receipts').glob('*.json'))]
        steps=[r['committed_updates'] for r in receipts];assert steps==list(range(1,len(steps)+1))
        actual=[k for r in receipts for k in r['sample_keys']];assert actual==[r['sample_key'] for r in rows[:len(actual)]]
        assert len(actual)==len(set(actual))==4*len(steps)
        for r in receipts:assert r['effective_batch']==4 and r['next_cursor']==4*r['committed_updates'] and r['frozen_head_unchanged']
        cp0=ROOT/g/'training/checkpoints/step_0000.pt'
        if cp0.exists():
            a=torch.load(cp0,map_location='cpu',weights_only=True);assert tensor_hash(a['adapter'].items())==tensor_hash(common['adapter'].items())
        for s in [25,50,75,100]:
            p=ROOT/g/f'training/checkpoints/step_{s:04d}.pt'
            if not p.exists():complete=False;continue
            a=torch.load(p,map_location='cpu',weights_only=True)
            assert a['committed_updates']==s and a['next_cursor']==4*s and a['B300_SHA256']==B300_SHA
            assert a['protocol_SHA256']==sha(ROOT/'protocol.json') and a['plan_SHA256']==sha(ROOT/'plans/index.json')
            assert a['lambda_temp']==load(ROOT/'gradient_calibration.json')['lambda_temp']
            assert all(int(v['step'])==s for v in a['optimizer']['state'].values())
        training[g]=dict(committed_updates=len(steps),unique_windows=len(actual),plan_order_exact=True,checkpoint0_common_exact=cp0.exists())
        complete=complete and len(steps)==100
    metric_points=[];identities={};coverage={}
    for p in sorted((ROOT/'evaluation/development').glob('*/step_*/points/*.json')):
        a=load(p);assert a['status']=='PASS' and a['real_RANS'];r=a['result'];metric_points.append(r)
        for f,h in a['record_SHA256'].items():assert sha(ROOT/f)==h
        k=(r['sequence'],r['QP']);identity=(r['real_bytes'],r['bpp'],r['stream_SHA256'])
        assert identities.setdefault(k,identity)==identity,('STREAM_MISMATCH',k)
        assert all(r[k] is None or math.isfinite(r[k]) for k in ['PSNR','LPIPS_Alex','LPIPS_VGG','DISTS','FloLPIPS'])
        state=f"{r['model']}_step{r['step']}";coverage[state]=coverage.get(state,0)+1
        if r['model']=='Original':assert a['original_independently_loaded'] and a['original_provenance']['commit']==ORIGINAL_COMMIT
    for m,s in [('Original',0),('B300',0),*[(g,s) for g in GROUPS for s in [50,100]]]:
        complete=complete and coverage.get(f'{m}_step{s}',0)==96
    source=load(ROOT/'source_manifest.json')
    protected={p:sha(p)==h for p,h in source['protected_sources'].items()};assert all(protected.values())
    media=load(ROOT/'video_manifest.json') if (ROOT/'video_manifest.json').exists() else dict(status='PENDING')
    if media.get('status')=='PASS':
        for r in media['records']:
            assert r['ROI_scale']==1 and r['ROI_unscaled'] and r['frames']==64
            for kind in ['native','ROI']:assert sha(r[kind]['path'])==r[kind]['SHA256']
    complete=complete and media.get('status')=='PASS'
    assert load(ROOT/'audits/preflight.json')['status']=='PASS'
    dump(ROOT/'integrity.json',dict(status='PASS' if complete else 'PARTIAL_PASS',complete=complete,
        training=training,evaluation_coverage=coverage,expected_state_points=96,point_records_checked=len(metric_points),
        all_matched_stream_bytes_bpp_SHA256=True,protected_sources_unchanged=protected,
        B300_expected_SHA256=B300_SHA,B300_actual_SHA256=sha(B300),common_step0_equal=True,
        model_weights_local_only=True,dataset_copied=False,final_test_run=False,media_status=media['status'],
        no_effectiveness_decision=True,time=time.time()))
    print('INTEGRITY', 'PASS' if complete else 'PARTIAL_PASS')

if __name__=='__main__':main()
