"""Verify the sole two-frame intervention and write objective comparisons."""
from common44 import *
import math

def main():
    task, = load(ROOT/'intervention_plan.json')
    assert validate_result(task)
    variant = load(result_path(task))
    baseline = source_baseline()
    single8 = load(SOURCE43/'tasks/hevc_b/ParkScene/t08_qminus2/result.json')
    single32 = load(SOURCE43/'tasks/hevc_b/ParkScene/t32_qminus2/result.json')
    for entry in (single8, single32):
        assert entry['status']=='PASS'
        for a in entry['artifacts']:
            p=Path(a['path']);assert p.stat().st_size==a['bytes'] and sha(p)==a['sha256']
    b,v=baseline['result'],variant['result']
    assert task['actual_qps'][8]==task['actual_qps'][32]==3
    assert all(task['actual_qps'][i]==baseline['task']['actual_qps'][i] for i in range(64) if i not in (8,32))
    assert b['total_bits']==977264 and b['LPIPS']==0.2153701321221888
    assert v['bytes_consumed']==v['total_bytes']==Path(variant['artifacts'][0]['path']).stat().st_size
    assert v['total_bits']==8*v['total_bytes'] and v['actual_bpp']==v['total_bits']/(64*1080*1920)
    for name,x,expected in [('baseline',baseline,b['stream_sha256']),('variant',variant,v['stream_sha256'])]:
        assert sha(x['artifacts'][0]['path'])==expected
    bp=load(SOURCE43/'tasks/hevc_b/ParkScene/baseline/proofs.json')
    vp=load(ROOT/'tasks/hevc_b/ParkScene/t08_t32_qminus2/proofs.json')
    assert len(bp['encoder'])==len(bp['decoder'])==len(vp['encoder'])==len(vp['decoder'])==64
    assert all(bp[k][:8]==vp[k][:8] for k in ('encoder','decoder','packets'))
    assert all(vp['packets'][i]['actual_qp']==task['actual_qps'][i] for i in range(64))
    assert sum(x['packet_bytes'] for x in vp['packets'])==v['total_bytes']
    assert variant['audits']['encoder_decoder_state_sync']=='PASS' and variant['audits']['prefix_equality']=='PASS'

    br=read(SOURCE43/'tasks/hevc_b/ParkScene/baseline/frames.csv')
    vr=read(ROOT/'tasks/hevc_b/ParkScene/t08_t32_qminus2/frames.csv')
    bt=read(SOURCE43/'tasks/hevc_b/ParkScene/baseline/transitions.csv')
    vt=read(ROOT/'tasks/hevc_b/ParkScene/t08_t32_qminus2/transitions.csv')
    assert len(br)==len(vr)==64 and len(bt)==len(vt)==63
    assert all(br[i]['packet_sha256']==vr[i]['packet_sha256'] for i in range(8))
    assert all(float(br[i][m])==float(vr[i][m]) for i in range(8) for m in ('LPIPS','DISTS','L1','pixel_SSE','real_bits'))
    fr=[]
    for i,(a,c) in enumerate(zip(br,vr)):
        assert int(a['frame'])==int(c['frame'])==i
        row=dict(frame=i,source_frame=int(a['source_frame']),baseline_actual_qp=int(a['actual_qp']),variant_actual_qp=int(c['actual_qp']))
        for m in ('real_bits','LPIPS','DISTS','L1','PSNR','pixel_SSE'):
            row[f'baseline_{m}']=float(a[m]);row[f'variant_{m}']=float(c[m]);row[f'delta_{m}']=float(c[m])-float(a[m])
        fr.append(row)
    write(ROOT/'frame_differences.csv',fr)
    tr=[]
    for i,(a,c) in enumerate(zip(bt,vt)):
        assert int(a['from_frame'])==int(c['from_frame'])==i and int(a['to_frame'])==int(c['to_frame'])==i+1
        tr.append(dict(from_frame=i,to_frame=i+1,baseline_FloLPIPS=float(a['FloLPIPS']),variant_FloLPIPS=float(c['FloLPIPS']),delta_FloLPIPS=float(c['FloLPIPS'])-float(a['FloLPIPS'])))
    write(ROOT/'transition_differences.csv',tr)
    assert math.isclose(v['PSNR'],-10*math.log10(max(sum(float(r['pixel_SSE']) for r in vr)/(64*1080*1920*3),1e-15)),rel_tol=0,abs_tol=1e-12)
    assert all(math.isclose(v[m],sum(float(r[m]) for r in vr)/64,rel_tol=0,abs_tol=1e-12) for m in ('LPIPS','DISTS','L1'))
    assert math.isclose(v['FloLPIPS'],sum(float(r['FloLPIPS']) for r in vt)/63,rel_tol=0,abs_tol=1e-12)
    expected_bits=sum(x['result']['total_bits']-b['total_bits'] for x in (single8,single32))
    expected_lpips=sum(x['result']['LPIPS']-b['LPIPS'] for x in (single8,single32))
    assert expected_bits==-4760
    observed_bits=v['total_bits']-b['total_bits'];observed_lpips=v['LPIPS']-b['LPIPS']
    result=dict(status='PASS',dataset='hevc_b',sequence='ParkScene',changed_frames=[8,32],baseline_actual_qps=[5,5],variant_actual_qps=[3,3],
                baseline=b,variant=v,predicted_additive_delta=dict(total_bits=expected_bits,LPIPS=expected_lpips),
                measured_delta={m:v[m]-b[m] for m in ('total_bits','actual_bpp','LPIPS','DISTS','L1','PSNR','FloLPIPS')},
                measured_minus_prediction=dict(total_bits=observed_bits-expected_bits,LPIPS=observed_lpips-expected_lpips),
                success_criterion_met=(v['total_bits']<=977264 and v['LPIPS']<0.2153701321221888),
                offline_oracle_selection=True,optimizer_updates=0,new_variants_encoded=1)
    dump(ROOT/'result.json',result)
    sources=[CP,SOURCE43/'tasks/hevc_b/ParkScene/baseline/result.json',SOURCE43/'tasks/hevc_b/ParkScene/baseline/proofs.json',
             SOURCE43/'tasks/hevc_b/ParkScene/t08_qminus2/result.json',SOURCE43/'tasks/hevc_b/ParkScene/t32_qminus2/result.json',
             Path(variant['artifacts'][0]['path'])]
    dump(ROOT/'artifact_index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sources])
    deps=['evaluate44.py','common44.py','adapter35.py','io32.py','io35.py','training35.py','protocol.json','intervention_plan.json',
          'data_manifest.json','baseline_manifest.json','source_manifest.json','evaluation_config.json','model_manifest.json']
    dump(ROOT/'source_hashes.json',{n:sha(ROOT/n) for n in deps})
    checks=dict(checkpoint_sha256=sha(CP)==CP_SHA,one_fixed_variant=len(load(ROOT/'intervention_plan.json'))==1,
                zero_updates=variant['audits']['optimizer_updates']==0,all_64_frames=len(fr)==64,all_63_transitions=len(tr)==63,
                independent_decode=variant['audits']['encoder_decoder_state_sync']=='PASS',stream_qps_from_headers=variant['audits']['qp_in_stream']=='PASS',
                complete_stream_accounting=variant['audits']['bit_accounting']=='PASS',prefix_0_to_7_bitwise=True,
                native_i_and_causal_reference=variant['audits']['no_extra_reference_reset'] is True,
                model_frozen=variant['audits']['freeze']=='PASS',source_baseline_reused=True,frame_8_32_qp_3=True,
                total_bits_equals_bytes_times_8=True,whole_segment_PSNR_from_RGB_SSE=True)
    assert all(checks.values())
    dump(ROOT/'final_integrity.json',dict(status='PASS',checks=checks,success_criterion_met=result['success_criterion_met'],
                                           variant_stream_sha256=v['stream_sha256'],source_result_sha256=sha(result_path(task))))
    textfile(ROOT/'PROTOCOL.md','# v6.44 fixed two-frame QP interaction\n\n'+
             'Only ParkScene, frames 8 and 32, actual QP 5→3; all other QPs copied from the v6.43 baseline. '
             'G_basic, 64 source frames, native I, complete causal real-RANS encode and independent decode. '
             'Zero training updates. The candidate was selected offline from earlier video measurements.\n')
    textfile(ROOT/'README.md','# v6.44 ParkScene two-frame QP check\n\nSee `protocol.json`, `result.json`, `final_integrity.json`, `frame_differences.csv`, `transition_differences.csv`, and `REPORT.md`.\n')
    def fmt(m):return f"{b[m]:.10f} | {v[m]:.10f} | {v[m]-b[m]:+.10f}"
    report=(f'# v6.44 measured result\n\n'+
      f'Fixed offline oracle candidate: ParkScene frame 8 and 32 actual QP 5→3. One new full 64-frame stream, zero optimizer updates.\n\n'+
      '| Metric | v6.43 baseline | two-frame measured | variant − baseline |\n|---|---:|---:|---:|\n'+
      f'| Complete stream bits | {b["total_bits"]} | {v["total_bits"]} | {observed_bits:+d} |\n'+
      ''.join(f'| {m} | {fmt(m)} |\n' for m in ('actual_bpp','LPIPS','DISTS','L1','PSNR','FloLPIPS'))+
      f'\nNon-interaction additive prediction from the two v6.43 single-frame runs: {expected_bits:+d} bit, {expected_lpips:+.10f} LPIPS. '
      f'Measured minus prediction: {observed_bits-expected_bits:+d} bit, {observed_lpips-expected_lpips:+.10f} LPIPS.\n\n'+
      f'**Prespecified criterion: {"PASS" if result["success_criterion_met"] else "FAIL"}.** '
      'The measured stream has no more than 977264 bit and mean LPIPS is below 0.2153701321221888.\n\n'+
      'The variant was encoded from native I0 with its own evolving references; all 64 frames were independently decoded from its persisted stream. '
      'Frames 0–7 match the baseline packets, reconstructions and reference states exactly. All packets, QP fields, bytes consumed, frozen models and 63 transitions passed checks. '
      'PSNR uses summed whole-segment RGB SSE. The source was chosen by an offline oracle; this single sequence does not establish a deployable rule or generalization.\n')
    textfile(ROOT/'REPORT.md',report)
    dump(ROOT/'run_status.json',dict(status='COMPLETE',calculation='PASS',publication='PENDING',optimizer_updates=0,new_variants_encoded=1))
    print('V644_PASS',v['total_bits'],v['LPIPS'],flush=True)

if __name__=='__main__':main()
