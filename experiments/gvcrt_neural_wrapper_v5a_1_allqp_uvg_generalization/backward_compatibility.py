"""Hash-verified bit-exact reuse of all 128 V5-A final QP0-3 points."""
import math
import numpy as np
from audit_utils import *
from stream_audit import structure

def main():
    config=load('config.json');checks=[]
    assert sha(ROOT/'test_manifest.json')==sha(V5/'test_manifest.json')==config['ulong_manifest_sha256']
    for path,digest in config['source_hashes'].items():assert sha(path)==digest,path
    fid=module('v51_fid_compat',REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py')
    old_points=read(V5/'final_rd_points.csv');old_summary=read(V5/'final_qp_summary.csv')
    references={}
    for method in METHODS:
        for q in range(4):
            pooled_real=[];pooled_recon=[];rows=[]
            for index in range(8):
                source=V5/f'parts/final/{method}/video_{index}_qp{q}.json';r=load(source)
                expected=next(v for v in old_points if v['method']==method and int(v['video_index'])==index and int(v['qp'])==q)
                assert r['checkpoint']==config['checkpoints'][method]['path'] and r['checkpoint_sha256']==config['checkpoints'][method]['sha256']
                assert r['manifest_sha256']==config['ulong_manifest_sha256'] and r['metric_audit_sha256']==config['metric_audit_sha256']
                assert r['num_frames']==64 and r['decode_status']=='PASS'
                assert all(truth(r[k]) for k in ('independent_decode_pass','metric_decode_pass','position_decode_pass','state_sync_pass'))
                for k in ('bitstream','feature','frame_metrics'):assert sha(r[k+'_path'])==r[k+'_sha256']
                assert int(r['real_bytes'])==int(r['bytes_consumed'])==Path(r['bitstream_path']).stat().st_size==int(expected['real_bytes'])
                assert r['reconstruction_sha256']==expected['reconstruction_sha256']
                for k in NUMERIC:assert math.isfinite(float(r[k])) and abs(float(r[k])-float(expected[k]))<1e-12
                header=structure(Path(r['bitstream_path']),q,64);frame=read(r['frame_metrics_path'])
                assert len(frame)==64
                for a,b in zip(frame,header):
                    assert int(a['frame'])==b['frame'] and int(a['actual_qp'])==b['actual_qp'] and int(a['real_bits'])==b['real_bits']
                    a.update(b,dataset='Fresh_U_Long',method=method,video_index=index)
                transitions=read(Path(r['feature_path']).with_suffix('.transitions.csv'))
                assert [(int(t['from_frame']),int(t['to_frame'])) for t in transitions]==[(i,i+1) for i in range(63)]
                assert abs(np.mean([float(t['FloLPIPS']) for t in transitions])-r['FloLPIPS'])<1e-12
                with np.load(r['feature_path']) as f:
                    real=f['real'];recon=f['reconstruction']
                    assert real.shape==recon.shape==(64,2048) and np.isfinite(real).all() and np.isfinite(recon).all()
                    if index in references:assert np.allclose(real,references[index],rtol=1e-5,atol=1e-6)
                    else:references[index]=real.copy()
                    pooled_real.append(real);pooled_recon.append(recon)
                dest=point_path('ulong',method,index,q);framepath=dest.with_suffix('.frames.csv');write(framepath,frame)
                r.update(dataset='Fresh_U_Long',external_qp=q,actual_i_qp=q,actual_p_qp=sorted({v['actual_qp'] for v in header[1:]}),
                    source_record=str(source),source_record_sha256=sha(source),reuse_mode='bit-exact cached artifacts; no new encoding or decoding',
                    source_frame_metrics_path=r['frame_metrics_path'],frame_metrics_path=str(framepath),frame_metrics_sha256=sha(framepath),
                    evaluation_only=True,physical_gpu=None)
                dump(dest,r);rows.append(r)
                checks.append(dict(method=method,external_qp=q,actual_i_qp=q,actual_p_qp=r['actual_p_qp'],video_index=index,
                    real_bytes_identical=True,decoded_reconstruction_hash_identical=True,
                    LPIPS_identical=True,DISTS_identical=True,FloLPIPS_identical=True,FID_features_hash_verified=True,
                    cached_independent_decode_pass=True,source_record_sha256=sha(source)))
            summary=next(v for v in old_summary if v['method']==method and int(v['qp'])==q)
            for k in NUMERIC:assert abs(np.mean([float(v[k]) for v in rows])-float(summary[k]))<2e-6,(method,q,k)
            value=fid.low_rank_fid(np.concatenate(pooled_real),np.concatenate(pooled_recon))
            assert abs(value-float(summary['FID']))<2e-6,(method,q,'FID',value,summary['FID'])
            assert int(summary['FID_num_samples'])==512
    dump('qp0_3_backward_compatibility_audit.json',dict(status='PASS',points=len(checks),all_128_points_compatible=True,
        mode='verified reuse of original bitstreams, cached independent decode hashes, scalar metrics and FID features',
        independent_decode_rerun=False,reconstruction_hash_evidence='original independent decoder and separate metric decoder recorded matching hashes; cache referenced without alteration',
        metric_tolerance=2e-6,pooled_FID_recomputed=True,source_final_summary_sha256=sha(V5/'final_qp_summary.csv'),
        source_final_points_sha256=sha(V5/'final_rd_points.csv'),checks=checks))
    print('QP0-3 BACKWARD COMPATIBILITY PASS 128/128',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump('qp0_3_backward_compatibility_audit.json',dict(status='FAIL',error=repr(exc),stop_all_new_evaluation=True));raise
