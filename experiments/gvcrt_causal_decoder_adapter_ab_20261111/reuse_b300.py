"""Reuse fully matched historical B300 points; never import H trained candidates."""
from common_s2 import *


def main():
    env()
    from cache_codec import canonical
    from evaluation import point_path
    assert load(ROOT/'audits/full_decoder_deployment.json')['status']=='PASS'
    # These five current-run points already verified every reconstructed RGB hash
    # against the historical B300 state and shared head implementation.
    measured=[]
    for p in (ROOT/'evaluation/development/B300/step_0000/points').glob('*.json'):
        a=load(p)
        if a.get('reuse_verified_every_RGB_frame'):measured.append(str(p))
    assert len(measured)>=5
    same_code=['model_context.py','convlora.py','fp32_island.py','temporal.py','quality.py']
    same_code += [str(p.relative_to(ROOT)) for p in (ROOT/'runtime_tree').rglob('*.py')]
    for f in same_code:assert sha(ROOT/f)==sha(H_SOURCE/f),(f,'B300_REUSE_CODE_MISMATCH')
    assert load(H_SOURCE/'protocol.json')['B300']['SHA256']==B300_SHA
    metric=load(ROOT/'audits/metric_identity_GPU2.json');records=[]
    for v in load(ROOT/'manifests/development_videos.json'):
        for q in (0,5,9):
            dest=point_path(v,q,'B300',0)
            if dest.exists():continue
            src=H_SOURCE/f'evaluation/development/B300/step_0000/points/{dest.name}'
            a=load(src);old=a['identity'];r=a['result'];can=canonical(v.get('source_row',v),q)
            assert a['status']=='PASS' and a['real_RANS'] and a['actual_I_frame_bits_included']
            assert old['checkpoint_SHA256']==B300_SHA and old['model']=='B300' and old['step']==0 and old['QP']==q
            assert old['source_frames']==v['source_frame_indices'] and old['source_SHA256']==v['source_sha256']
            assert old['source_RGB_hashes']==v['frame_rgb_sha256'] and a['metric_weights']==metric['weights']
            assert old['precision']=='inherited FP16 codec/Bridge/backbone; FP32 TCG and LoRA; RGB FP32-first'
            assert r['stream_SHA256']==can['SHA256'] and r['real_bytes']==can['bytes'] and r['bpp']==can['bpp']
            files={}
            for f,h in a['record_SHA256'].items():
                if '/frames/' not in f and '/transitions/' not in f:continue
                p=H_SOURCE/f;assert sha(p)==h
                folder='frames' if '/frames/' in f else 'transitions';out=dest.parent.parent/folder/dest.with_suffix('.csv').name
                out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out);files[str(out.relative_to(ROOT))]=sha(out)
            frpath=next(ROOT/f for f in files if '/frames/' in f);fr=list(csv.DictReader(frpath.open()))
            assert len(fr)==64 and [int(x['actual_QP']) for x in fr]==can['actual_qps']
            assert [x['source_RGB_SHA256'] for x in fr]==v['frame_rgb_sha256']
            assert sum(int(x['real_bits']) for x in fr)==can['bytes']*8
            identity=dict(model='B300',step=0,checkpoint_SHA256=B300_SHA,metric_id=metric['metric_id'],source_frames=v['source_frame_indices'],
                source_SHA256=v['source_sha256'],source_RGB_hashes=v['frame_rgb_sha256'],QP=q,stream_SHA256=can['SHA256'],
                real_bytes=can['bytes'],actual_qps=can['actual_qps'],precision=DEPLOYMENT)
            r=dict(r,actual_qps=json.dumps(can['actual_qps']))
            record=dict(status='PASS',identity=identity,result=r,record_SHA256=files,real_RANS=True,I_frame_bits_included=True,
                metric_weights=metric['weights'],metric_version='matched inherited metric code/weights; LPIPS0.1; MCResidual_RAFT_pixelcenter_v1',
                original_independently_loaded=True,model_provenance=dict(B300_SHA256=B300_SHA,full_recovery_chain_verified=True),
                reconstruction_uses_GT=False,history_bytes=0,
                reuse_source=dict(source=str(src),source_SHA256=sha(src),style='H_B300_entire_point_verified_identity'),
                reused_metrics='all requested scalar/frame/transition metrics',reuse_verified_every_RGB_frame=False,
                new_decode_performed=False,matching_current_run_RGB_checks=measured,
                reuse_identity_checks=['video/source frames/RGB','actual QPs','real stream bytes/SHA256/bpp','metric code/weight version','deployment precision','full B300 identity/shared module head'],
                physical_GPU=None,seconds=0.,evaluation_code_SHA256=sha(ROOT/'evaluation.py'))
            dump(dest,record);records.append(dict(point=str(dest),source=str(src),source_SHA256=sha(src)))
    dump(ROOT/'audits/B300_reuse.json',dict(status='PASS',new_historical_point_reuses=len(records),records=records,
        current_run_exact_RGB_points=len(measured),only_B300_reused=True,no_H_candidate_import=True,no_latent_RMS_records_imported=True,time=time.time()))
    print('MATCHED_B300_REUSED',len(records))

if __name__=='__main__':main()
