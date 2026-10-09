import argparse
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
from experiment_utils import ROOT, REPO, V4, sha, dump, read, write
from rd_analysis import METRICS, compare, gate

VALIDATION_METHODS = ('original', 'v3', 'v4_10000', 'v4_15000', 'v4_20000')
FINAL_METHODS = ('original', 'v3', 'v4_10000', 'selected')
NUMERIC = ('kbps', 'bpp', 'LPIPS', 'DISTS', 'FloLPIPS', 'PSNR', 'SSIM', 'MS_SSIM')

def truth(value): return str(value).lower() == 'true'
def mean(rows, key): return float(np.mean([float(r[key]) for r in rows]))
def load_json(path): return json.loads((ROOT/path).read_text())

def collect(split):
    methods = VALIDATION_METHODS if split == 'validation' else FINAL_METHODS
    n, frames = (6, 32) if split == 'validation' else (8, 64)
    audit_hash = sha(ROOT/'metric_implementation_audit.json')
    records = []; checkpoint_hashes = {}
    for method in methods:
        for video in range(n):
            for qp in range(4):
                row = load_json(f'parts/{split}/{method}/video_{video}_qp{qp}.json')
                assert (row['split'], row['method'], int(row['video_index']), int(row['qp'])) == (split, method, video, qp)
                assert row['num_frames'] == frames and row['num_transitions'] == frames-1
                assert row['metric_audit_sha256'] == audit_hash
                assert row['decode_status'] == 'PASS' and truth(row['state_sync_pass'])
                assert truth(row['independent_decode_pass']) and truth(row['metric_decode_pass'])
                assert row['compression_hash_before'] == row['compression_hash_after'] == load_json('resume_audit.json')['compression_hash']
                assert sha(row['bitstream_path']) == row['bitstream_sha256']
                assert Path(row['bitstream_path']).stat().st_size == int(row['real_bytes']) == int(row['bytes_consumed'])
                assert sha(row['feature_path']) == row['feature_sha256']
                if row['checkpoint']:
                    if row['checkpoint'] not in checkpoint_hashes: checkpoint_hashes[row['checkpoint']] = sha(row['checkpoint'])
                    assert row['checkpoint_sha256'] == checkpoint_hashes[row['checkpoint']]
                for key in NUMERIC:
                    row[key] = float(row[key]); assert math.isfinite(row[key])
                assert row['kbps'] > 0 and row['bpp'] > 0
                transitions = read(Path(row['feature_path']).with_suffix('.transitions.csv'))
                assert len(transitions) == frames-1
                assert [(int(r['from_frame']), int(r['to_frame'])) for r in transitions] == [(i, i+1) for i in range(frames-1)]
                assert all(math.isfinite(float(r['FloLPIPS'])) for r in transitions)
                assert abs(mean(transitions, 'FloLPIPS')-row['FloLPIPS']) < 1e-12
                records.append(row)
    assert len(records) == len(methods)*n*4
    return records

def summarize(rows):
    result = []
    for method in dict.fromkeys(r['method'] for r in rows):
        for qp in range(4):
            subset = [r for r in rows if r['method'] == method and int(r['qp']) == qp]
            result.append(dict(method=method, qp=qp, num_videos=len(subset), **{k:mean(subset,k) for k in NUMERIC}))
    return result

def metrics(rows, split):
    spec = importlib.util.spec_from_file_location('v41_fid', REPO/'expericent_generation_input/expericent_perceptual_temporal_refiner_v6/src/fid_metric.py')
    fid_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(fid_module)
    fids = []; flows = []; reference_features = {}
    for method in dict.fromkeys(r['method'] for r in rows):
        own = [r for r in rows if r['method'] == method]
        for r in own:
            flows.append(dict(method=method, checkpoint=r['checkpoint'], scope='video', video_index=r['video_index'],
                              qp=r['qp'], num_transitions=r['num_transitions'], FloLPIPS=r['FloLPIPS']))
        for qp in range(4):
            subset = [r for r in own if int(r['qp']) == qp]
            features = []
            for r in subset:
                with np.load(r['feature_path']) as values:
                    real, recon = values['real'], values['reconstruction']
                    assert real.shape == recon.shape == (r['num_frames'], 2048)
                    assert np.isfinite(real).all() and np.isfinite(recon).all()
                    key = int(r['video_index'])
                    if key in reference_features:
                        assert np.allclose(real, reference_features[key], rtol=1e-5, atol=1e-6), 'source FID features differ across QPs/methods'
                    else: reference_features[key] = real.copy()
                    features.append((real, recon))
            real = np.concatenate([x[0] for x in features]); recon = np.concatenate([x[1] for x in features])
            count = 192 if split == 'validation' else 512
            assert real.shape == recon.shape == (count, 2048)
            value = fid_module.low_rank_fid(real, recon); assert math.isfinite(value)
            fids.append(dict(method=method, checkpoint=subset[0]['checkpoint'], qp=qp,
                             num_real_samples=count, num_reconstruction_samples=count, FID=value,
                             mean_real_kbps=mean(subset,'kbps'), pooling='all videos in fixed manifest, single QP'))
            flows.append(dict(method=method, checkpoint=subset[0]['checkpoint'], scope='pooled_qp', video_index='',
                              qp=qp, num_transitions=sum(r['num_transitions'] for r in subset), FloLPIPS=mean(subset,'FloLPIPS')))
        flows.append(dict(method=method, checkpoint=own[0]['checkpoint'], scope='all_qp', video_index='',
                          qp='all', num_transitions=sum(r['num_transitions'] for r in own), FloLPIPS=mean(own,'FloLPIPS')))
    write(f'fid_{split}.csv', fids); write(f'flolpips_{split}.csv', flows)
    return fids

def comparisons(summary, anchor='original'):
    anchors = [r for r in summary if r['method'] == anchor]
    equal, bd = [], []
    for method in dict.fromkeys(r['method'] for r in summary if r['method'] != anchor):
        candidates = [r for r in summary if r['method'] == method]
        for metric in METRICS:
            eq, rate = compare(anchors,candidates,metric,anchor,method)
            equal.append(eq); bd.append(rate)
    return equal, bd

def same_qp(rows, anchor='original'):
    result = []
    for method in dict.fromkeys(r['method'] for r in rows if r['method'] != anchor):
        for qp in [0,1,2,3,'all']:
            a = [r for r in rows if r['method'] == anchor and (qp == 'all' or int(r['qp']) == qp)]
            c = [r for r in rows if r['method'] == method and (qp == 'all' or int(r['qp']) == qp)]
            result.append(dict(anchor=anchor,method=method,qp=qp,mean_real_kbps=mean(c,'kbps'),
                               mean_rate_change_percent=100*(mean(c,'kbps')/mean(a,'kbps')-1),
                               aggregation='ratio of arithmetic mean real kbps across matched video/QP points',
                               **{k+'_change':mean(c,k)-mean(a,k) for k in METRICS}))
    return result

def validation():
    rows = collect('validation'); write('checkpoint_validation.csv',rows)
    summary = summarize(rows); write('validation_qp_summary.csv',summary)
    fids = metrics(rows,'validation')
    eq, bd = comparisons(summary)
    write('equal_rate_perceptual_summary.csv',eq); write('perceptual_bd_rate.csv',bd)
    bookkeeping = same_qp(rows); write('same_qp_summary.csv',bookkeeping)
    convergence, choices = [], []
    previous = None
    for step in (10000,15000,20000):
        method = f'v4_{step}'; own = [r for r in rows if r['method'] == method]
        delta = next(r for r in bookkeeping if r['method'] == method and r['qp'] == 'all')
        record = dict(step=step, mean_real_kbps=mean(own,'kbps'),
                      **{'mean_'+m:mean(own,m) for m in METRICS},
                      same_qp_mean_rate_change_percent=delta['mean_rate_change_percent'],
                      **{'same_qp_'+m+'_change':delta[m+'_change'] for m in METRICS})
        for m in METRICS:
            er = next(r for r in eq if r['method']==method and r['metric']==m)
            br = next(r for r in bd if r['method']==method and r['metric']==m)
            record.update({m+'_BD_rate':br['BD_rate_percent'], m+'_BD_status':br['status'],
                           m+'_BD_reason':br['reason'], 'mean_equal_rate_delta_'+m:er['mean_equal_rate_delta'],
                           m+'_equal_rate_status':er['status'],m+'_better_fraction':er['fraction_of_common_rate_range_better']})
        record['previous_step'] = previous['step'] if previous else ''
        record['rate_change_from_previous_percent'] = 100*(record['mean_real_kbps']/previous['mean_real_kbps']-1) if previous else ''
        for m in METRICS: record[m+'_change_from_previous'] = record['mean_'+m]-previous['mean_'+m] if previous else ''
        convergence.append(record); previous=record
        eligible, reason = gate([r for r in eq if r['method']==method],[r for r in bd if r['method']==method])
        choices.append(dict(step=step,method=method,eligible=eligible,reason=reason,
                            real_bitrate_reduction_percent=-delta['mean_rate_change_percent'],
                            checkpoint=str(ROOT/'checkpoints/beta_high'/f'step_{step}.pt'),
                            used_final_test=False,FloLPIPS_gate=False,FID_gate=False,
                            **{k:v for k,v in record.items() if 'BD_' in k or 'equal_rate' in k}))
    write('convergence_summary.csv',convergence); write('checkpoint_selection.csv',choices)
    eligible = [r for r in choices if r['eligible']]
    selected = dict(status='NO_ELIGIBLE_CANDIDATE',used_final_test=False,step=None,checkpoint=None,
                    validation_sha256=sha(ROOT/'checkpoint_validation.csv'),
                    selection_csv_sha256=sha(ROOT/'checkpoint_selection.csv'),
                    rule='LPIPS/DISTS equal-rate deltas < 0; each valid BD-rate < 0; rank by real bitrate reduction',
                    tie_break='earliest step', candidates=[r['step'] for r in choices])
    if eligible:
        best = max(eligible,key=lambda r:(r['real_bitrate_reduction_percent'],-r['step']))
        selected.update(status='PASS',step=best['step'],checkpoint=best['checkpoint'],
                        checkpoint_sha256=sha(best['checkpoint']),real_bitrate_reduction_percent=best['real_bitrate_reduction_percent'])
    dump('selected_checkpoint.json',selected)
    dump('parts/validation_report_done.json',dict(status='PASS',rows=len(rows),fid_rows=len(fids),selection_status=selected['status']))
    print('VALIDATION REPORT',selected['status'],selected['step'],flush=True)

def final():
    selection = load_json('selected_checkpoint.json'); assert selection['status']=='PASS' and not selection['used_final_test']
    assert sha(ROOT/'checkpoint_validation.csv')==selection['validation_sha256']
    rows = collect('final'); write('final_rd_points.csv',rows)
    summary = summarize(rows); write('final_qp_summary.csv',summary)
    metrics(rows,'final')
    eq,bd = comparisons(summary)
    write('final_equal_rate_perceptual_summary.csv',eq); write('final_perceptual_bd_rate.csv',bd)
    write('final_same_qp_summary.csv',same_qp(rows))
    for anchor,label in [('original','original'),('v4_10000','v4_10k')]:
        rates=same_qp(rows,anchor)
        e,b=comparisons(summary,anchor)
        output=[]
        for q in [0,1,2,3,'all']:
            entry=dict(next(r for r in rates if r['method']=='selected' and r['qp']==q))
            for metric in METRICS:
                er=next(r for r in e if r['method']=='selected' and r['metric']==metric)
                br=next(r for r in b if r['method']=='selected' and r['metric']==metric)
                entry.update({metric+'_BD_rate':br['BD_rate_percent'] if q=='all' else '',
                              metric+'_BD_status':br['status'] if q=='all' else 'not_applicable_single_qp',
                              metric+'_BD_reason':br['reason'] if q=='all' else '',
                              'mean_equal_rate_delta_'+metric:er['mean_equal_rate_delta'] if q=='all' else ''})
            output.append(entry)
        write(f'v4_1_vs_{label}_summary.csv',output)
    dump('parts/final_report_done.json',dict(status='PASS',rows=len(rows)))
    print('FINAL REPORT PASS',len(rows),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--split',choices=('validation','final'),required=True)
    args=parser.parse_args()
    (validation if args.split=='validation' else final)()
