from diag_utils import *
def main():
    assert load('final_integrity.json')['status']=='PASS'
    print('visual_diagnostics directory:',ROOT)
    print('random UVG sequences:',[s['name'] for s in samples() if s['dataset']=='uvg'])
    print('random U-Long videos:',[s['name'] for s in samples() if s['dataset']=='ulong'])
    for r in load('comparison_video_manifest.json')['videos']:print('comparison MP4:',r['path'])
    for r in load(f'parts/comparison_{tag(samples()[0])}_709.json')['videos']:print('BT709 comparison MP4:',r['path'])
    print('contact sheet directory:',ROOT/'contact_sheets')
    for r in load('uvg_color_conversion_audit.json')['conversions']:print('BT601 vs BT709 MP4:',r['videos'][2])
    print('BT709 smoke summary:',ROOT/'bt709_smoke_summary.csv')
    for r in read('bt709_smoke_summary.csv'):
        if r['source_conversion']=='BT709_EXPLICIT':
            print(r['method'],'QP'+r['QP'],*(f'{k}={r[k]}' for k in ('kbps','LPIPS','DISTS','FloLPIPS')))
    print('U-Long minimum measured bitrate:')
    for r in read('rd_curves/ulong/minimum_bitrates.csv'):print(LABELS[r['method']],r['mean_real_kbps'],'kbps')
    print('new U-Long RD curve directory:',ROOT/'rd_curves/ulong')
    for name in ('ulong_per_qp_diagnostic.csv','uvg_per_sequence_diagnostic.csv','color_statistics.csv'):print(name,ROOT/name)
    print('final_integrity PASS')
if __name__=='__main__':main()
