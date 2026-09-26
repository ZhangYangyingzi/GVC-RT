from v5_utils import ROOT,BRANCHES,load,read

def main():
    integrity=load('final_integrity.json');rows=read('clip_ablation_summary.csv')
    print('experiment directory:',ROOT)
    print('source V4.1 checkpoint:',load('config.json')['source_checkpoint'])
    for branch in BRANCHES:
        v=next(r for r in rows if r['split']=='validation' and r['method']==branch)
        f=next(r for r in rows if r['split']=='final' and r['method']==branch)
        print(branch+':')
        print('updates:',f['optimizer_updates']);print('optimized P frames:',f['optimized_P_frames'])
        print('validation mean kbps:',v['mean_real_kbps']);print('final mean kbps:',f['mean_real_kbps'])
        for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
            print(metric+' BD-rate:',f[metric+'_BD_rate_vs_Original'] or 'invalid: '+f[metric+'_BD_reason'])
        print('mean equal-rate FloLPIPS delta vs Original:',f['mean_equal_rate_delta_FloLPIPS'])
    comp=read('final_clip8_vs_clip4_summary.csv')[0]
    print('clip8 vs clip4:');print('rate change (%):',comp['clip8_rate_change_vs_clip4_percent'])
    for metric in ('LPIPS','DISTS','FloLPIPS','FID'):
        print('equal-rate '+metric+' delta:',comp['mean_equal_rate_delta_'+metric] or 'invalid: '+comp[metric+'_reason'])
    for label,key in [('temporal DPB detach unchanged','temporal_DPB_detach_unchanged'),('BPTT disabled','BPTT_disabled'),('compression frozen','compression_core_frozen'),('independent decode','independent_decode_pass')]:
        print(label+':','PASS' if integrity[key] else 'FAIL')
    print('final_integrity:',integrity['status'])
    for path in ('clip_ablation_summary.csv','training_position_summary.csv','validation_temporal_position_summary.csv','final_equal_rate_summary.csv','rd_curves'):print(path+':',ROOT/path)

if __name__=='__main__':main()
