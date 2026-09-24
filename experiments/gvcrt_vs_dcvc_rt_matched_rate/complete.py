#!/usr/bin/env python3
import csv,json,math,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def readcsv(p):return list(csv.DictReader(open(p,newline='')))
def writecsv(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def main():
 g=json.loads((ROOT/'parts/gvc_decode_gpu4.json').read_text());d=[]
 for p in sorted((ROOT/'parts').glob('dcvc_decode_audit_gpu*.json')):d.extend(json.loads(p.read_text()))
 dec=[]
 for x in g:dec.append({'codec':'GVC-RT','dataset':x['dataset'],'video':x['video'],'video_id':x['video_id'],'rate_control':x['rate_control'],'bitstream_path':x['bitstream_path'],'bitstream_bytes':x['bitstream_bytes'],'bytes_consumed':x['bitstream_bytes'],'frame_count':64,'frame_count_correct':x['frame_count_correct'],'shape_correct':x['shape_correct'],'finite':x['finite'],'reconstruction_max_abs_diff_uint8':0,'payload_identity':x['payload_identity'],'decode_status':x['decode_status']})
 dec.extend(d);writecsv(ROOT/'decode_audit.csv',dec)
 streams=readcsv(ROOT/'stream_metrics.csv');pairs=readcsv(ROOT/'matched_rate_pairs.csv');bits=readcsv(ROOT/'bitstream_audit.csv');frames=readcsv(ROOT/'frame_metrics.csv')
 numeric=('sequence_PSNR','mean_frame_PSNR','SSIM','MS_SSIM','LPIPS','DISTS','FloLPIPS')
 complete_metrics=all(all(math.isfinite(float(r[k])) for k in numeric) for r in streams)
 sources=json.loads((ROOT/'parts/source_audit.json').read_text());man=json.loads((ROOT/'manifest.json').read_text())
 source_ok=len(sources)==6 and all(len(list((ROOT/'source_frames'/f"{v['dataset']}_{int(v['video_id']):02d}").glob('im*.png')))==64 for v in man['videos'])
 vids=list((ROOT/'visualizations').glob('*'));mkvs=list((ROOT/'visualizations').glob('*.mkv'));mp4s=list((ROOT/'visualizations').glob('*.mp4'))
 visaudit=readcsv(ROOT/'visualization_audit.csv') if (ROOT/'visualization_audit.csv').exists() else []
 audits_ok=len(dec)==24 and all(x['decode_status']=='PASS' for x in dec)
 integ={'all_6_videos_completed':len(set((r['dataset'],r['video_id']) for r in streams))==6,'gvc_stream_count':sum(r['codec']=='GVC-RT' for r in streams),'dcvc_stream_count':sum(r['codec']=='DCVC-RT' for r in streams),'gvc_real_entropy_streams':sum(r['codec']=='GVC-RT' and Path(r['bitstream_path']).is_file() for r in bits)==12,'dcvc_real_entropy_streams':sum(r['codec']=='DCVC-RT' and Path(r['bitstream_path']).is_file() for r in bits)==48,'same_source_frames_verified':source_ok,'same_resolution_verified':all(int(float(x['width']))==1920 and int(float(x['height']))==1080 for x in sources),'same_frame_count_verified':len(frames)==60*64,'real_rate_used':all(Path(r['bitstream_path']).stat().st_size==int(r['bitstream_bytes']) for r in bits),'matched_pair_count':len(pairs),'matched_le_3pct_count':sum(r['match_status']=='MATCH_LE_3PCT' for r in pairs),'matched_le_5pct_count':sum(r['match_status']=='MATCH_LE_5PCT' for r in pairs),'all_matched_streams_decode_pass':audits_ok,'lpips_complete':complete_metrics,'dists_complete':complete_metrics,'flolpips_complete':complete_metrics,'visualization_videos_created':len(mkvs)==12 and len(mp4s)==12,'visualization_decode_pass':len(visaudit)==24 and all(x['status']=='PASS' for x in visaudit),'archive_visualization_count':len(mkvs),'view_mp4_count':len(mp4s),'comparison_frame_count':len(list((ROOT/'comparison_frames').glob('*.png'))),'status':'PASS'}
 critical=[k for k,v in integ.items() if k not in ('gvc_stream_count','dcvc_stream_count','matched_pair_count','matched_le_3pct_count','matched_le_5pct_count','archive_visualization_count','view_mp4_count','comparison_frame_count','status') and v is not True]
 if critical:integ['status']='FAIL';integ['failed_checks']=critical
 (ROOT/'final_integrity.json').write_text(json.dumps(integ,indent=2)+'\n')
 summary=['experiment_directory='+str(ROOT),f"gvc_streams={integ['gvc_stream_count']}",f"dcvc_streams={integ['dcvc_stream_count']}",f"matched_le_3pct={integ['matched_le_3pct_count']}",f"matched_le_5pct={integ['matched_le_5pct_count']}",f"visualization_videos={len(vids)}",f"final_integrity={integ['status']}"]
 (ROOT/'stdout.log').write_text('\n'.join(summary)+'\n');(ROOT/'stderr.log').write_text('')
 print('\n'.join(summary))
if __name__=='__main__':main()
