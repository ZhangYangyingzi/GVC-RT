"""Verify completed FFV1 artifacts' native geometry, frame count and source cadence."""
from common_s2 import *
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor

def probe(a):
 raw=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_packets','-show_entries','stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_packets','-of','json',a['path']],text=True)
 s=json.loads(raw)['streams'];assert len(s)==1;s=s[0]
 native='RGB8_frame_hashes' in a
 width=a['width'] if native else a['ROI']['width']*7
 height=a['height'] if native else a['ROI']['height']+48
 assert (s['width'],s['height'])==(width,height),(a['path'],s,width,height)
 assert s['codec_name']=='ffv1' and s['pix_fmt']=='bgr0'
 assert int(s['nb_read_packets'])==a['frames'],(a['path'],s['nb_read_packets'],a['frames'])
 assert Fraction(s['avg_frame_rate'])==Fraction(str(a['fps'])).limit_denominator(1000000),(a['path'],s['avg_frame_rate'],a['fps'])
 return dict(path=a['path'],manifest_SHA256=a['SHA256'],kind='native' if native else 'ROI',probe=s,status='PASS')

def main():
 env();manifest=load(ROOT/'video_manifest.json');assert manifest['status']=='PASS'
 for seq in {a['sequence'] for a in manifest['native_clips']}:
  clips=[a for a in manifest['native_clips'] if a['sequence']==seq]
  assert {a['model'] for a in clips}=={'GT','Original','A2000','B300',*GROUPS}
  assert len({(a['stream_SHA256'],a['stream_bytes'],a['codec_trace_SHA256'],a['source_SHA256']) for a in clips})==1
  assert len({a['RGB8_frame_hashes'][0] for a in clips if a['model']!='GT'})==1
 for a in manifest['ROI_comparisons']:
  assert a['pixels_unchanged'] and a['scale']==1 and a['labels_outside_original_ROI']
  if a['anonymous']:assert sha(a['mapping_path'])==a['mapping_SHA256']
 entries=manifest['native_clips']+manifest['ROI_comparisons']
 with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(probe,entries))
 dump(ROOT/'audits/media_metadata.json',dict(status='PASS',files=len(records),native_clips=len(manifest['native_clips']),ROI_comparisons=len(manifest['ROI_comparisons']),records=records,FFV1_one_packet_per_frame=True,source_cadence_geometry_and_frame_count_verified=True,time=time.time()))

if __name__=='__main__':main()
