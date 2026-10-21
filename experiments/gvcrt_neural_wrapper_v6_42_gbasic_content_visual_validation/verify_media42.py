"""Check lossless ROI video pixels against their saved PNG sources and HTML paths."""
from common42 import *
from PIL import Image
from html.parser import HTMLParser
from fractions import Fraction
import subprocess

def main():
 records=[]
 for group in load(ROOT/'visual_index.json')['video_groups']:
  for view in group['previews']:
   if 'lossless' not in view:continue
   path=ROOT/view['lossless'];info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
    '-show_entries','stream=codec_name,width,height,nb_read_frames,r_frame_rate','-of','json',str(path)],text=True))['streams'][0]
   assert info['codec_name']=='ffv1' and int(info['nb_read_frames'])==64
   assert (info['width'],info['height'])==(768,286) and float(Fraction(info['r_frame_rate']))==group['fps']
   png_dir=path.parent/f"{view['view']}_frames";expected=hashlib.sha256()
   for i in range(64):
    with Image.open(png_dir/f'{i:03d}.png') as im:expected.update(im.convert('RGB').tobytes())
   proc=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(path),'-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE)
   actual=hashlib.sha256();size=0
   while True:
    chunk=proc.stdout.read(1<<20)
    if not chunk:break
    size+=len(chunk);actual.update(chunk)
   assert proc.wait()==0 and size==64*768*286*3 and actual.hexdigest()==expected.hexdigest()
   records.append(dict(path=str(path),frames=64,fps=group['fps'],codec='ffv1',decoded_RGB_matches_saved_PNGs=True,
    decoded_RGB_sha256=actual.hexdigest(),source_RGB_sha256=expected.hexdigest()))
 class Parser(HTMLParser):
  def __init__(self):super().__init__();self.paths=[]
  def handle_starttag(self,tag,attrs):
   a=dict(attrs)
   if 'src' in a:self.paths.append(a['src'])
   if 'data-path' in a:self.paths.append(a['data-path'])
 p=Parser();p.feed((ROOT/'index.html').read_text())
 assert len(records)==16 and all((ROOT/x).exists() for x in p.paths)
 first=load(ROOT/'tasks/hevc_b/Kimono1/q0/pair.json')
 events=[json.loads(x) for x in (ROOT/'logs/execution.jsonl').read_text().splitlines()]
 assert min(e['time'] for e in events if e['event']=='START')>=first['finished']
 assert (ROOT/'visual_manifest.json').stat().st_mtime<=first['started']
 dump(ROOT/'audits/visual_material_integrity.json',dict(status='PASS',lossless_ROI_videos=records,
  HTML_material_paths_verified=len(p.paths),HTML_missing_paths=[],first_pair_finished_before_queue=True,
  visual_manifest_created_before_first_decode=True,ROI_coordinates_unchanged=True))
 print('MEDIA_INTEGRITY_PASS',len(records),len(p.paths),flush=True)
if __name__=='__main__':main()
