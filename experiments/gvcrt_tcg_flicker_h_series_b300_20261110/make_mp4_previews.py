"""CPU-only H.264 MP4 viewing copies; original FFV1 media and metrics are immutable."""
from common_s2 import *
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from fractions import Fraction
import struct,traceback

def probe(path,count=False):
 args=['ffprobe','-v','error','-select_streams','v:0']
 if count:args+=['-count_frames']
 args+=['-show_entries','stream=codec_name,codec_tag_string,profile,level,pix_fmt,width,height,avg_frame_rate,nb_read_frames','-of','json',str(path)]
 return json.loads(subprocess.check_output(args,text=True))['streams'][0]

def faststart(path):
 boxes={}
 with path.open('rb') as file:
  size=path.stat().st_size
  while file.tell()+8<=size:
   pos=file.tell();length,kind=struct.unpack('>I4s',file.read(8))
   if length==1:length=struct.unpack('>Q',file.read(8))[0]
   if length==0:length=size-pos
   assert length>=8
   boxes[kind.decode('ascii')]=pos;file.seek(pos+length)
 return boxes.get('moov',size)<boxes.get('mdat',-1)

def convert(a):
 source=Path(a['path']);kind='native' if 'RGB8_frame_hashes' in a else 'ROI'
 dest=ROOT/'media/previews'/kind/(source.stem+'.mp4');dest.parent.mkdir(parents=True,exist_ok=True)
 identity=digest(dict(source_SHA256=a['SHA256'],settings='libx264_high_level4.1_crf14_medium_yuv420p_bt709_tv_no_resize_v1'))
 receipt=ROOT/'media/previews/receipts'/f'{identity}.json'
 if receipt.exists():
  record=load(receipt);assert record['status']=='PASS' and sha(dest)==record['SHA256'] and record['source_SHA256']==a['SHA256'];return record
 assert sha(source)==a['SHA256'];before=probe(source);fps=before['avg_frame_rate'];partial=dest.with_name(dest.stem+'.partial.mp4')
 logpath=ROOT/'logs/mp4_previews'/(source.stem+'.log');logpath.parent.mkdir(parents=True,exist_ok=True);started=time.time()
 command=['ffmpeg','-hide_banner','-v','warning','-nostdin','-y','-threads','2','-i',str(source),'-map','0:v:0','-an','-sn','-dn','-filter_threads','1','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','libx264','-preset','medium','-crf','14','-profile:v','high','-level:v','4.1','-pix_fmt','yuv420p','-r',fps,'-vsync','1','-threads','2','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv','-tag:v','avc1','-movflags','+faststart',str(partial)]
 with logpath.open('ab') as log:subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
 actual=probe(partial,True)
 assert actual['codec_name']=='h264' and actual['codec_tag_string']=='avc1' and actual['pix_fmt']=='yuv420p' and actual['level']==41
 assert (actual['width'],actual['height'])==(before['width'],before['height'])
 assert Fraction(actual['avg_frame_rate'])==Fraction(fps) and int(actual['nb_read_frames'])==a['frames']
 assert faststart(partial);partial.replace(dest)
 record=dict(status='PASS',identity=identity,path=str(dest),SHA256=sha(dest),bytes=dest.stat().st_size,source=str(source),source_SHA256=a['SHA256'],kind=kind,sequence=a['sequence'],model=a.get('model'),ROI_name=a.get('ROI_name'),anonymous=a.get('anonymous',False),step=a.get('step'),QP=5,frames=a['frames'],fps=fps,width=actual['width'],height=actual['height'],codec='H.264 High level4.1, yuv420p, avc1',encoding='libx264 medium CRF14; BT.709 limited; faststart',spatial_resize=False,lossy_viewing_only=True,used_for_metrics=False,used_as_neural_bitrate=False,original_FFV1_preserved=True,full_decode_verified=True,seconds=time.time()-started,time=time.time())
 dump(receipt,record);return record

def main():
 env()
 with lock('locks/mp4_previews.lock'):
  original_manifest_SHA256=sha(ROOT/'video_manifest.json');manifest=load(ROOT/'video_manifest.json');entries=manifest['ROI_comparisons']+manifest['native_clips'];done=[];begin=time.time()
  dump(ROOT/'media/previews/status.json',dict(status='RUNNING',completed=0,expected=len(entries),CPU_workers=2,GPUs_used=[],time=time.time()))
  try:
   first=convert(entries[0]);done.append(first);event('mp4_preview_READY',path=first['path'],completed=1,expected=len(entries))
   with ThreadPoolExecutor(max_workers=2) as executor:
    pending={executor.submit(convert,a) for a in entries[1:]}
    while pending:
     finished,pending=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
     for future in finished:
      record=future.result();done.append(record);event('mp4_preview_READY',path=record['path'],completed=len(done),expected=len(entries))
     dump(ROOT/'media/previews/status.json',dict(status='RUNNING',completed=len(done),expected=len(entries),CPU_workers=2,GPUs_used=[],seconds=time.time()-begin,time=time.time()))
   assert sha(ROOT/'video_manifest.json')==original_manifest_SHA256
   record=dict(status='PASS',ffmpeg_version=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],completed=len(done),expected=len(entries),native_clips=sum(x['kind']=='native' for x in done),ROI_comparisons=sum(x['kind']=='ROI' for x in done),original_FFV1_manifest_SHA256=original_manifest_SHA256,original_FFV1_preserved=True,metrics_and_training_unchanged=True,CPU_workers=2,GPUs_used=[],records=sorted(done,key=lambda x:x['path']),time=time.time())
   dump(ROOT/'media/previews/manifest.json',record);dump(ROOT/'audits/mp4_previews.json',record);dump(ROOT/'media/previews/status.json',dict(status='COMPLETE',completed=len(done),expected=len(entries),time=time.time()))
  except Exception as error:
   dump(ROOT/'media/previews/status.json',dict(status='FAILED',completed=len(done),expected=len(entries),error=repr(error),traceback=traceback.format_exc(),resume_command=f'{PYTHON} -B {ROOT}/make_mp4_previews.py',time=time.time()));raise

if __name__=='__main__':main()
