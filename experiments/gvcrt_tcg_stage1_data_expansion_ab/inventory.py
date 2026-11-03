"""Resumable CRC/SHA/native-read inventory, four CPU workers, bounded private cache."""
from ab_common import *
import concurrent.futures,collections,traceback,zipfile,zlib
def archive_entries():
 entries=[];packages=[]
 for f in sorted(ARCHIVES.glob('*.zip')):
  with zipfile.ZipFile(f) as z:
   rows=[dict(archive=f.name,entry=i.filename,identity=Path(i.filename).stem.lower(),size_bytes=i.file_size,compressed_bytes=i.compress_size,crc32=f'{i.CRC:08x}') for i in z.infolist() if i.filename.lower().endswith('.mp4')]
  entries.extend(rows);packages.append(dict(filename=f.name,size_bytes=f.stat().st_size,video_entries=len(rows),central_directory_sha256=digest(rows)))
 dump(ROOT/'inventory/archives.json',dict(packages=packages,package_count=len(packages),entry_count=len(entries),unique_entry_names=len({r['entry'] for r in entries})))
 dump(ROOT/'inventory/entries.json',entries);return entries
def inspect_media(path):
 import cv2,numpy as np
 cv2.setNumThreads(1)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,duration,codec_name','-of','json',str(path)],timeout=120))['streams'][0]
 width,height=int(probe['width']),int(probe['height'])
 if width<256 or height<256:raise ValueError('GEOMETRY_BELOW_256')
 cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fp=[];rgb=[]
 try:
  for i in range(64):
   ok,a=cap.read()
   if not ok:raise ValueError('FEWER_THAN_64_READABLE_CONSECUTIVE_FRAMES')
   if a.shape!=(height,width,3):raise ValueError('VARIABLE_OR_INCORRECT_GEOMETRY')
   if i in (0,31,63):
    b=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);small=cv2.resize(b,(9,8),interpolation=cv2.INTER_AREA)
    fp.append(f'{int.from_bytes(np.packbits(small[:,1:]>small[:,:-1]).tobytes(),"big"):016x}')
    rgb.append(hashlib.sha256(cv2.cvtColor(a,cv2.COLOR_BGR2RGB).tobytes()).hexdigest())
 finally:cap.release()
 if count<64:raise ValueError('METADATA_FRAME_COUNT_BELOW_64')
 return dict(width=width,height=height,frames=count,probe=probe,fingerprint_first64_dhash=fp,fingerprint_first64_RGB=rgb,fingerprint_scope='Native decoded frames 0/31/63; 9x8 grayscale only for duplicate screening, never training resize; temporal crops/reencodes beyond these anchors may remain unresolved')
def inspect_entry(row):
 from audit import identity
 key=digest([row['archive'],row['entry']]);dest=ROOT/f'inventory/records/{key}.json'
 if dest.exists():
  r=load(dest);assert r['archive']==row['archive'] and r['crc32']==row['crc32'];return r
 disk();tmp=ROOT/f'cache/inventory_{key}.mp4';crc=0;h=hashlib.sha256();size=0
 try:
  # zipfile validates the source CRC at EOF, while explicit CRC/SHA index bytes.
  with zipfile.ZipFile(ARCHIVES/row['archive']) as z,z.open(row['entry']) as src,tmp.open('wb') as out:
   while chunk:=src.read(8*2**20):
    out.write(chunk);h.update(chunk);crc=zlib.crc32(chunk,crc);size+=len(chunk)
    if size>20*2**30:raise RuntimeError('ONE_MEDIA_EXCEEDS_CACHE_WORKER_LIMIT_20_GiB')
  assert size==row['size_bytes'] and f'{crc&0xffffffff:08x}'==row['crc32']
  r=dict(row,status='ELIGIBLE',sha256=h.hexdigest(),CRC_verified=True,**inspect_media(tmp))
 except Exception as exc:
  r=dict(row,status='EXCLUDED',reason=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  if tmp.exists():tmp.unlink()
 dump(dest,r);return r
def main():
 with lock('inventory.lock'):
  status('inventory');entries=archive_entries();done=[]
  with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
   for r in pool.map(inspect_entry,entries,chunksize=1):
    done.append(r)
    if len(done)%25==0:
     dump(ROOT/'inventory/status.json',dict(status='RUNNING',completed=len(done),expected=len(entries),counts=dict(collections.Counter(x['status'] for x in done)),CPU_workers=4,cache_cap_bytes=80*2**30,**disk()))
     event('INVENTORY_PROGRESS',completed=len(done),expected=len(entries))
  dump(ROOT/'inventory/qualified.json',done)
  dump(ROOT/'inventory/status.json',dict(status='PASS',completed=len(done),expected=len(entries),counts=dict(collections.Counter(x['status'] for x in done)),CPU_workers=4))
  event('INVENTORY_COMPLETE',counts=dict(collections.Counter(x['status'] for x in done)))
if __name__=='__main__':main()
