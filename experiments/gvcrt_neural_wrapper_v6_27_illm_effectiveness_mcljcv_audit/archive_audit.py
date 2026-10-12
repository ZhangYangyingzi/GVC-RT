from io21 import *
import zlib,zipfile
DATA=Path('/Huang_group/zyyz/datasets/MCL-JCV')
def main():
 entries={};archives=[]
 for z in sorted(DATA.glob('*.zip')):
  archives.append(dict(path=str(z),bytes=z.stat().st_size,sha256=sha(z)))
  with zipfile.ZipFile(z) as f:
   for i in f.infolist():
    if '/1080P/YUV_source/' in i.filename and i.filename.endswith('.yuv'):entries[Path(i.filename).name]=dict(archive=str(z),member=i.filename,CRC32=i.CRC,bytes=i.file_size)
 checked=[]
 for p in sorted((DATA/'1080p_source').glob('*.yuv')):
  crc=0
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4<<20),b''):crc=zlib.crc32(b,crc)
  e=entries[p.name];assert (crc&0xffffffff)==e['CRC32'] and p.stat().st_size==e['bytes'];checked.append(dict(path=str(p),**e,CRC32_reproduced=crc&0xffffffff,status='PASS'))
 dump(ROOT/'audits/archive_source_verification.json',dict(status='PASS',archives=archives,sources=checked,source_count=len(checked),all_member_CRC_verified=True));print('ARCHIVE SOURCE CRC PASS',len(checked),flush=True)
if __name__=='__main__':main()
