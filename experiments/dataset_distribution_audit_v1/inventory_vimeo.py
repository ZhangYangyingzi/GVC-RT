"""Recursive inventory and bounded archive validation; never bulk-extract/download."""
import collections
import io
import random
import struct
import traceback
import zipfile
from PIL import Image
from audit_io import *

def seqkey(name):
    parts=Path(name).parts
    if 'sequences' not in parts:return None
    i=parts.index('sequences')
    if len(parts)!=i+4:return None
    return '/'.join(parts[i+1:i+3])

def main():
    for d in ('logs','manifests','cache','parts','plots'): (ROOT/d).mkdir(parents=True,exist_ok=True)
    assert VIMEO.is_dir()
    extensions=collections.Counter();sizes=collections.Counter();resolutions=collections.Counter();framecounts=collections.Counter()
    splitpaths=[];archives=[];symlinks=[];header_errors=[];sequence_files={};directory_count=0;file_count=0;total_bytes=0
    tree=ROOT/'vimeo_directory_tree.txt';listing=ROOT/'vimeo_file_inventory.csv'
    with tree.open('w') as tr,listing.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','suffix','symlink','width','height','header_status']);w.writeheader()
        for base,dirs,files in os.walk(VIMEO,followlinks=False):
            dirs.sort();files.sort();directory_count+=1;bp=Path(base)
            tr.write(str(bp.relative_to(VIMEO))+'/  ['+str(len(files))+' files]\n')
            for name in dirs:
                p=bp/name
                if p.is_symlink():symlinks.append(dict(path=str(p),target=os.readlink(p),kind='directory'))
            for name in files:
                p=bp/name;st=p.lstat();suffix=p.suffix.lower();file_count+=1;total_bytes+=st.st_size;extensions[suffix]+=1;sizes[suffix]+=st.st_size
                width=height='';status='not_image'
                if p.is_symlink():symlinks.append(dict(path=str(p),target=os.readlink(p),kind='file'))
                if suffix=='.zip':archives.append(p)
                if name in ('sep_trainlist.txt','sep_testlist.txt','tri_trainlist.txt','tri_testlist.txt'):splitpaths.append(p)
                if suffix in ('.png','.jpg','.jpeg','.bmp','.webp'):
                    try:
                        if suffix=='.png':
                            with p.open('rb') as pf:header=pf.read(24)
                            assert header[:8]==b'\x89PNG\r\n\x1a\n' and header[12:16]==b'IHDR'
                            width,height=struct.unpack('>II',header[16:24])
                        else:
                            with Image.open(p) as image:width,height=image.size
                        assert width>0 and height>0;resolutions[f'{width}x{height}']+=1;status='valid_header'
                        key=seqkey(str(p))
                        if key is not None:sequence_files.setdefault(key,[]).append(str(p))
                    except Exception as e:status='ERROR';header_errors.append(dict(path=str(p),error=repr(e)))
                w.writerow(dict(path=str(p),bytes=st.st_size,suffix=suffix,symlink=p.is_symlink(),width=width,height=height,header_status=status))
    for paths in sequence_files.values():framecounts[len(paths)]+=1
    dump(ROOT/'vimeo_inventory.json',dict(status='PASS' if not header_errors else 'FAIL',dataset_exists=True,root=str(VIMEO),
        regular_file_and_symlink_entries=file_count,directory_count=directory_count,total_file_entry_bytes=total_bytes,
        extension_counts=dict(extensions),extension_bytes=dict(sizes),resolution_distribution_from_all_image_headers=dict(resolutions),
        extracted_sequence_count=len(sequence_files),frame_count_per_extracted_sequence=dict(framecounts),
        archives=[str(p) for p in archives],split_files=[str(p) for p in splitpaths],symlinks=symlinks,header_errors=header_errors,
        resolution_audit_scope='All extracted image headers; only explicitly reported samples fully decoded',
        inventory_csv_sha256=sha(listing),directory_tree_sha256=sha(tree)))
    print('RECURSIVE INVENTORY',file_count,'files',len(sequence_files),'sequences',flush=True)
    archive_audits=[];archive_sequences={};archive_splits={}
    for archive in archives:
        record=dict(path=str(archive),bytes=archive.stat().st_size,mtime_ns=archive.stat().st_mtime_ns,errors=[])
        try:
            with zipfile.ZipFile(archive) as z:
                infos=z.infolist();central=hashlib.sha256();members=[];local={};counts=collections.Counter()
                for info in infos:
                    central.update(json.dumps([info.filename,info.CRC,info.file_size,info.compress_size,info.header_offset],ensure_ascii=True).encode())
                    assert 0<=info.header_offset<archive.stat().st_size
                    assert info.header_offset+info.compress_size<=archive.stat().st_size
                    suffix=Path(info.filename).suffix.lower();counts[suffix]+=1
                    key=seqkey(info.filename)
                    if key and suffix=='.png':local.setdefault(key,[]).append(info.filename);members.append(info.filename)
                    if Path(info.filename).name in ('sep_trainlist.txt','sep_testlist.txt','tri_trainlist.txt','tri_testlist.txt'):
                        archive_splits[Path(info.filename).name]=dict(archive=str(archive),member=info.filename,text=z.read(info).decode())
                sample=random.Random(SEED).sample(sorted(members),min(64,len(members)));checks=[]
                for name in sample:
                    payload=z.read(name)  # zipfile verifies the selected member CRC on EOF.
                    with Image.open(io.BytesIO(payload)) as image:image.load();dims=list(image.size)
                    checks.append(dict(member=name,sha256=hashlib.sha256(payload).hexdigest(),resolution=dims,CRC_checked=True,decoded=True))
                record.update(status='PASS',central_directory_valid=True,central_directory_sha256=central.hexdigest(),entry_count=len(infos),
                    uncompressed_bytes=sum(i.file_size for i in infos),sequence_count=len(local),frame_count_distribution=dict(collections.Counter(map(len,local.values()))),
                    extension_counts=dict(counts),sample_member_checks=checks,
                    full_archive_payload_CRC_scan=False,corruption_scope='Central directory validated; 64 seeded image members CRC-checked and fully decoded. Unread payloads are not certified.')
                if 'septuplet' in archive.name.lower() or not archive_sequences:
                    archive_sequences={k:dict(archive=str(archive),members=sorted(v)) for k,v in local.items()}
        except Exception as e:record.update(status='FAIL',central_directory_valid=False);record['errors'].append(repr(e))
        archive_audits.append(record)
    dump(ROOT/'vimeo_archive_audit.json',dict(status='PASS' if all(r['status']=='PASS' for r in archive_audits) else 'FAIL',archives=archive_audits,
        no_download=True,no_bulk_extraction=True))
    preferred='sep' if any(p.name=='sep_trainlist.txt' for p in splitpaths) or 'sep_trainlist.txt' in archive_splits else 'tri'
    def readsplit(name):
        found=next((p for p in splitpaths if p.name==name),None)
        if found:
            text=found.read_text()
            if name in archive_splits:assert text.split()==archive_splits[name]['text'].split()
            return text.split(),dict(path=str(found),sha256=sha(found))
        assert name in archive_splits,('Official split missing',name)
        entry=archive_splits[name];return entry['text'].split(),{k:entry[k] for k in ('archive','member')}
    train,train_info=readsplit(preferred+'_trainlist.txt');test,test_info=readsplit(preferred+'_testlist.txt')
    assert len(train)==len(set(train)) and len(test)==len(set(test)) and not set(train)&set(test)
    expected=7 if preferred=='sep' else 3
    usable_extracted={k for k,v in sequence_files.items() if len(v)==expected and {Path(p).name for p in v}=={f'im{i}.png' for i in range(1,expected+1)}}
    usable_archive={k for k,v in archive_sequences.items() if len(v['members'])==expected}
    usable=sorted(set(train)&(usable_extracted|usable_archive));missing=sorted(set(train)-set(usable));assert usable
    chosen=random.Random(SEED).sample(usable,min(2048,len(usable)))
    sample_rows=[];corrupt=[];manifest=[]
    for ordinal,key in enumerate(chosen):
        if key in usable_extracted:
            paths=sorted(sequence_files[key]);source='existing_extracted'
        else:
            entry=archive_sequences[key];destination=ROOT/'cache/vimeo_samples'/key;destination.mkdir(parents=True,exist_ok=True);paths=[]
            with zipfile.ZipFile(entry['archive']) as z:
                for member in entry['members']:
                    p=destination/Path(member).name
                    if not p.exists():
                        data=z.read(member);tmp=p.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(p)
                    paths.append(str(p))
            source='deterministic_subset_extracted'
        try:
            hashes=[];dimensions=set()
            for p in paths:
                with Image.open(p) as image:image.load();assert image.mode=='RGB';dimensions.add(image.size)
                hashes.append(sha(p))
            assert len(dimensions)==1 and len(paths)==expected;width,height=next(iter(dimensions))
            identity=hashlib.sha256(''.join(hashes).encode()).hexdigest()
            row=dict(dataset='vimeo_train2048',sample_id='vimeo_'+key.replace('/','_'),sequence=key,official_split='train',
                path=str(Path(paths[0]).parent),frames=expected,width=width,height=height,source_mode=source,sequence_file_hash=identity,status='PASS')
            sample_rows.append(row);manifest.append(dict(**row,frame_paths=paths,frame_sha256=hashes,fps=None,fps_status='Not provided by official image sequence metadata'))
        except Exception as e:corrupt.append(dict(sequence=key,error=repr(e)));sample_rows.append(dict(sequence=key,status='FAIL',error=repr(e)))
        if (ordinal+1)%256==0:print('SAMPLE DECODE',ordinal+1,'/',len(chosen),flush=True)
    write(ROOT/'vimeo_sample_manifest.csv',sample_rows)
    dump(ROOT/'manifests/vimeo_train2048.json',dict(status='PASS' if not corrupt else 'FAIL',seed=SEED,official_train_source=train_info,selected_before_reference_statistics=True,videos=manifest))
    dump(ROOT/'vimeo_split_audit.json',dict(status='PASS',preferred_structure='septuplet' if preferred=='sep' else 'triplet',
        official_train_count=len(train),official_test_count=len(test),train_test_intersection=[],official_train_source=train_info,official_test_source=test_info,
        usable_train_sequences=len(usable),missing_train_sequences=missing,usable_extracted_train=len(set(train)&usable_extracted),
        usable_archive_train=len(set(train)&usable_archive),chosen_train_count=len(chosen),seed=SEED,selected_test_sequences=[]))
    sample_bytes=sum(Path(p).stat().st_size for v in manifest for p in v['frame_paths'])
    dump(ROOT/'vimeo_training_readiness.json',dict(status='PASS' if not corrupt and not header_errors else 'FAIL',dataset_exists=True,
        official_train_split_available=True,structure=preferred,number_of_usable_train_sequences=len(usable),number_of_usable_train_frames=len(usable)*expected,
        complete_train_extracted=set(train)<=usable_extracted,archive_present=bool(archives),sampled_train_sequences=len(chosen),
        fully_decoded_sample_frames=len(manifest)*expected,corrupt_samples_count=len(corrupt),corrupt_samples=corrupt,
        unsampled_full_decode_corruption_status='Not exhaustively tested; all extracted image headers inventoried',
        resolutions=sorted({f'{v["width"]}x{v["height"]}' for v in manifest}),
        direct_current_wrapper_loader_compatible=False,reason='Current sample_clip uses cv2.VideoCapture on a video path, not an image-sequence directory',
        minimal_adapter='Read im1.png through im7.png in temporal order as RGB; choose four consecutive frames and one shared random 256x256 crop; float /255; no resize, interpolation or looped frames',
        sampled_extraction_bytes=sample_bytes,remaining_full_extraction_required_for_this_experiment=False,
        full_archive_uncompressed_bytes=sum(r.get('uncompressed_bytes',0) for r in archive_audits),
        codec_32_frame_requirement_satisfied=expected>=32,codec_frame_count_decision='Await user approval for 7 native frames; no artificial 32-frame sequences created'))
    assert not corrupt and not header_errors
    dump(ROOT/'pipeline_status.json',dict(status='AWAITING_VIMEO_CODEC_FRAME_DECISION',inventory_complete=True,source_statistics_complete=False,codec_profiling_started=False,updated_unix=time.time()))
    print('VIMEO INVENTORY COMPLETE',len(usable),'usable train',expected,'frames each',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'logs/inventory_failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
