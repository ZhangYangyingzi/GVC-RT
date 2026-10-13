"""Read byte-verified multipart publication indices."""
from io21 import *
def _parts(header,key):
 for spec in header.get(key,[]):
  p=ROOT/spec['path'];assert p.resolve().is_relative_to(ROOT)
  assert p.stat().st_size==spec['bytes']<=524288 and sha(p)==spec['sha256']
  yield load(p)
def read_file_index():
 h=load(ROOT/'file_index.json');entries=list(h.get('files',[]))
 for part in _parts(h,'index_parts'):entries.extend(part['files'])
 assert len(entries)==h.get('entry_count',len(entries))
 return entries
def read_upload_manifest():
 h=load(ROOT/'upload_manifest.json');entries=dict(h.get('files',{}))
 for part in _parts(h,'manifest_parts'):
  assert not entries.keys() & part['files'].keys();entries.update(part['files'])
 assert len(entries)==h.get('entry_count',len(entries))
 return entries

