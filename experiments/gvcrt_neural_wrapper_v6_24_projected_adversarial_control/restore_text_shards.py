"""Reassemble published text shards without replacing any differing file."""
from io21 import *
for item in load(ROOT/'file_index.json')['files']:
    if 'shards' not in item:continue
    target=ROOT/item['path'];assert target.resolve().is_relative_to(ROOT)
    if target.exists():
        assert sha(target)==item['sha256'],('existing file differs; preserved',str(target))
        continue
    data=[]
    for spec in item['shards']:
        path=ROOT/spec['path'];assert path.resolve().is_relative_to(ROOT) and sha(path)==spec['sha256'];data.append(path.read_bytes())
    content=b''.join(data);assert len(content)==item['bytes'] and hashlib.sha256(content).hexdigest()==item['sha256']
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
print('TEXT SHARDS VERIFIED AND RESTORED')

