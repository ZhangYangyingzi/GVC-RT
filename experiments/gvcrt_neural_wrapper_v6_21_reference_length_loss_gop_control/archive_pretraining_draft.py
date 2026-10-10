"""Preserve the untrained draft affected by the dimension-validation bug."""
from io21 import *
assert not list((ROOT/'branches').glob('*/checkpoints/*.pt'))
dest=ROOT/'archived_drafts/native_dimensions_correction';assert not dest.exists()
records=[]
for name in ('plans','audits/sample_records','audits/video_inventory','train_manifest_ulong.json','train_manifest_uvg.json','audits/pool_filter.json','dataset_split.json','sampling_order.json','preparation_status.json','preflight_audit.json','audits/protocol_hashes.json'):
    p=ROOT/name
    if not p.exists():continue
    fs=list(p.rglob('*')) if p.is_dir() else [p]
    for f in fs:
        if f.is_file():records.append(dict(path=str(f.relative_to(ROOT)),sha256=sha(f)))
    target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);p.rename(target)
dump(ROOT/'audits/draft_correction.json',dict(status='CORRECTED_BEFORE_TRAINING',reason='native dimensions must be inherited per video, not force 1080p for training sources',optimizer_updates=0,archive=str(dest),files=records))

