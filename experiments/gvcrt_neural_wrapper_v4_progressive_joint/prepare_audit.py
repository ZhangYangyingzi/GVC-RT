"""Verify unchanged V3 splits and record input provenance without test evaluation."""
import json
from pathlib import Path
from report import ROOT, V3, sha, dump

records = {}
snapshot = {}
for name, count in (("train",256),("validation",6),("test",8)):
    path = ROOT/f"{name}_manifest.json"
    source = V3/path.name
    assert path.read_bytes() == source.read_bytes(), name
    videos = json.loads(path.read_text())["videos"]
    assert len(videos) == count
    hashes = {v.get("sha256", v.get("source_sha256")) for v in videos}
    assert None not in hashes and len(hashes) == count
    names = {Path(v.get("filename", v.get("name"))).stem for v in videos}
    assert len(names) == count
    records[name] = (hashes, names)
    snapshot[str(source)] = sha(source)
result = {"train_video_count":256,"validation_video_count":6,"test_video_count":8}
for a,b in (("train","validation"),("train","test"),("validation","test")):
    assert not(records[a][0] & records[b][0]) and not(records[a][1] & records[b][1]), (a,b)
    result[f"{a}_{b}_intersection_zero"] = True
for path in V3.iterdir():
    if path.suffix in ("json","csv") or path.suffix in (".json",".csv"):
        snapshot[str(path)] = sha(path)
cfg = json.loads((ROOT/"config.json").read_text())
snapshot[cfg["initial_checkpoint"]] = sha(cfg["initial_checkpoint"])
previous = ROOT/"source_snapshot.json"
if previous.exists():
    old = json.loads(previous.read_text())
    assert all(sha(p) == digest for p,digest in old.items()), "source snapshot changed"
else:
    dump("source_snapshot.json", snapshot)
result["status"] = "PASS"
dump("split_audit.json", result)
print(json.dumps(result))
