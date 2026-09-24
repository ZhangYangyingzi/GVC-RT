#!/usr/bin/env python3
import hashlib
import json
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V2 = REPO / "experiments/gvcrt_neural_wrapper_v2_retest_fixed_uvg"
SPLIT = REPO / "expericent_generation_input/expericent_generator_aware_latent_distortion_v11/data_split.json"
ARCHIVES = Path("/Huang_group/zyyz/datasets/UltraVideo-Long/clips_long_1920")
TARGET = 256


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    previous = json.loads((V2 / "test_manifest.json").read_text())["videos"]
    split = json.loads(SPLIT.read_text())["splits"]
    train_old = split["train"]
    # Five V2 added tests use validation indices 0..4. Freeze 5..9 plus confirmation 0.
    validation = split["validation"][5:10] + split["confirmation"][:1]
    test_paths = {Path(v["source_path"]).resolve() for v in previous if v["dataset"] == "fresh_ulong"}
    test_hashes = {v["source_sha256"] for v in previous if v["dataset"] == "fresh_ulong"}
    frozen_validation = split["validation"] + split["confirmation"]
    val_names = {Path(v["filename"]).name for v in frozen_validation}
    val_hashes = {v["file_sha256"] for v in frozen_validation}
    test_names = {Path(v["source_path"]).name.split("__")[-1] for v in previous if v["dataset"] == "fresh_ulong"}
    rows = []
    for item in train_old:
        path = Path(item["local_path"])
        if path.is_file():
            rows.append({"filename": item["filename"], "path": str(path), "sha256": item["file_sha256"], "origin": "v11_train"})
    used_names = {Path(r["filename"]).name for r in rows} | val_names | test_names
    archives = sorted(ARCHIVES.glob("clips_long_1920_*.zip"))
    available = 0
    candidates = []
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if not info.filename.lower().endswith(".mp4"):
                    continue
                available += 1
                basename = Path(info.filename).name
                if basename not in used_names:
                    candidates.append((str(archive), info.filename, info.CRC))
                    used_names.add(basename)
    random.Random(20260924).shuffle(candidates)
    cache = ROOT / "training_videos"
    cache.mkdir(parents=True, exist_ok=True)
    for archive_path, member, crc in candidates:
        if len(rows) >= TARGET:
            break
        target = cache / Path(member).name
        if not target.is_file():
            with zipfile.ZipFile(archive_path) as z, z.open(member) as src, open(target, "wb") as dst:
                while block := src.read(1 << 20):
                    dst.write(block)
        rows.append({"filename": member, "path": str(target), "sha256": sha(target),
                     "origin": "extracted_archive", "archive": archive_path, "crc32": f"{crc:08x}"})
    if len(rows) < 200:
        raise RuntimeError(f"only {len(rows)} training videos available")
    train_hashes = {r["sha256"] for r in rows}
    if len(train_hashes) != len(rows):
        raise RuntimeError("duplicate training video SHA256")
    if train_hashes & (val_hashes | test_hashes):
        raise RuntimeError("train/validation/test SHA256 intersection")
    val_rows = [{"filename": v["filename"], "path": v["local_path"], "sha256": v["file_sha256"], "fps": v["probe"]["avg_frame_rate"]} for v in validation]
    test_rows = [v for v in previous if v["dataset"] == "fresh_ulong"]
    assert len(val_rows) == 6 and len(test_rows) == 8
    assert not ({v["sha256"] for v in val_rows} & {v["source_sha256"] for v in test_rows})
    (ROOT / "train_manifest.json").write_text(json.dumps({"available_video_count": available,
        "train_video_count": len(rows), "videos": rows, "seed": 20260924,
        "online_sampling": True}, indent=2) + "\n")
    (ROOT / "validation_manifest.json").write_text(json.dumps({"validation_video_count": len(val_rows), "videos": val_rows}, indent=2) + "\n")
    (ROOT / "test_manifest.json").write_text(json.dumps({"test_video_count": len(test_rows), "videos": test_rows}, indent=2) + "\n")
    print(json.dumps({"available": available, "train": len(rows), "validation": len(val_rows), "test": len(test_rows)}))


if __name__ == "__main__": main()
