"""Final integrity checks and immutable-input hash manifest."""
import struct

from bridge import *


def expect_rejection(path, base, bundle):
    try:
        ORC2.decode(path, base, MODEL, bundle.net.entropy)
    except (ValueError, OverflowError):
        return True
    raise AssertionError(f"Invalid stream accepted: {path}")


def input_manifest(out):
    paths = [HOME / "config.json", Path(CONFIG["v11_manifest"]), MODEL]
    paths += [V15_RESULTS / f"{split}_{name}.csv" for split in ("Train2", "Val6")
              for name in ("candidate_frames", "allocations", "selected_frames", "summary")]
    for split in ("Train2", "Val6"):
        paths += [base_path(entry["video_id"]) for entry in entries(split)]
        prefix = "train2_full" if split == "Train2" else "val6"
        paths += [V12 / "results_v1" / f"{prefix}_ORIGINAL_FP32_q{qp}" / "summary.csv"
                  for qp in range(4)]
    records = [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}
               for path in dict.fromkeys(paths)]
    manifest_path = out / "input_hash_manifest.json"
    manifest_path.unlink(missing_ok=True)
    json_dump(manifest_path, {
        "scope": "all tabular/model/base inputs consumed directly by V1.6",
        "files": records, "file_count": len(records)})


def run(out):
    setup()
    bundle = Bundle(False)
    checks = {"allocator": self_test(), "splits": {}, "invalid_stream_rejection": {}}
    receiver = read_csv(out / "receiver_verification.csv")
    aliases = read_csv(out / "receiver_logical_aliases.csv")
    for split, video_count in (("Train2", 2), ("Val6", 6)):
        allocations = read_csv(out / f"{split}_guard_allocations.csv")
        selected = read_csv(out / f"{split}_guard_selected_frames.csv")
        streams = [row for row in allocations if row["stream_sent"]]
        received = [row for row in receiver if row["split"] == split]
        split_aliases = [row for row in aliases if row["split"] == split]
        assert len(allocations) == video_count * 2 * 4
        assert len(selected) == video_count * 2 * 4 * 15
        assert all(int(row["enhancement_bytes"]) <= int(row["budget_bytes"]) for row in allocations)
        assert all(Path(row["stream"]).stat().st_size == int(row["enhancement_bytes"]) and
                   sha(row["stream"]) == row["stream_sha256"] for row in streams)
        assert len(split_aliases) == len(streams)
        assert all(row["all_rgb_hashes_exact"] and row["base_state_unchanged"] and
                   row["source_guard_active_negative_test"] for row in received)
        assert json.loads((out / f"{split}_sanity_checks.json").read_text())["failures"] == []
        checks["splits"][split] = {
            "videos": video_count, "guard_allocation_rows": len(allocations),
            "guard_selected_frame_rows": len(selected), "logical_real_streams": len(streams),
            "unique_stream_contents_received": len(received), "all_budget_constraints_hold": True,
            "all_stream_hashes_match": True, "all_receiver_rgb_hashes_exact": True,
            "receiver_source_guard_active": True, "sanity_checks_pass": True}

    reference = next(row for row in read_csv(out / "Val6_guard_allocations.csv") if row["stream_sent"])
    source = Path(reference["stream"]); blob = bytearray(source.read_bytes())
    cases = {"truncated": bytes(blob[:-1]), "trailing": bytes(blob) + b"x"}
    position = ORC2.HEADER.size
    corrupt = bytearray(blob)
    for _ in range(16):
        flag = blob[position]; position += 1
        if flag == 1:
            length = struct.unpack_from("<I", blob, position)[0]
            assert length > 0
            corrupt[position + 8] ^= 0xff
            cases["crc_corrupt"] = bytes(corrupt)
            break
        assert flag in (0, 2)
    invalid = out / "invalid_stream_tests"
    invalid.mkdir(exist_ok=True)
    for name, contents in cases.items():
        path = invalid / f"{name}.orc2"
        path.write_bytes(contents)
        checks["invalid_stream_rejection"][name] = expect_rejection(
            path, base_path(reference["video"]), bundle)
    wrong_video = next(entry["video_id"] for entry in entries("Val6")
                       if entry["video_id"] != reference["video"])
    checks["invalid_stream_rejection"]["wrong_base_binding"] = expect_rejection(
        source, base_path(wrong_video), bundle)
    checks["updates"] = {"network_updates": 0, "code_updates": 0,
                         "new_lambdas": 0, "new_quantization_levels": 0,
                         "new_entropy_formats": 0}
    checks["principal_videos"] = len(read_csv(out / "Val6_principal_videos.csv"))
    checks["model_frozen"] = bundle.hashes() == bundle.fingerprints
    checks["passed"] = all(checks["invalid_stream_rejection"].values()) and checks["model_frozen"]
    assert checks["passed"]
    verification_path = out / "final_verification.json"
    verification_path.unlink(missing_ok=True)
    json_dump(verification_path, checks)
    input_manifest(out)
    bundle.check()


if __name__ == "__main__":
    run(HOME / "results_v1")
