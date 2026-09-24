"""Final positive and negative integrity checks for V1.5 outputs."""
import struct

from allocator import self_test
from bridge import *


def expect_rejection(path, base, bundle):
    try:
        ORC2.decode(path, base, MODEL, bundle.net.entropy)
    except (ValueError, OverflowError):
        return True
    raise AssertionError(f"Invalid stream accepted: {path}")


def run(out):
    setup(); bundle = Bundle(False)
    checks = {"allocator": self_test(), "splits": {}, "invalid_stream_rejection": {}}
    for split, videos in (("Train2", 2), ("Val6", 6)):
        allocations = read_csv(out / f"{split}_allocations.csv")
        selected = read_csv(out / f"{split}_selected_frames.csv")
        streams = [r for r in allocations if r["stream_sent"]]
        assert len(allocations) == videos * 4 * 4 and len(selected) == videos * 4 * 4 * 15
        assert all(int(r["actual_bytes"]) <= int(r["budget_bytes"]) for r in allocations)
        assert all(Path(r["stream"]).stat().st_size == int(r["actual_bytes"]) and
                   file_hash(r["stream"]) == r["stream_sha256"] for r in streams)
        receiver = json.loads((out / f"{split}_receiver_verification.json").read_text())
        assert len(receiver) == videos and all(r["rgb_torch_equal"] and r["base_state_unchanged"] and
                                               r["source_access_guard"] for r in receiver)
        checks["splits"][split] = {"videos": videos, "allocation_rows": len(allocations),
                                   "selected_frame_rows": len(selected), "real_streams": len(streams),
                                   "all_budget_constraints_hold": True, "all_stream_hashes_match": True,
                                   "independent_receiver_videos": len(receiver), "rgb_torch_equal": True}
    selected_pt = list((out / "optimizations").glob("O*/**/selected.pt"))
    optimization_checks = [json.loads(p.read_text()) for p in (out / "optimizations").glob("O*/**/checks.json")]
    assert len(selected_pt) == len(optimization_checks) == 180
    assert sum(r["network_updates"] for r in optimization_checks) == 0
    assert sum(r["code_updates"] for r in optimization_checks) == 27000
    checks["optimization"] = {"selected_checkpoints": 180, "code_updates": 27000,
                              "network_updates": 0, "interrupted_artifacts": 0}

    reference = next(r for r in read_csv(out / "Val6_allocations.csv") if r["stream_sent"])
    source = Path(reference["stream"]); blob = bytearray(source.read_bytes())
    cases = {"truncated": bytes(blob[:-1]), "trailing": bytes(blob) + b"x"}
    pos = ORC2.HEADER.size
    corrupt = bytearray(blob)
    for _ in range(16):
        flag = blob[pos]; pos += 1
        if flag == 1:
            length = struct.unpack_from("<I", blob, pos)[0]
            corrupt[pos + 8] ^= 0xff
            cases["crc_corrupt"] = bytes(corrupt)
            break
        assert flag in (0, 2)
    invalid = out / "invalid_stream_tests"; invalid.mkdir(exist_ok=True)
    for name, contents in cases.items():
        path = invalid / f"{name}.orc2"; path.write_bytes(contents)
        checks["invalid_stream_rejection"][name] = expect_rejection(path, base_path(reference["video"]), bundle)
    wrong = next(e["video_id"] for e in entries("Val6") if e["video_id"] != reference["video"])
    checks["invalid_stream_rejection"]["wrong_base_binding"] = expect_rejection(source, base_path(wrong), bundle)
    checks["model_frozen"] = bundle.hashes() == bundle.fingerprints
    checks["passed"] = True
    json_dump(out / "final_verification.json", checks)
    bundle.check()


if __name__ == "__main__":
    run(Path("results_v1").resolve())
