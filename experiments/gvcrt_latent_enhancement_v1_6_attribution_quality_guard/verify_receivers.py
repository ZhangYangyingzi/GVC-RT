"""Verify every unique new guarded stream with an independent receiver process."""
import subprocess

from bridge import *


def sender_hashes(split, video, stream, bundle):
    decoded = ORC2.decode(stream, base_path(video), MODEL, bundle.net.entropy)
    base = base_rows(next(e for e in entries(split) if e["video_id"] == video))
    hashes = []
    with torch.no_grad():
        for frame, row in enumerate(base):
            if frame == 0:
                reconstruction = rgb01(row["x_base"])
            else:
                ec = bundle.fixed.ell_c(row["ell"].cuda().float())
                residual = bundle.net.synthesize(decoded["frames"][frame].cuda().float() * DELTA, ec)
                reconstruction = rgb01(bundle.fixed.g(ec + residual, row["q_recon"].cuda().float()))
            hashes.append(tensor_hash(reconstruction.cpu()))
    return hashes


def run(out):
    setup()
    bundle = Bundle(False)
    records, aliases = [], []
    for split in ("Train2", "Val6"):
        allocations = [r for r in read_csv(out / f"{split}_guard_allocations.csv") if r["stream_sent"]]
        unique = {}
        for row in allocations:
            key = (row["video"], row["stream_sha256"])
            unique.setdefault(key, row)
            aliases.append({"split": split, "video": row["video"], "method": row["method"],
                            "alpha": row["alpha"], "stream_sha256": row["stream_sha256"],
                            "unique_receiver_key": f"{row['video']}_{row['stream_sha256'][:16]}"})
        for (video, stream_hash), row in unique.items():
            folder = out / "receiver" / split
            folder.mkdir(parents=True, exist_ok=True)
            output = folder / f"{video}_{stream_hash[:16]}.json"
            command = [sys.executable, str(HOME / "receiver.py"), "--base", str(base_path(video)),
                       "--stream", row["stream"], "--output", str(output)]
            subprocess.run(command, check=True, cwd=HOME)
            received = json.loads(output.read_text())
            expected = sender_hashes(split, video, row["stream"], bundle)
            exact = received["rgb_hashes"] == expected
            assert exact and received["base_state_unchanged"] and received["source_guard_active_negative_test"]
            records.append({"split": split, "video": video, "stream": row["stream"],
                            "stream_sha256": stream_hash, "frame_count": len(expected),
                            "all_rgb_hashes_exact": exact, "zero_exact_frames": received["zero_exact_frames"],
                            "base_state_unchanged": True, "source_guard_active_negative_test": True,
                            "receiver_output": str(output), "receiver_output_sha256": sha(output)})
    csv_dump(out / "receiver_verification.csv", records)
    csv_dump(out / "receiver_logical_aliases.csv", aliases)
    bundle.check()


if __name__ == "__main__":
    run(HOME / "results_v1")
