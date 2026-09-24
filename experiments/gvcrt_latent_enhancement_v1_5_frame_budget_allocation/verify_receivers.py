"""Run independent receiver processes and compare exact RGB tensors."""
import subprocess

from bridge import *


def run(out, split):
    setup()
    allocations = [r for r in read_csv(out / f"{split}_allocations.csv") if r["stream_sent"]]
    chosen = {}
    for row in allocations:
        key = row["video"]
        if key not in chosen or row["actual_bytes"] > chosen[key]["actual_bytes"]:
            chosen[key] = row
    records = []
    for video, row in chosen.items():
        output = out / "receiver_outputs" / split / f"{video}.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(HOME / "receiver.py"), "--base", str(base_path(video)),
                   "--stream", row["stream"], "--output", str(output)]
        if not output.exists():
            subprocess.run(command, check=True, cwd=HOME)
        received = torch.load(output, map_location="cpu", weights_only=False)
        bundle = Bundle(False); decoded = ORC2.decode(row["stream"], base_path(video), MODEL, bundle.net.entropy)
        base = base_rows(next(e for e in entries(split) if e["video_id"] == video))
        sender = []
        with torch.no_grad():
            for frame, base_row in enumerate(base):
                if frame == 0:
                    reconstruction = rgb01(base_row["x_base"])
                else:
                    ec = bundle.fixed.ell_c(base_row["ell"].cuda().float())
                    q = base_row["q_recon"].cuda().float()
                    residual = bundle.net.synthesize(decoded["frames"][frame].cuda().float() * DELTA, ec)
                    reconstruction = rgb01(bundle.fixed.g(ec + residual, q))
                sender.append(reconstruction.cpu())
        exact = all(torch.equal(a, b) for a, b in zip(sender, received["frames"]))
        assert exact and received["base_state_unchanged"] and received["source_access_guard"]
        records.append({"split": split, "video": video, "stream": row["stream"],
                        "rgb_torch_equal": exact, "base_state_unchanged": True,
                        "source_access_guard": True, "zero_exact_frames": received["zero_exact_frames"]})
    json_dump(out / f"{split}_receiver_verification.json", records)
