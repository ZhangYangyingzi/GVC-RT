"""Cheap post-run artifact and malformed-container checks; no model execution."""
import argparse
import cv2

from bridge import *
import sparse_bitstream as ORS1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results_v1")
    args = p.parse_args()
    out = HOME / args.output
    source = out / "same_integer_streams/OPT_lambda1/fb63c34e-0c28-4a4d-b8f8-59932b5e7875.ors1"
    blob = source.read_bytes()
    cases = {
        "truncated_container": blob[:-1],
        "trailing_container_byte": blob + b"\x00",
        "unknown_frame_mode": blob[:ORS1.HEADER.size + 1] + b"\xff" + blob[ORS1.HEADER.size + 2:],
    }
    rows = []
    temp = out / "malformed_container_tests"
    temp.mkdir(exist_ok=True)
    video = "fb63c34e-0c28-4a4d-b8f8-59932b5e7875"
    # Decode requires only a compatible entropy object; loading the checkpoint on CPU avoids GPU work.
    ck = torch.load(MODEL, map_location="cpu", weights_only=False)
    entropy = ResidualCodec(ck["residual_scale"])
    entropy.load_state_dict(ck["model"], strict=True)
    invalid_csv = out / "sparse_container_invalid_tests.csv"
    if invalid_csv.exists():
        rows = read_csv(invalid_csv)
        assert len(rows) == 3 and all(r["rejected"] for r in rows)
    else:
        for name, data in cases.items():
            path = temp / f"{name}.ors1"
            with path.open("xb") as f:
                f.write(data)
            try:
                ORS1.decode(path, base_path(video), MODEL, entropy.entropy)
            except ValueError as error:
                rows.append({"case": name, "rejected": True, "error": str(error)})
            else:
                raise AssertionError(f"Malformed container accepted: {name}")
        csv_dump(invalid_csv, rows)
    for path in (out / "diagnostic_plots").glob("*.png"):
        assert cv2.imread(str(path)) is not None
    videos = list((out / "sequences").glob("**/*.mkv"))
    assert len(videos) >= 8 and all(path.stat().st_size > 0 for path in videos)
    manifest = json.loads((out / "artifact_manifest.json").read_text())
    manifest.update({"report_sha256": file_hash(HOME / "report.md"),
                     "decision_sha256": file_hash(out / "decision.json"),
                     "verification_sha256": file_hash(out / "verification.json"),
                     "sparse_container_invalid_tests_sha256": file_hash(out / "sparse_container_invalid_tests.csv"),
                     "representative_video_count": len(videos),
                     "all_diagnostic_plots_readable": True,
                     "all_representative_videos_nonempty": True})
    manifest["source_sha256"] = {p.name: file_hash(p) for p in sorted(HOME.glob("*.py"))}
    path = out / "artifact_manifest.json"
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"malformed_container_cases": len(rows), "videos": len(videos),
                      "plots": len(list((out / "diagnostic_plots").glob("*.png")))}, indent=2))


if __name__ == "__main__":
    main()
