"""Source-blocked independent receiver emitting compact exact tensor hashes."""
import argparse

from bridge import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    setup()
    source_root = Path(json.loads((V1 / "config.json").read_text())["dataset_root"])
    guard_rejections = []

    def guard(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            blocked = {"teachers_train6", "sender_oracles", "optimizations", "input_cache"}
            if (path.is_relative_to(source_root) or path.is_relative_to(V1 / "data") or
                    path.is_relative_to(V11 / "results_v1/oracle") or blocked.intersection(path.parts) or
                    (path.parent.name == "base" and path.suffix == ".pt") or
                    path.name.endswith("candidate_frames.csv")):
                guard_rejections.append(str(path))
                raise RuntimeError(f"Receiver cannot access sender data: {path}")
    sys.addaudithook(guard)

    # Active negative test proves that the hook rejects source-side data.
    try:
        open(V1 / "data" / "forbidden_probe", "rb")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Source guard did not reject a forbidden path")

    bundle = Bundle(False)
    intra, inter, _ = load_models(json.loads((V1 / "config.json").read_text()))
    rows = decode_base(intra, inter, args.base)
    decoded = ORC2.decode(args.stream, args.base, MODEL, bundle.net.entropy)
    assert decoded["level"] == CONFIG["level"] and decoded["delta"] == DELTA
    before = dpb_hash(inter)
    rgb_hashes, symbol_hashes, zero_exact = [], [], []
    with torch.no_grad():
        for row in rows:
            frame = row["frame"]
            if row["type"] == "I":
                assert decoded["frames"][frame] is None
                reconstruction = rgb01(row["x_base"])
                symbol_hashes.append(None)
            else:
                ec = bundle.fixed.ell_c(row["ell"].cuda().float())
                q = row["q_recon"].cuda().float()
                symbols = decoded["frames"][frame]
                residual = bundle.net.synthesize(symbols.cuda().float() * DELTA, ec)
                reconstruction = rgb01(bundle.fixed.g(ec + residual, q))
                if torch.count_nonzero(symbols) == 0:
                    assert torch.count_nonzero(residual) == 0
                    assert torch.equal(reconstruction, rgb01(bundle.fixed.g(ec, q)))
                    zero_exact.append(frame)
                symbol_hashes.append(tensor_hash(symbols))
            rgb_hashes.append(tensor_hash(reconstruction.cpu()))
    assert dpb_hash(inter) == before
    bundle.check()
    json_dump(Path(args.output), {"base_sha256": sha(args.base), "stream_sha256": sha(args.stream),
              "model_sha256": sha(MODEL), "rgb_hashes": rgb_hashes,
              "symbol_hashes": symbol_hashes, "zero_exact_frames": zero_exact,
              "base_state_unchanged": True, "source_guard_active_negative_test": True,
              "guard_rejections": guard_rejections})


if __name__ == "__main__":
    main()
