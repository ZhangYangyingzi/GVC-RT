"""Source-free receiver for either historical ORC2 or V1.4 ORS1 streams."""
import argparse

from bridge import *
import sparse_bitstream as ORS1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--stream", required=True)
    p.add_argument("--format", choices=["old", "sparse"], required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    setup()
    source_root = Path(json.loads((V1 / "config.json").read_text())["dataset_root"])

    def guard(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            forbidden_names = {"teachers_train6", "sender_oracles", "optimizations", "input_cache"}
            if (path.is_relative_to(source_root) or path.is_relative_to(V1 / "data") or
                    path.is_relative_to(V11 / "results_v1/oracle") or any(k in path.parts for k in forbidden_names) or
                    (path.parent.name == "base" and path.suffix == ".pt")):
                raise RuntimeError(f"Receiver cannot access sender data: {path}")

    sys.addaudithook(guard)
    bundle = Bundle(perceptual=False)
    i, pnet, _ = load_models(json.loads((V1 / "config.json").read_text()))
    torch.cuda.synchronize()
    start = time.perf_counter()
    rows = decode_base(i, pnet, args.base)
    torch.cuda.synchronize()
    base_seconds = time.perf_counter() - start
    start = time.perf_counter()
    decoded = (ORC2.decode(args.stream, args.base, MODEL, bundle.net.entropy) if args.format == "old"
               else ORS1.decode(args.stream, args.base, MODEL, bundle.net.entropy))
    parse_seconds = time.perf_counter() - start
    assert decoded["level"] == 3 and decoded["delta"] == DELTA and decoded["shape"] == (8, 17, 30)
    before = dpb_hash(pnet)
    frames, hashes, zero_exact, synthesis = [], [], [], []
    with torch.no_grad():
        for row in rows:
            frame = row["frame"]
            torch.cuda.synchronize()
            start = time.perf_counter()
            if row["type"] == "I":
                assert decoded["frames"][frame] is None
                y = rgb01(row["x_base"])
                hashes.append(None)
            else:
                ec = bundle.fixed.ell_c(row["ell"].cuda().float())
                q = row["q_recon"].cuda().float()
                symbols = decoded["frames"][frame]
                residual = bundle.net.synthesize(symbols.cuda().float() * DELTA, ec)
                y = rgb01(bundle.fixed.g(ec + residual, q))
                if torch.count_nonzero(symbols) == 0:
                    assert torch.count_nonzero(residual) == 0
                    assert torch.equal(y, rgb01(bundle.fixed.g(ec, q)))
                    zero_exact.append(frame)
                hashes.append(tensor_hash(symbols))
            frames.append(y.cpu())
            torch.cuda.synchronize()
            synthesis.append(time.perf_counter() - start)
    assert dpb_hash(pnet) == before
    bundle.check()
    save_pt(args.output, {"frames": frames, "symbols_hash": hashes, "zero_exact_frames": zero_exact,
                          "base_state_unchanged": True, "source_access_guard": True,
                          "base_decode_seconds": base_seconds, "stream_parse_seconds": parse_seconds,
                          "synthesis_seconds": synthesis, "format": args.format})


if __name__ == "__main__":
    main()
