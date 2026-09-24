"""Load exact frozen V1.5 candidate symbols and their provenance."""
from bridge import *


def o2_path(spec):
    tag = "B6q_lambda3" if spec["group"] == "network_train6" else "Bfull_lambda3"
    return V13 / "results_v1/optimizations" / tag / spec["sample"] / "selected.pt"


def selected_symbols(path):
    item = torch.load(path, map_location="cpu", weights_only=False)
    symbols = item["symbols"].cpu().short()
    ORC2.checked_symbols(symbols)
    assert tuple(symbols.shape) == tuple(CONFIG["code_shape"])
    return symbols


def load_video_candidates(split, video, bundle, output=None):
    stream = e_stream(split, video)
    decoded = ORC2.decode(stream, base_path(video), MODEL, bundle.net.entropy)
    assert decoded["level"] == 3 and decoded["delta"] == DELTA and decoded["shape"] == (8, 17, 30)
    result, provenance = {}, []
    for spec in sorted((s for s in specs(split) if s["video"] == video), key=lambda s: s["frame"]):
        e = decoded["frames"][spec["frame"]].short()
        if split == "Train2":
            paths = {"O1": V14 / "results_v1/optimizations/adaptive_lambda1" / spec["sample"] / "selected.pt",
                     "O2": o2_path(spec)}
        else:
            if output is None:
                raise ValueError("Val6 O1/O2 require an output directory")
            paths = {name: output / "optimizations" / name / spec["sample"] / "selected.pt" for name in ("O1", "O2")}
        candidates = {"Z": torch.zeros(CONFIG["code_shape"], dtype=torch.int16), "E": e,
                      "O1": selected_symbols(paths["O1"]), "O2": selected_symbols(paths["O2"])}
        result[spec["frame"]] = candidates
        provenance.extend([
            {"split": split, "video": video, "frame": spec["frame"], "candidate": "Z", "source": "fixed_zero", "sha256": None},
            {"split": split, "video": video, "frame": spec["frame"], "candidate": "E", "source": str(stream), "sha256": file_hash(stream)},
            *({"split": split, "video": video, "frame": spec["frame"], "candidate": name,
               "source": str(path), "sha256": file_hash(path)} for name, path in paths.items())])
    assert len(result) == 15
    return result, provenance
