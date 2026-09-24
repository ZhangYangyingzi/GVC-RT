"""Complete V1.6 pipeline orchestration."""
import argparse
import importlib.util

from bridge import *


def load_run(name):
    spec = importlib.util.spec_from_file_location(f"gvcrt_v16_{name}", HOME / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_v1")
    parser.add_argument("--force", action="store_true", help="rerun and replace an existing output directory")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = (HOME / output).resolve()
    if output.exists() and any(output.iterdir()):
        if args.force:
            raise ValueError("Refusing destructive --force; choose a new empty output directory")
        if ((output / "final_verification.json").exists() and
                (output / "input_hash_manifest.json").exists() and
                (HOME / "report.md").exists()):
            print(f"Already complete: {output}")
            return
        raise ValueError(f"Output directory is non-empty and incomplete: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Scripts default to HOME/results_v1. A new custom output is supported through
    # direct module calls to avoid ambiguous imports from historical experiments.
    setup()
    for stage in ("experiment", "verify_receivers", "analyze", "diagnostics", "final_verify", "report"):
        print(f"Running {stage}...")
        load_run(stage)(output)
    print(f"Complete: {output}")


if __name__ == "__main__":
    main()
