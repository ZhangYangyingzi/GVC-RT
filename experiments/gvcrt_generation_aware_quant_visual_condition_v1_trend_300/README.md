# Generation-aware local quantization and spatial visual condition, v1

This experiment adds two receiver-synchronous modules to the existing GVC-RT codec and runs a fixed four-group, 300-update continuation. See `METHOD_MAPPING.md` for tensor domains, `PROTOCOL.md` for the fixed data and evaluation contract, and `preflight.json` for engineering checks.

`runtime.py` contains the modules and codec overrides. `train.py`, `preflight.py` and `evaluate.py` are the executable entry points. `train_plan.json` records the exact 300 matched windows, crops and QPs. Source data and inherited model files remain in their original locations. Large optimizer checkpoints, complete RANS streams and per-frame caches reside under `server_artifacts/`; only code, manifests, tables, figures and a few previews are intended for GitHub.

The inherited checkpoint has known mixed U-Long/UVG training exposure. The new continuation uses only the selected cleaned U-Long training sources. Cross-dataset rows describe outcomes for those sequences and do not establish historical nonexposure.
