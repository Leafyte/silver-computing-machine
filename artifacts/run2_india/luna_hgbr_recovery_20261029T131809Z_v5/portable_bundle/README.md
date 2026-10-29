# RUN2 India portable inference bundle

This bundle contains the recovered GBDT checkpoint, both trained two-qubit residual checkpoints, the train-only Q feature scaler, the feature schema and preprocessing artifact, and an inference loader. It contains no source dataset.

Install the runtime versions in `bundle_manifest.json`. Add this folder to `PYTHONPATH`, then use `from inference import load_bundle, predict_minutes`. The operational policy returns the GBDT fallback (`alpha=0`); candidate modes are retained for scientific comparison. Inputs must follow the exact feature orders in `bundle_manifest.json`.

Run `python portable_smoke.py` from this directory to verify portable checkpoint loading and deterministic API parity. The fixture is synthetic and has no labels.
