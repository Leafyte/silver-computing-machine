# India ETA 4Q CPU release

This bundle predicts delivery duration in minutes from the committed 59 numeric GBDT features. It derives the four frozen VQC inputs from the supplied feature matrix. It includes both trained 4Q residual candidates, the frozen operational selection/fallback policy, and the Ridge comparison control.

The NumPy loader does not require Torch, QpiAI, cloud access, or a QPU. The original test partition was exposed in an earlier 2Q run, so any 4Q test scores are retrospective and are not a pristine holdout claim. No raw dataset, OOF table, or validation rows are included.

Run `python predictor.py --verify-reference` to check the synthetic-only fixture, or pass a JSON file containing `feature_names` and `rows` to `--input-json`.
