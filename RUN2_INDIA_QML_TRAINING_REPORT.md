# RUN2 Indian ETA QML Training Report

## Current status update (2026-10-29)

**VQC training is on hold pending architecture review. No VQC training updates have been run, no test labels have been opened, and no QML release checkpoint has been produced.** The previous missing-baseline blocker was resolved by Worker A: a frozen HistGradientBoostingRegressor and leakage-safe courier-grouped predictions now exist under `artifacts/india_eta_recovery/`. The checkpoint SHA-256 is `f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf`; OOF has 29,592 rows, validation has 5,989 rows, with 819 train couriers and 175 validation couriers. The validation role split is 4,124 model-selection rows / 117 couriers and 1,865 fusion-calibration rows / 58 couriers. Worker A reports validation MAE 3.14284 minutes and RMSE 3.90597 minutes for the recovered GBDT.

The available baseline and prediction tables passed the cheap keyed-row, unique-ID, role-map, and courier-fold checks. The raw CSV checksum discrepancy noted in the initial audit was reconciled by canonical LF normalization against the committed source identity; no source data was edited to mask it. The initial environment section below remains historical evidence from before the recovery and should not be read as the current artifact inventory.

The CUDA Torch wheel download was stopped before installation at the explicit resource gate; the isolated runtime still has no Torch. A lightweight Numba kernel attempt failed with `NvvmSupportError` because `nvvm.dll` was unavailable. Existing NumPy circuit finite-difference checks and QpiAI local simulator parity passed, and the portable four-feature circuit parity test passed. These checks do not imply a trained model or usable CUDA backend. See [`docs/RUN2_OPUS_ARCHITECTURE_REVIEW.md`](docs/RUN2_OPUS_ARCHITECTURE_REVIEW.md) for the baseline-vs-extension gate sequence and review questions.

**Test partition remains sealed:** no test prediction or metric file was generated or read for this work. The portable inference test result was 2 passed in 3.25s using the sibling SDK environment. No training was run.

## Initial artifact audit (historical; before Worker A recovery)

**Initial status: BLOCKED before model training.** At that point no residual VQC was trained, no test set was evaluated, and no release checkpoint was produced. The missing-model and prediction findings in the following audit describe that earlier state only.

## Verified source artifacts

The same India data artifacts are present in the `trainingv3` worktree and the original `silver-computing-machine` checkout. Their matching digests show these are duplicate copies, not independent trained-model releases.

| Artifact | Path in `trainingv3` | SHA-256 |
|---|---|---|
| Clean source table | `data/indian_delivery_clean.parquet` | `9034d24389c7162f0ea0e275e5b4cc9ec16f04bba75c0f239dbcef8535a1f7c1` |
| GBDT feature table | `data/indian_delivery_gbdt.parquet` | `bd03b47c3d3e23050329cefc69d752aad607031cacdb9696151d237e3534a627` |
| VQC feature table | `data/indian_delivery_vqc.parquet` | `a633677472e6e0e874c2797a66eb7b011e3a307285553f8fa00c6ee664df4b4e` |
| Raw CSV | `data/train.csv` | `d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1` |
| Feature schema | `data/feature_schema.json` | `a32f3c4491347be51c10a70d00d5bcc24dbb1d1dba1a0105e40d809e8bed51` |
| Split manifest | `data/split_manifest.json` | `10bee51eebac51bf68813a35a42b06adeb2e30588accdc582bbeb9b0d8136c1bb` |
| Preprocessing metadata | `data/preprocessing_artifacts.json` | `df20646f12b77cb9fb03bf831b3373e5f8c9f69d6e5f0c92c35bbd46c682f351` |

The recorded source CSV checksum does **not** match the available `data/train.csv`: `split_manifest.json` records `4baa699e1c09d286f06790010a8c0b284ed3376cbde7e346e19d0ded16a240e3`, while the file at the recorded path hashes to `d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1`. The clean, GBDT-view, and VQC-view digests match the values recorded in the split manifest. The source mismatch is preserved as an additional data-provenance blocker; no files were changed to mask it.

The schema states the outcome is `delivery_duration_min`, in minutes. The split manifest records 29,592 train, 5,989 validation, and 5,941 test records, with `Delivery_person_ID` group separation. The preprocessing JSON records train-derived imputation/scaling constants. These are data/schema artifacts; preprocessing constants are not a fitted GBDT checkpoint.

## Search and blocker evidence

Read-only inspection covered the `trainingv3` worktree, the original sibling checkout, its ignored `.runtime` tree (excluding vendored dependency and unrelated MAMUT source directories), Git branches/history including `origin/codex/india-eta-integration-plan`, and serialized JSON/JSONL/NPY/NPZ/joblib/pickle/PyTorch/ONNX artifact names and contents relevant to India ETA.

- No serialized Indian ETA GBDT checkpoint was found.
- No row-level saved GBDT ETA predictions or out-of-fold predictions were found.
- No fitted model/preprocessor pipeline object was found. The available preprocessing JSON contains constants, not estimator state.
- The ignored `silver-computing-machine/.runtime/run1/**/models/newton_gbdt.json` files are Solomon/Dabia binary route-risk models. They do not predict India delivery duration and cannot serve as the requested model.
- The India ETA planning branch contains a handoff/plan, not the model or predictions.
- Targeted artifact-name search also covered the other Downloads top-level directories and Desktop/Documents; it found no plausible India ETA model or prediction bundle there.
- The local Python environment is 3.12.10 with pandas 3.1.0 but has no PyArrow or fastparquet available for Parquet reading; no dependencies were installed. This does not change the artifact absence finding.

The environment profile reports Windows 11 x64, AMD Ryzen, NumPy 2.5.4, pandas 3.1.0, pytest 9.2.0, and importable `qpiai-quantum==0.3.0`/`qpiai-opt==0.0.9`. PennyLane, PyTorch, scikit-learn, XGBoost, and LightGBM are absent. An NVIDIA GeForce RTX 5060 with 8,151 MiB was detected, but no CUDA-enabled PyTorch capability is installed; one GPU compute PID was not visible to the current process permissions, so active-process visibility is incomplete. No dependencies, GPU jobs, or paid jobs were started.

Machine-readable environment/profile evidence: [`artifacts/run2_india/environment.json`](artifacts/run2_india/environment.json); profile narrative: [`docs/RUN2_ENVIRONMENT.md`](docs/RUN2_ENVIRONMENT.md). QpiAI verification was import-only; it did not run a circuit, authenticate, or submit a job.

The first missing requirement is therefore upstream of quantum execution: there is no frozen predictor to independently reload and no leakage-safe prediction evidence from which to construct residual targets. Training a substitute GBDT, deriving residuals from in-sample predictions, or training on a test-informed reconstruction would breach the mission's constraints. No QML parameters, MAE/RMSE comparison, QpiAI execution, hardware benchmark, or portable model release can honestly be reported.

## Integrity and next input required

The report intentionally contains no surrogate metrics. To resume, supply or point to a trusted frozen-model bundle containing the India ETA estimator, the matching fitted preprocessing object or complete inference mapping, exact feature schema/version, dataset/split identity, and saved leakage-safe train/validation predictions. The sealed test predictions must remain withheld until the residual circuit/fusion selection is frozen. The bundle should include a digest manifest and enough metadata to independently match the 41,522 eligible-row source tables and `delivery_duration_min` target.

## Reproducible preflight

The standard-library-only preflight is [`experiments_india_qml/preflight.py`](experiments_india_qml/preflight.py). From the repository root:

```powershell
& 'C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv\Scripts\python.exe' experiments_india_qml\preflight.py --repo . --search-root ..\silver-computing-machine --json
```

It reports available data hashes, checks the source-data digests against the split manifest, and blocks with exit code 2 until the recorded raw source is reconciled and trusted `--checkpoint` and leakage-safe `--predictions` paths are provided. It never loads candidate files, fits a substitute, or treats route-risk JSON as an ETA model.

Executed from `trainingv3` with `& 'C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv\Scripts\python.exe' experiments_india_qml\preflight.py --repo . --search-root ..\silver-computing-machine --json`: **exit 2 (expected BLOCKED)**; 7/7 declared data files present; 0 candidate serialized ETA artifacts; source CSV digest mismatch; model and prediction artifacts missing. The exit code is deliberately nonzero while required frozen inputs are absent.

Independent acceptance evidence is recorded in [`docs/RUN2_ACCEPTANCE.md`](docs/RUN2_ACCEPTANCE.md). Command: `python -m pytest tests/test_run2_india_qml_acceptance.py -q` from `trainingv3` using the original checkout's Python 3.12.10 venv. Result: **6 passed, 1 skipped, 0 failed**. The skip is the explicit missing frozen GBDT checkpoint gate. The passing checks cover the stated target/units, static feature-list leakage exclusions, manifest-declared group split counts/overlap fields, and file hash checks for the three Parquet tables. Parquet contents and row-level group disjointness were not independently read because PyArrow, fastparquet, Polars, and DuckDB are unavailable. No test-set metrics were computed.

Three Luna workers contributed read-only artifact, environment, and independent acceptance audits; **zero workers trained a model**. Changes from this task are confined to uncommitted files on branch `trainingv3`; the original `main` checkout and protected model/data files were not modified.

## Required summary

| Item | Result |
|---|---|
| Actual Luna training status | Not started; blocked at frozen-model verification |
| Luna audit status | 3 workers audited artifacts/environment/acceptance; 0 training workers |
| Frozen India GBDT available | No |
| Leakage-safe classical residual predictions available | No |
| Classical MAE/RMSE | Not measured in this run |
| Quantum residual MAE/RMSE | Not measured |
| Hybrid MAE/RMSE | Not measured |
| CPU/GPU VQC training runtime | Not run |
| QpiAI execution evidence | Not run |
| Checkpoint/release hashes | None |
| Runtime/test results | Independent acceptance: 6 passed, 1 skipped, 0 failed. Row-level Parquet inspection unavailable (no PyArrow/fastparquet/Polars/DuckDB). Preflight returned expected blocker exit code 2. No runtime dependencies installed. |
| Release recommendation | No release. Keep the stated operational fallback `alpha=0` until frozen inputs and a valid evaluation exist. |

## Luna recovery v5 supplemental result (2026-10-29)

The historical blocker above was resolved under the user's explicit bounded-recovery authorization. A replacement India ETA GBDT, courier-safe OOF residuals, trained one-block and reupload quantum candidates, validation-only selection, portable bundle, and sealed test evaluation are now recorded in [`RUN2_LUNA_RECOVERY_REPORT.md`](RUN2_LUNA_RECOVERY_REPORT.md). The GBDT is a recovered experiment model, not the historical/Gemini/Dell model. The test comparison did not improve over the GBDT baseline, so the operational alpha remains 0. The exact pre-test freeze manifest is preserved at `artifacts/run2_india/luna_hgbr_recovery_20261029T131809Z_v5/release_manifest_pretest.json`; the post-test manifest and metric/prediction evidence are in the same run folder and `.runtime/run2`.
