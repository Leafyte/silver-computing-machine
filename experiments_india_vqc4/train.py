"""Bounded RUN2 training for four-qubit residual ETA circuits.

Only precomputed training OOF residuals and original validation roles are read.
The test partition is not opened by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from experiments_india_vqc4.circuit import FEATURE_NAMES, TorchFourQubitVQC


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "india_eta_recovery"
OUT = ROOT / "artifacts" / "india_eta_vqc4_run2"
FEATURES = list(FEATURE_NAMES)
KEYS = ["delivery_record_id", "courier_id", "split_partition"]
EXPECTED_GBDT_SHA = "f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf"
EXPECTED_OOF_SHA = "78a5461c89387e8630421008b92dfffcc84864e9e25bea7476eefa2c8d48032f"
EXPECTED_VAL_SHA = "3d9485c513e6a997516baea1a4a6c497c6021240f9cbf64d524037e9e8753f1a"
EXPECTED_ROLE_SHA = "631dbb800d8759858ccbed05345d95549940782fa8c81390e9a4affeb0ba6425"
EXPECTED_CONTROL_SHA = "5203fe9844a9721cfed1f26411a32bb0ebd872cafa8f64e661c73c2afc8189e2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_min": float(mean_absolute_error(y, pred)),
        "rmse_min": float(math.sqrt(mean_squared_error(y, pred))),
        "median_absolute_error_min": float(median_absolute_error(y, pred)),
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    oof_path = DATA / "oof_predictions.parquet"
    val_path = DATA / "validation_predictions.parquet"
    recovery = json.loads((DATA / "recovery_manifest.json").read_text(encoding="utf-8"))
    if sha256(DATA / "gbdt" / "india_eta_hgb_regressor.joblib") != EXPECTED_GBDT_SHA:
        raise RuntimeError("frozen GBDT checkpoint hash changed")
    if sha256(oof_path) != EXPECTED_OOF_SHA:
        raise RuntimeError("frozen OOF prediction hash changed")
    if sha256(val_path) != EXPECTED_VAL_SHA:
        raise RuntimeError("frozen validation prediction hash changed")
    if recovery["validation_role_assignment_sha256"] != EXPECTED_ROLE_SHA:
        raise RuntimeError("validation role assignment hash changed")
    oof, val = pd.read_parquet(oof_path), pd.read_parquet(val_path)
    for name, frame, split in (("OOF", oof, "train"), ("validation", val, "val")):
        if frame.duplicated(KEYS).any() or not frame["split_partition"].eq(split).all():
            raise RuntimeError(f"{name} prediction join keys/split are invalid")
        if frame["delivery_record_id"].duplicated().any():
            raise RuntimeError(f"{name} delivery_record_id is not unique")
        if not np.isfinite(frame[FEATURES + ["y_true_min", "gbdt_pred_min"]].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"{name} contains non-finite model inputs/targets")
    if len(oof) != 29592 or len(val) != 5989:
        raise RuntimeError("frozen split row counts changed")
    if set(oof.delivery_record_id).intersection(val.delivery_record_id):
        raise RuntimeError("training and validation delivery_record_id values overlap")
    if set(oof.courier_id).intersection(val.courier_id):
        raise RuntimeError("training and validation courier groups overlap")
    if set(val.validation_role.unique()) != {"model_selection", "fusion_calibration"}:
        raise RuntimeError("validation role labels changed")
    if (val.groupby("courier_id").validation_role.nunique() != 1).any():
        raise RuntimeError("validation courier appears in both selection and calibration roles")
    role_couriers = val.groupby("validation_role").courier_id.nunique().to_dict()
    if role_couriers != {"fusion_calibration": 58, "model_selection": 117}:
        raise RuntimeError("validation courier role group counts changed")
    x_train = oof[FEATURES].to_numpy(dtype=np.float32)
    residual = (oof["y_true_min"] - oof["gbdt_pred_min"]).to_numpy(dtype=np.float32)
    scale = max(float(np.std(residual, ddof=0)), 1.0)
    val_selection = val["validation_role"].eq("model_selection").to_numpy()
    if int(val_selection.sum()) != 4124 or int((~val_selection).sum()) != 1865:
        raise RuntimeError("frozen validation selection/calibration role counts changed")
    x_val = val.loc[val_selection, FEATURES].to_numpy(dtype=np.float32)
    r_val = (val.loc[val_selection, "y_true_min"] - val.loc[val_selection, "gbdt_pred_min"]).to_numpy(dtype=np.float32)
    return oof, val, x_train, residual, x_val, r_val


def run_stage(model: TorchFourQubitVQC, stage: str, x: np.ndarray, y: np.ndarray,
              xv: np.ndarray, yv: np.ndarray, base_val: np.ndarray, *, epochs: int, batch_size: int,
              lr: float, weight_decay: float, patience: int, device: str,
              seed: int, log_path: Path, ckpt_path: Path, resume: bool = False) -> tuple[TorchFourQubitVQC, dict]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    xvt = torch.as_tensor(xv, dtype=torch.float32, device=device)
    yvt = torch.as_tensor(yv, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = float("inf")
    best_payload = None
    best_epoch = 0
    stale = 0
    epoch_rows = []
    resume_path = ckpt_path.with_name(ckpt_path.stem + "_latest_resume.pt")
    start_epoch = 1
    if resume and resume_path.exists():
        snap = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_payload(snap["model_payload"])
        opt.load_state_dict(snap["optimizer_state"])
        generator.set_state(snap["shuffle_generator_state"].cpu())
        start_epoch = int(snap["epoch"]) + 1
        best = float(snap["best_mae"])
        best_epoch = int(snap["best_epoch"])
        stale = int(snap["stale_epochs"])
        best_payload = snap["best_payload"]
        if stale >= patience:
            start_epoch = epochs + 1
        print(json.dumps({"event": "resume", "stage": stage, "start_epoch": start_epoch,
                          "best_epoch": best_epoch}), flush=True)
    for epoch in range(start_epoch, epochs + 1):
        tic = time.perf_counter()
        model.train()
        order = torch.randperm(len(x), generator=generator)
        loss_sum = 0.0
        seen = 0
        step_count = 0
        for batch_ids in order.split(batch_size):
            ids = batch_ids.to(device)
            pred = model(xt.index_select(0, ids))
            loss = torch.mean((pred - yt.index_select(0, ids)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at {stage} epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n = int(ids.numel())
            loss_sum += float(loss.detach().cpu()) * n
            seen += n
            step_count += 1
        model.eval()
        with torch.no_grad():
            pv = model(xvt).detach().cpu().numpy().astype(np.float64)
        base64 = base_val.astype(np.float64)
        truth64 = base64 + yv.astype(np.float64)
        val_metrics = metrics(truth64, np.maximum(0.0, base64 + pv))
        row = {
            "stage": stage, "epoch": epoch, "train_mse_min2": loss_sum / seen,
            "selection": val_metrics, "optimizer_steps": step_count,
            "epoch_seconds": time.perf_counter() - tic,
            "device": device, "parameters": model.parameter_count,
        }
        epoch_rows.append(row)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
        latest_path = ckpt_path.with_name(ckpt_path.stem + "_latest.json")
        write_json_atomic(latest_path, model.export_payload())
        if val_metrics["mae_min"] < best:
            best, best_epoch, stale = val_metrics["mae_min"], epoch, 0
            best_payload = model.export_payload()
            write_json_atomic(ckpt_path, best_payload)
        else:
            stale += 1
        snapshot_tmp = resume_path.with_suffix(resume_path.suffix + ".tmp")
        torch.save({"stage": stage, "epoch": epoch, "model_payload": model.export_payload(),
                    "optimizer_state": opt.state_dict(), "shuffle_generator_state": generator.get_state(),
                    "best_mae": best, "best_epoch": best_epoch, "stale_epochs": stale,
                    "best_payload": best_payload}, snapshot_tmp)
        snapshot_tmp.replace(resume_path)
        print(json.dumps({"progress": stage, **row, "best_epoch": best_epoch, "stale": stale}), flush=True)
        if stale >= patience:
            break
    if best_payload is None and start_epoch > epochs:
        best_payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
        best = float("inf") if not math.isfinite(best) else best
    if best_payload is None:
        raise RuntimeError(f"no checkpoint produced for {stage}")
    model.load_payload(best_payload)
    return model, {"best_epoch": best_epoch, "best_selection_mae_min": best, "epochs_run": len(epoch_rows),
                   "optimizer_steps": sum(r["optimizer_steps"] for r in epoch_rows),
                   "checkpoint_sha256": sha256(ckpt_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs60", type=int, default=30)
    ap.add_argument("--epochs76", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--lr60", type=float, default=0.01)
    ap.add_argument("--lr76", type=float, default=0.002)
    ap.add_argument("--resume", action="store_true", help="resume stage from the latest atomic epoch snapshot")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch reports it unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(os.cpu_count() or 1, 8))
    torch.use_deterministic_algorithms(True)
    oof, val, x, residual, xv, r_val = load_data()
    scale = max(float(np.std(residual.astype(np.float64), ddof=0)), 1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    run_start = time.time()
    log_path = OUT / "training_progress.jsonl"
    if log_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing training log: {log_path}")
    stage60_path = OUT / "baseline60_best.json"
    stage76_path = OUT / "extended76_best.json"
    print(json.dumps({"event": "start", "device": args.device, "torch": torch.__version__,
                      "cuda_available": torch.cuda.is_available(), "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "rows_train": len(oof), "rows_selection": len(xv), "rows_calibration_held": int((val.validation_role == "fusion_calibration").sum()),
                      "residual_scale_minutes": scale, "batch_size": args.batch_size, "out": str(OUT)}), flush=True)
    base = TorchFourQubitVQC("baseline60", residual_scale_minutes=scale, seed=args.seed, dtype="float32", device=args.device)
    selection_base = val.loc[val.validation_role.eq("model_selection"), "gbdt_pred_min"].to_numpy(dtype=np.float64)
    base, res60 = run_stage(base, "baseline60", x, residual, xv, r_val, epochs=args.epochs60,
                            base_val=selection_base,
                            batch_size=args.batch_size, lr=args.lr60, weight_decay=1e-4,
                            patience=args.patience, device=args.device, seed=args.seed + 1,
                            log_path=log_path, ckpt_path=stage60_path, resume=args.resume)
    extended = TorchFourQubitVQC.expand_baseline(base, device=args.device)
    extended, res76 = run_stage(extended, "extended76", x, residual, xv, r_val, epochs=args.epochs76,
                                base_val=selection_base,
                                batch_size=args.batch_size, lr=args.lr76, weight_decay=1e-4,
                                patience=args.patience, device=args.device, seed=args.seed + 2,
                                log_path=log_path, ckpt_path=stage76_path, resume=args.resume)
    # Freeze model choices before evaluating the separate calibration couriers.
    cal = val["validation_role"].eq("fusion_calibration").to_numpy()
    selection_mask = val["validation_role"].eq("model_selection").to_numpy()
    x_selection = torch.as_tensor(val.loc[selection_mask, FEATURES].to_numpy(dtype=np.float32), device=args.device)
    x_calibration = torch.as_tensor(val.loc[cal, FEATURES].to_numpy(dtype=np.float32), device=args.device)
    gbdt_cal = val.loc[cal, "gbdt_pred_min"].to_numpy(dtype=np.float64)
    truth_cal = val.loc[cal, "y_true_min"].to_numpy(dtype=np.float64)
    models = {"baseline60": (base, stage60_path), "extended76": (extended, stage76_path)}
    val_out = val[KEYS + ["validation_role", *FEATURES, "y_true_min", "gbdt_pred_min"]].copy()
    selection_corrections = {}
    result = {
        "format": "india_eta_vqc4_run2_training_v1", "status": "frozen_models_selection_only_no_test_read",
        "run_started_unix": run_start, "run_finished_unix": time.time(),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": args.device, "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "thread_count": torch.get_num_threads(), "seed": args.seed,
        "training_config": {"optimizer": "AdamW", "loss": "MSE of minute residuals",
                            "batch_size": args.batch_size, "epochs60_max": args.epochs60,
                            "epochs76_max": args.epochs76, "patience": args.patience,
                            "lr60": args.lr60, "lr76": args.lr76, "weight_decay": 1e-4,
                            "resume_requested": args.resume},
        "train_rows": len(oof), "selection_rows": len(xv), "calibration_rows_held_until_freeze": int(cal.sum()),
        "feature_names": FEATURES, "input_scale": "committed VQC features, unchanged in [0,1]",
        "target": "y_true_min - gbdt_pred_min; minutes",
        "residual_scale_minutes": scale, "scale_fit": "population std of training OOF residual only; floor 1.0",
        "source_hashes": {"gbdt_checkpoint": EXPECTED_GBDT_SHA, "oof_predictions": EXPECTED_OOF_SHA,
                          "validation_predictions": EXPECTED_VAL_SHA,
                          "validation_role_assignment": EXPECTED_ROLE_SHA,
                          "ridge_control": EXPECTED_CONTROL_SHA},
        "role_algorithm": json.loads((DATA / "recovery_manifest.json").read_text(encoding="utf-8"))["validation_role_algorithm"],
        "stages": {"baseline60": res60, "extended76": res76},
        "selection_choice": None,
        "calibration_metrics_after_freeze": {},
        "selection_metrics": {"gbdt_only": metrics(
            val.loc[selection_mask, "y_true_min"].to_numpy(dtype=np.float64),
            np.maximum(0.0, selection_base))},
    }
    control_path = ROOT / "artifacts" / "india_eta_residual_control_v1" / "residual_ridge.json"
    if sha256(control_path) != EXPECTED_CONTROL_SHA:
        raise RuntimeError("frozen classical residual control hash changed")
    control = json.loads(control_path.read_text(encoding="utf-8"))
    coefficients = np.asarray(control["coefficients_min_per_unit"], dtype=np.float64)
    intercept = float(control["intercept_min"])
    selection_x_np = val.loc[selection_mask, FEATURES].to_numpy(dtype=np.float64)
    control_sel = selection_x_np @ coefficients + intercept
    control_truth = val.loc[selection_mask, "y_true_min"].to_numpy(dtype=np.float64)
    control_base = val.loc[selection_mask, "gbdt_pred_min"].to_numpy(dtype=np.float64)
    result["selection_metrics"]["ridge_control"] = metrics(control_truth, np.maximum(0.0, control_base + control_sel))
    # Compare model checkpoints on model-selection couriers only; calibration remains unopened.
    baseline_sel = result["selection_metrics"]["gbdt_only"]
    for name, (_, path) in models.items():
        model = models[name][0]
        model.eval()
        with torch.no_grad():
            correction = model(x_selection).detach().cpu().numpy().astype(np.float64)
        selection_corrections[name] = correction
        pred = np.maximum(0.0, val.loc[selection_mask, "gbdt_pred_min"].to_numpy(dtype=np.float64) + correction)
        truth = val_out.loc[selection_mask, "y_true_min"].to_numpy(dtype=np.float64)
        result["selection_metrics"][name] = metrics(truth, pred)
        result.setdefault("checkpoint_paths", {})[name] = path.name
    ranked = sorted((k for k in models), key=lambda k: result["selection_metrics"][k]["mae_min"])
    best_name = ranked[0]
    control_sel_metrics = result["selection_metrics"]["ridge_control"]
    if (result["selection_metrics"][best_name]["mae_min"] < baseline_sel["mae_min"]
            and result["selection_metrics"][best_name]["rmse_min"] <= baseline_sel["rmse_min"]
            and result["selection_metrics"][best_name]["mae_min"] < control_sel_metrics["mae_min"]
            and result["selection_metrics"][best_name]["rmse_min"] <= control_sel_metrics["rmse_min"]):
        result["selection_choice"] = best_name
        result["selected_alpha"] = 1.0
    else:
        result["selection_choice"] = "alpha0_gbdt_only"
        result["selected_alpha"] = 0.0
    # The model choice is frozen from selection-only metrics. Now evaluate the reserved
    # courier groups and write complete validation predictions for independent review.
    result["calibration_metrics_after_freeze"] = {
        "gbdt_only": metrics(truth_cal, np.maximum(0.0, gbdt_cal)),
        "ridge_control": metrics(truth_cal, np.maximum(0.0, gbdt_cal + val.loc[cal, FEATURES].to_numpy(dtype=np.float64) @ coefficients + intercept)),
    }
    for name, (model, _) in models.items():
        model.eval()
        with torch.no_grad():
            correction_cal = model(x_calibration).detach().cpu().numpy().astype(np.float64)
        correction_all = np.empty(len(val_out), dtype=np.float64)
        correction_all[selection_mask] = selection_corrections[name]
        correction_all[cal] = correction_cal
        val_out[f"{name}_correction_min"] = correction_all
        result["calibration_metrics_after_freeze"][name] = metrics(truth_cal, np.maximum(0.0, gbdt_cal + correction_cal))
        val_out[f"{name}_hybrid_eta_min"] = np.maximum(0.0, val["gbdt_pred_min"].to_numpy(dtype=np.float64) + correction_all)
    val_out.to_parquet(OUT / "validation_predictions.parquet", index=False)
    result["validation_predictions_sha256"] = sha256(OUT / "validation_predictions.parquet")
    result["training_log_sha256"] = sha256(log_path)
    result["training_code_sha256"] = sha256(Path(__file__))
    result["circuit_code_sha256"] = sha256(Path(__file__).with_name("circuit.py"))
    backend_profile = ROOT / ".runtime" / "run2" / "backend_profile.json"
    result["backend_profile_path"] = str(backend_profile.relative_to(ROOT))
    result["backend_profile_sha256"] = sha256(backend_profile)
    result["backend_profile"] = json.loads(backend_profile.read_text(encoding="utf-8"))
    result["test_partition"] = "not read, scored, or used; final evaluation is separate after freeze"
    (OUT / "run_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "frozen", "choice": result["selection_choice"], "selection": result["selection_metrics"],
                      "calibration": result["calibration_metrics_after_freeze"], "manifest": str(OUT / "run_manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
