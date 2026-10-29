"""Reproduce frozen RUN2 VQC checkpoints while capturing process/gradient metrics.

This uses the original train.run_stage implementation, fixed inputs, fixed seed,
and frozen hyperparameters. It never selects a new model or reads test data.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiments_india_vqc4 import train
from experiments_india_vqc4.circuit import TorchFourQubitVQC


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RUN = ROOT / "artifacts" / "india_eta_vqc4_run2"
ORIGINAL_TRAIN_SHA = "88e0506c94d86c6650270599f4133488b6e2c5059a307bd34859a2c0aee56154"
ORIGINAL_CIRCUIT_SHA = "801468f544e25ae913c26e1f2131cf5f88b384b8495172da62be9051d5a96ba0"
ORIGINAL_CHECKPOINT_SHA = {
    "baseline60": "6a6458d42ac9561845bd606543f9017f997a01ae31b4afa75a4f554d0d30cffe",
    "extended76": "002626418344b9650a24da3ca02309acc743d9fc57a166899941c1f0d216ec14",
}
SEED = 20261029
BATCH_SIZE = 128
PATIENCE = 5
MAX_EPOCHS = {"baseline60": 30, "extended76": 25}
LEARNING_RATES = {"baseline60": 0.01, "extended76": 0.002}
WEIGHT_DECAY = 1e-4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def win_memory() -> dict[str, int | None]:
    class CountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    if os.name != "nt":
        return {"peak_working_set_bytes": None, "current_working_set_bytes": None,
                "peak_private_bytes": None, "current_private_bytes": None}
    counters = CountersEx()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE,
                                           ctypes.POINTER(CountersEx), ctypes.wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return {"peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            "current_working_set_bytes": int(counters.WorkingSetSize),
            "peak_private_bytes": int(counters.PeakPagefileUsage),
            "current_private_bytes": int(counters.PrivateUsage)}


def atomic_json(path: Path, obj: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite reproduction output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    source_train = ROOT / "experiments_india_vqc4" / "train.py"
    source_circuit = ROOT / "experiments_india_vqc4" / "circuit.py"
    original_archive = ROOT / ".runtime" / "run2" / "repro_original" / "train.py"
    if sha256(source_train) != ORIGINAL_TRAIN_SHA or sha256(original_archive) != ORIGINAL_TRAIN_SHA:
        raise RuntimeError("source trainer no longer equals the archived frozen trainer bytes")
    if sha256(source_circuit) != ORIGINAL_CIRCUIT_SHA:
        raise RuntimeError("circuit implementation changed since the frozen run")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(min(os.cpu_count() or 1, 8))
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    device = "cpu"
    oof, val, x, residual, xv, r_val = train.load_data()
    selection_mask = val["validation_role"].eq("model_selection").to_numpy()
    base_val = val.loc[selection_mask, "gbdt_pred_min"].to_numpy(dtype=np.float64)
    residual_scale = max(float(np.std(residual.astype(np.float64), ddof=0)), 1.0)
    log_path = out / "training_progress.jsonl"

    # Record counts of actual model forwards and batch rows via transparent hooks.
    forward_counters: dict[str, dict[str, int]] = {}
    grad_stats: dict[str, dict] = {}
    original_step = torch.optim.AdamW.step
    active: dict[str, object] = {}

    def measured_step(optimizer, *args, **kwargs):
        model = active.get("model")
        key = str(active.get("stage"))
        if model is not None:
            grads = [p.grad.detach().reshape(-1) for group in optimizer.param_groups
                     for p in group["params"] if p.grad is not None]
            if grads:
                norm = float(torch.cat(grads).norm().item())
                stats = grad_stats[key]
                stats["gradient_l2_by_step_min"] = min(stats["gradient_l2_by_step_min"], norm)
                stats["gradient_l2_by_step_max"] = max(stats["gradient_l2_by_step_max"], norm)
                stats["gradient_l2_by_step_sum"] += norm
                stats["gradient_steps"] += 1
                if key == "extended76":
                    ext = model.quantum_angles.grad[60:].detach()
                    ext_l2 = float(ext.norm().item())
                    ext_abs = ext.abs()
                    stats["extension16_gradient_l2_first_step"] = stats.get("extension16_gradient_l2_first_step", ext_l2)
                    stats["extension16_gradient_l2_by_step_min"] = min(stats["extension16_gradient_l2_by_step_min"], ext_l2)
                    stats["extension16_gradient_l2_by_step_max"] = max(stats["extension16_gradient_l2_by_step_max"], ext_l2)
                    stats["extension16_gradient_l2_by_step_sum"] += ext_l2
                    stats["extension16_gradient_abs_component_min"] = min(
                        stats["extension16_gradient_abs_component_min"], float(ext_abs.min().item()))
                    stats["extension16_gradient_abs_component_max"] = max(
                        stats["extension16_gradient_abs_component_max"], float(ext_abs.max().item()))
        return original_step(optimizer, *args, **kwargs)

    torch.optim.AdamW.step = measured_step
    stages = {}
    memory_snapshots = {"before_training": win_memory()}
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    stage_cpu_seconds = {}

    def instrument_and_run(model: TorchFourQubitVQC, stage: str, **kwargs):
        counters = {"forward_calls": 0, "forward_input_rows": 0, "statevector_amplitude_elements": 0}
        forward_counters[stage] = counters
        grad_stats[stage] = {"gradient_l2_by_step_min": float("inf"), "gradient_l2_by_step_max": 0.0,
                             "gradient_l2_by_step_sum": 0.0, "gradient_steps": 0,
                             "initial_quantum_angle_l2": float(model.quantum_angles.detach().norm().cpu()),
                             "initial_readout_l2": float(model.readout.detach().norm().cpu())}
        if stage == "extended76":
            ext = model.quantum_angles.detach()[60:]
            grad_stats[stage].update({"extension16_initial_l2": float(ext.norm().cpu()),
                                      "extension16_initial_abs_component_min": float(ext.abs().min().cpu()),
                                      "extension16_initial_abs_component_max": float(ext.abs().max().cpu()),
                                      "extension16_gradient_l2_by_step_min": float("inf"),
                                      "extension16_gradient_l2_by_step_max": 0.0,
                                      "extension16_gradient_l2_by_step_sum": 0.0,
                                      "extension16_gradient_abs_component_min": float("inf"),
                                      "extension16_gradient_abs_component_max": 0.0})

        def count_forward(_module, inputs, _output):
            rows = int(inputs[0].shape[0])
            counters["forward_calls"] += 1
            counters["forward_input_rows"] += rows
            counters["statevector_amplitude_elements"] += rows * 16

        hook = model.register_forward_hook(count_forward)
        active.update({"stage": stage, "model": model})
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        try:
            result = train.run_stage(model, stage, x, residual, xv, r_val, base_val=base_val,
                                     epochs=MAX_EPOCHS[stage], batch_size=BATCH_SIZE,
                                     lr=LEARNING_RATES[stage], weight_decay=WEIGHT_DECAY,
                                     patience=PATIENCE, device=device,
                                     seed=SEED + (1 if stage == "baseline60" else 2),
                                     log_path=log_path, ckpt_path=out / f"{stage}_best.json", resume=False)
        finally:
            hook.remove()
            active.clear()
        stage_cpu_seconds[stage] = time.process_time() - start_cpu
        counters["processed_statevector_rows"] = counters["forward_input_rows"]
        stats = grad_stats[stage]
        steps = stats.pop("gradient_steps")
        stats["gradient_l2_by_step_mean"] = stats.pop("gradient_l2_by_step_sum") / steps if steps else None
        if not steps:
            stats["gradient_l2_by_step_min"] = None
        if stage == "extended76" and steps:
            stats["extension16_gradient_l2_by_step_mean"] = stats.pop("extension16_gradient_l2_by_step_sum") / steps
            stats["extension16_gradient_abs_component_min"] = (None if stats["extension16_gradient_abs_component_min"] == float("inf") else stats["extension16_gradient_abs_component_min"])
            stats["extension16_gradient_l2_by_step_min"] = (None if stats["extension16_gradient_l2_by_step_min"] == float("inf") else stats["extension16_gradient_l2_by_step_min"])
        stages[stage] = result[1]
        memory_snapshots[f"after_{stage}"] = win_memory()
        return result[0]

    try:
        baseline = TorchFourQubitVQC("baseline60", residual_scale_minutes=residual_scale,
                                    seed=SEED, dtype="float32", device=device)
        baseline = instrument_and_run(baseline, "baseline60")
        baseline_sha = sha256(out / "baseline60_best.json")
        if baseline_sha != ORIGINAL_CHECKPOINT_SHA["baseline60"]:
            status = "baseline_checkpoint_mismatch_stop_before_extended_stage"
            atomic_json(out / "reproduction_metrics.json", {
                "status": status, "baseline_reproduced_sha256": baseline_sha,
                "baseline_expected_sha256": ORIGINAL_CHECKPOINT_SHA["baseline60"],
                "gradient_stats": grad_stats, "forward_counts": forward_counters,
                "stage_cpu_seconds": stage_cpu_seconds, "memory_snapshots": memory_snapshots,
                "original_train_sha256": ORIGINAL_TRAIN_SHA,
                "reproduction_code_sha256": sha256(Path(__file__)),
            })
            print(json.dumps({"status": status, "baseline_sha": baseline_sha,
                              "expected": ORIGINAL_CHECKPOINT_SHA["baseline60"]}), flush=True)
            return
        extended = TorchFourQubitVQC.expand_baseline(baseline, device=device)
        extended = instrument_and_run(extended, "extended76")
        extended_sha = sha256(out / "extended76_best.json")
        if extended_sha != ORIGINAL_CHECKPOINT_SHA["extended76"]:
            status = "extended_checkpoint_mismatch_do_not_replace_or_select"
        else:
            status = "both_frozen_checkpoints_reproduced_bitwise"
        total_cpu = time.process_time() - cpu_start
        total_wall = time.perf_counter() - wall_start
        memory_snapshots["after_training"] = win_memory()
        summary = {
            "format": "india_eta_vqc4_reproduction_resource_metrics_v1",
            "status": status,
            "reproduction_only": True,
            "selection_or_model_tuning_performed": False,
            "test_partition_read": False,
            "device": "cpu",
            "torch": torch.__version__,
            "python": platform.python_version(),
            "cpu_threads": torch.get_num_threads(),
            "architecture_and_training": {"seed": SEED, "batch_size": BATCH_SIZE,
                "epochs_max": MAX_EPOCHS, "patience": PATIENCE, "optimizer": "AdamW",
                "learning_rates": LEARNING_RATES, "weight_decay": WEIGHT_DECAY,
                "training_row_order": "same deterministic torch.randperm generator reset per stage as original trainer",
                "stage_order": ["baseline60", "extended76 exact-zero-extension warm start"],
                "training_rows": len(oof), "selection_rows": int(selection_mask.sum()),
                "calibration_rows_not_opened": int((val.validation_role == "fusion_calibration").sum())},
            "source_hashes": {"oof": train.EXPECTED_OOF_SHA, "validation": train.EXPECTED_VAL_SHA,
                              "gbdt": train.EXPECTED_GBDT_SHA, "ridge_control": train.EXPECTED_CONTROL_SHA,
                              "role_map": train.EXPECTED_ROLE_SHA},
            "original_run_manifest_sha256": sha256(ORIGINAL_RUN / "run_manifest.json"),
            "original_train_code_sha256": ORIGINAL_TRAIN_SHA,
            "archived_original_train_sha256": sha256(original_archive),
            "reproduction_code_sha256": sha256(Path(__file__)),
            "circuit_code_sha256": sha256(source_circuit),
            "expected_checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA,
            "reproduced_checkpoint_sha256": {"baseline60": baseline_sha, "extended76": extended_sha},
            "checkpoint_match": {"baseline60": baseline_sha == ORIGINAL_CHECKPOINT_SHA["baseline60"],
                                 "extended76": extended_sha == ORIGINAL_CHECKPOINT_SHA["extended76"]},
            "stage_metrics": stages,
            "forward_counts": forward_counters,
            "gradient_stats": grad_stats,
            "cpu_process_seconds": {"baseline60": stage_cpu_seconds["baseline60"],
                                     "extended76": stage_cpu_seconds.get("extended76"), "total": total_cpu},
            "wall_seconds": {"baseline60": None, "extended76": None, "total": total_wall},
            "windows_memory_snapshots": memory_snapshots,
            "process_peak_rss_bytes": memory_snapshots["after_training"]["peak_working_set_bytes"],
            "model_evaluation": "only deterministic model-selection early stopping within exact checkpoint reproduction; no calibration or test evaluation",
        }
        # Stage wall times come from the original per-epoch records, without re-timing model kernels.
        stage_epochs = {}
        for line in log_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            stage_epochs.setdefault(entry["stage"], []).append(entry)
        summary["wall_seconds"] = {
            "baseline60": float(sum(e["epoch_seconds"] for e in stage_epochs["baseline60"])),
            "extended76": float(sum(e["epoch_seconds"] for e in stage_epochs["extended76"])),
            "total": total_wall,
        }
        summary["total_optimizer_steps"] = int(sum(v["optimizer_steps"] for v in stages.values()))
        summary["forward_counts"]["total"] = {
            k: sum(v[k] for v in forward_counters.values())
            for k in ("forward_calls", "forward_input_rows", "statevector_amplitude_elements")
        }
        summary["progress_log_sha256"] = sha256(log_path)
        atomic_json(out / "reproduction_metrics.json", summary)
        print(json.dumps({"status": status, "reproduced_checkpoint_sha256": summary["reproduced_checkpoint_sha256"],
                          "cpu_process_seconds": summary["cpu_process_seconds"],
                          "peak_rss_bytes": summary["process_peak_rss_bytes"],
                          "forward_counts": summary["forward_counts"]["total"]}), flush=True)
    finally:
        torch.optim.AdamW.step = original_step


if __name__ == "__main__":
    main()
