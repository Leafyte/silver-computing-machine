# RUN2 India QML environment profile

## Current RUN2 hold update (2026-10-29)

Worker A has since reconstructed and frozen the classical GBDT and courier-safe predictions. Available artifacts are under `artifacts/india_eta_recovery/`: 29,592 train OOF rows and 5,989 validation rows, grouped by courier. Validation has 4,124 model-selection rows and 1,865 fusion-calibration rows across 117 and 58 disjoint couriers. The GBDT checkpoint SHA-256 is `f1f94fbcc932f3519aaafcff7dc99d65b7951c0240730e54c320259152bbebdf`. The initial environment capture below predates this recovery and describes the earlier missing-checkpoint state.

The isolated scratch runtime `trainingv3/.runtime/run2/venv` now has NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.4.2, and fastparquet 2024.11.0. It has no PyTorch. A Torch 2.14.0 CUDA 13.0 wheel download was stopped before install at the explicit resource gate; no training updates were run. Do not infer CUDA support from GPU visibility: an actual two-qubit Numba kernel probe failed with `NvvmSupportError` because `nvvm.dll` could not be found, and Numba reported `cuda.is_available() == False`.

Cheap circuit checks are summarized in [`RUN2_OPUS_ARCHITECTURE_REVIEW.md`](RUN2_OPUS_ARCHITECTURE_REVIEW.md): the existing NumPy models passed parameter-shift versus finite-difference checks and QpiAI local statevector parity; the portable four-feature extension passed its local QpiAI parity test. These do not establish Torch/autograd correctness, trained accuracy, or GPU runtime. VQC training remains on hold pending architecture review.

Captured 2026-10-29 on the Lenovo workstation. Machine-readable evidence is in [`artifacts/run2_india/environment.json`](../artifacts/run2_india/environment.json); regenerate it with the command below.

## Runtime

- Host: Windows 11 x64, AMD64 Family 26 Model 68 (AMD Ryzen), Python 3.12.10.
- Original checkout: `C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine`.
- Its `.venv` is a directory junction resolving to `C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine-main\.venv`.
- Use the venv interpreter directly because the `python.exe` and `py.exe` WindowsApps aliases fail to launch:

  ```powershell
  & 'C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine\.venv\Scripts\python.exe' tools/run2_profile_environment.py --original-checkout '..\silver-computing-machine' --output 'artifacts\run2_india\environment.json'
  ```

- Installed versions: `qpiai-quantum==0.3.0`, `qpiai-opt==0.0.9`, NumPy 2.5.4, pandas 3.1.0, pytest 9.2.0, Numba 0.68.1. Both QpiAI modules imported successfully. PyTorch, PennyLane, scikit-learn, XGBoost, and LightGBM are absent from this venv.
- The QpiAI check was package metadata plus Python import only. It did not execute a quantum circuit, create a cloud client, authenticate, or submit a job.

## GPU

- Device: NVIDIA GeForce RTX 5060, 8,151 MiB reported VRAM; driver 610.88, CUDA UMD 13.3.
- `nvidia-smi` reported 0% utilization, 246 MiB memory used, and no process in its displayed process table.
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits` returned PID 44700 with `[Insufficient Permissions]` and `[N/A]`; `Get-Process -Id 44700` could not resolve it. Process visibility is therefore incomplete.
- CUDA capability was probed through Numba rather than inferred from the driver: Numba 0.68.1 imports in the recorded probe, but `numba.cuda.is_available()` is `false`; no CUDA device is available to Numba and no CUDA kernel was executed. A preceding standalone import attempt failed while loading `_extras` with an Application Control DLL-block message; the failure did not reproduce in the later captured probe. PyTorch, CuPy, JAX, TensorFlow, and cuda-python are absent; `nvcc` is unavailable. Thus CUDA execution is unavailable through the profiled Python runtime, although Windows and `nvidia-smi` can see the GPU.

## RUN2 workload availability

The inspected original checkout and this trainingv3 checkout contain India datasets, schema, and preprocessing/split metadata but no frozen India ETA model checkpoint or leakage-safe classical prediction artifact. The original model artifact search included pickle/joblib, PyTorch, ONNX, CatBoost, NumPy, safetensors, and model JSON formats under data, artifacts, models, checkpoints, experiments, and `.runtime/run2_india`. The found `indian_delivery_{clean,gbdt,vqc}.parquet` files are feature data, not model checkpoints.

At initial environment capture, the India checkpoint and prediction artifacts were absent. No CPU-versus-CUDA timing or VRAM measurement for the actual India workload is available. The environment profiler records the initial state; the separate kernel-only profile below is not a replacement model measurement.

## Quantum kernel throughput-only profile

After the quantum kernel implementation became available, a separate synthetic throughput-only profile was run and saved to [`artifacts/run2_india/quantum_profile_synthetic.json`](../artifacts/run2_india/quantum_profile_synthetic.json). It times the MSE plus parameter-shift gradient kernel (no optimizer updates) on a deterministic 512-row numeric fixture, 256 repetitions at each batch size. It is not India ETA model throughput or accuracy evidence.

| Circuit | Batch 128 | Batch 256 | Batch 512 |
|---|---:|---:|---:|
| One block | 256,346 examples/s | 406,812 examples/s | 739,569 examples/s |
| Two-stage reupload | 88,142 examples/s | 153,009 examples/s | 214,512 examples/s |

These are wall-clock examples processed through one MSE-gradient step per repetition. The profile also records batched circuit evaluations, CPU time, and finite-difference/QpiAI parity. CUDA profile values are unavailable because Numba reports `cuda.is_available() == false`; driver visibility alone does not establish a usable CUDA backend.

## Probe commands

```powershell
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
& 'C:\Users\avkir\Downloads\silver-computing-machine-main\silver-computing-machine\.venv\Scripts\python.exe' tools/run2_profile_environment.py --original-checkout '..\silver-computing-machine' --output 'artifacts\run2_india\environment.json'
```

No dependencies were installed, no serialized model was loaded for the CPU kernel profile, no India training workload was run by this worker, and no credential values were read or recorded. The profile below measures only the synthetic numerical kernel described above.
