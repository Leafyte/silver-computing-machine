# RUN2 VQC Architecture Review Handoff

**Purpose:** compare the existing two-feature residual circuit with the proposed four-feature reuploading extension. This is an architecture review artifact, not an approval or a training result. No VQC training updates have been run.

## Shared model contract

- Target: residual minutes, `delivery_duration_min - gbdt_pred_min`.
- Training labels must use Worker A's courier-grouped OOF predictions. The current frozen source has 29,592 OOF rows; validation has 5,989 rows split into 4,124 model-selection rows and 1,865 fusion-calibration rows by courier.
- Inputs are precomputed numeric features in `[0, 1]`; a feature is encoded as `RY(pi * x)` in radians. Each trainable angle `theta` is also in radians.
- Both circuits use two qubits, computational-basis statevector simulation, the pair of observables `Z(q0)` and `Z(q1)`, and a three-parameter affine readout (two expectation weights and bias). The readout output is multiplied by a positive residual scale in minutes fitted on train OOF labels only.
- The established `experiments_india_qml/quantum.py` implementation remains the reference and is unchanged. The Torch and NumPy portable four-feature implementations are candidate extensions awaiting review.

## Circuit inventory

| Property | Existing one-block baseline | Proposed two-stage four-feature extension |
|---|---|---|
| Ordered input features | `x0` distance, `x1` congestion | `x0` distance, `x1` congestion, `x2` multi-delivery, `x3` preparation time |
| Gate sequence | `RY(pi*x0) q0`; `RY(pi*x1) q1`; `CX(0,1)`; `RY(theta0) q0`; `RY(theta1) q1`; `CX(1,0)` | Initial `RY(pi*x0) q0`, `RY(pi*x1) q1`; `RY(theta0) q0`, `RY(theta1) q1`; `CX(0,1)`; reupload `RY(pi*x2) q0`, `RY(pi*x3) q1`; `RY(theta2) q0`, `RY(theta3) q1`; `CX(1,0)` |
| Trainable circuit angles | 2, independent | 4, independent across stages |
| Trainable readout parameters | 3 | 3 |
| Total trainable parameters | 5 | 7 |
| Gate count | 6 (4 RY, 2 CX) | 10 (8 RY, 2 CX) |
| Logical gate depth | 4 layers: parallel feature RYs; CX; parallel trainable RYs; CX | 6 layers: parallel feature RYs; parallel trainable RYs; CX; parallel reupload RYs; parallel trainable RYs; CX |
| Candidate feature repetitions | one encoding stage | two encoding stages, with the second stage carrying distinct `x2/x3` inputs |

Depth above is a dependency-DAG count under the displayed circuit and parallel disjoint single-qubit gates; it is not a compiler report. No backend transpilation or hardware depth was measured.

The legacy `quantum.py` `reupload` variant has four independent trainable angles but re-encodes the same two features twice. It is not the proposed four-feature circuit. The extension is separately versioned as `two_stage_reupload_4feature` in `experiments_india_qml/torch_residual.py` and `experiments_india_qml/portable_inference.py`.

## Cheap verification completed

- Existing NumPy baseline `one_block` and `reupload`: parameter-shift MSE gradients matched centered finite differences on four bounded fixture rows. Maximum absolute errors were `7.38e-12` and `8.35e-12`; both passed the `1e-5` criterion.
- Existing NumPy baseline expectation values matched QpiAI's local `StatevectorSimulator` for both variants with maximum absolute error `2.23e-16`.
- Portable inference tests: `2 passed in 3.25s` using the original SDK sibling venv. The four-feature portable NumPy prediction matched the actual QpiAI local simulator to `1e-12` in the test. The baseline portable mapping matched the existing NumPy circuit.
- A cheap two-qubit CUDA kernel probe using Numba 0.68.1 did not execute. The exact error was `NvvmSupportError: libNVVM cannot be found` / `Could not find module 'nvvm.dll' (or one of its dependencies)`. Numba reported `cuda.is_available() == False`; `nvidia-smi` visibility is not evidence that this Python runtime can launch a kernel.
- The isolated RUN2 scratch interpreter has NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.4.2 and fastparquet 2024.11.0, but no Torch. The attempted official Torch 2.14.0 CUDA 13.0 wheel download was stopped before installation at the usage gate. No autograd test has run.

## Review questions / evidence boundary

1. Is the second stage's mapping of `x2` to q0 and `x3` to q1 the intended feature-to-wire contract, or should review choose a different mapping before training?
2. Is a 7-parameter two-qubit circuit with 6 logical layers a reasonable ablation against the 5-parameter, 4-layer baseline for the available 819 training couriers?
3. Should the four-feature circuit be eligible for validation selection only after confirming per-feature sensitivity and gradient checks, or remain diagnostic until a separate review?
4. Does review approve attempting an isolated PyTorch install and full VQC fit? This document itself does not lift the current training hold.

No train/validation metric, selected architecture, fusion coefficient, CUDA timing, final test prediction, or trained model is reported here.
