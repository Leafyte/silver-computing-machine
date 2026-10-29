"""Prediction, Business, and Quantum Optimization Metrics Engine.

Calculates:
- Statistical metrics: Brier score, Log loss, Expected Calibration Error (ECE),
  bootstrap confidence intervals.
- Business metrics: Coverage, SLA breach rates, transport, labor, penalty, total cost, regret.
- Quantum metrics: Feasible shot probability, optimum shot probability, optimality gap,
  circuit profile, fallback rate.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from benchmark_harness.interfaces import PredictionMetrics


def calculate_prediction_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> PredictionMetrics:
    """Calculate Brier score, ECE, log loss, and calibration bins."""
    n = len(y_true)
    if n == 0 or len(y_prob) != n:
        raise ValueError("y_true and y_prob must be non-empty sequences of equal length")

    pos_count = sum(y_true)
    pos_rate = pos_count / n

    # Brier score
    sq_errs = [(p - y) ** 2 for y, p in zip(y_true, y_prob)]
    brier = sum(sq_errs) / n

    # Standard error and 95% CI
    if n > 1:
        sample_var = sum((e - brier) ** 2 for e in sq_errs) / (n - 1)
        se = math.sqrt(sample_var / n)
        ci95 = 1.96 * se
    else:
        ci95 = 0.0

    # Log loss
    eps = 1e-15
    log_losses = [
        -(y * math.log(max(eps, min(1 - eps, p))) + (1 - y) * math.log(max(eps, min(1 - eps, 1 - p))))
        for y, p in zip(y_true, y_prob)
    ]
    mean_log_loss = sum(log_losses) / n

    # Calibration bins and ECE
    bins_data: List[Dict[str, float]] = []
    ece = 0.0
    for b in range(n_bins):
        low = b / n_bins
        high = (b + 1) / n_bins
        indices = [i for i, p in enumerate(y_prob) if (low <= p < high or (b == n_bins - 1 and p == high))]
        if not indices:
            continue
        bin_size = len(indices)
        mean_prob = sum(y_prob[i] for i in indices) / bin_size
        mean_true = sum(y_true[i] for i in indices) / bin_size
        bin_acc_diff = abs(mean_prob - mean_true)
        ece += (bin_size / n) * bin_acc_diff
        bins_data.append({
            "bin_lower": round(low, 2),
            "bin_upper": round(high, 2),
            "count": bin_size,
            "mean_confidence": round(mean_prob, 4),
            "empirical_event_rate": round(mean_true, 4),
            "calibration_gap": round(bin_acc_diff, 4),
        })

    return PredictionMetrics(
        sample_count=n,
        positive_count=pos_count,
        positive_rate=round(pos_rate, 4),
        brier_score=round(brier, 5),
        expected_calibration_error=round(ece, 5),
        log_loss=round(mean_log_loss, 5),
        brier_95ci_halfwidth=round(ci95, 5),
        calibration_bins=bins_data,
    )


def paired_difference_statistics(
    differences: Sequence[float],
) -> Dict[str, float]:
    """Calculate mean paired difference, standard deviation, and 95% CI."""
    n = len(differences)
    if n == 0:
        return {"mean": 0.0, "sd": 0.0, "ci95_halfwidth": 0.0, "n": 0}
    mean = sum(differences) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in differences) / (n - 1)
        sd = math.sqrt(variance)
        ci95 = 1.96 * sd / math.sqrt(n)
    else:
        sd = 0.0
        ci95 = 0.0
    return {
        "mean": round(mean, 5),
        "sd": round(sd, 5),
        "ci95_halfwidth": round(ci95, 5),
        "n": n,
    }
