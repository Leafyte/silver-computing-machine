"""Compute Measurement and 5-Stage Resource Accounting.

Tracks wall-clock time, process CPU time, peak memory usage, and circuit evaluations
across the 5 critical pipeline stages:
1. Preprocessing + LP Relaxation
2. Classical Parameter Optimization (Grid / Interpolation / Pattern)
3. Circuit Execution / Statevector Simulation
4. Finite-Shot Sampling
5. Independent Plan Decoding and Validation
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable, Dict, Optional, Tuple

from benchmark_harness.interfaces import StageResourceBreakdown


class StageTimer:
    """Context manager / helper to record wall and CPU times for an individual stage."""

    def __init__(self) -> None:
        self.wall_start = 0.0
        self.cpu_start = 0.0
        self.wall_elapsed = 0.0
        self.cpu_elapsed = 0.0

    def start(self) -> None:
        self.wall_start = time.perf_counter()
        self.cpu_start = time.process_time()

    def stop(self) -> None:
        self.wall_elapsed += time.perf_counter() - self.wall_start
        self.cpu_elapsed += time.process_time() - self.cpu_start


class ResourceTracker:
    """Manages full 5-stage accounting for an optimization pipeline."""

    def __init__(self, wall_deadline_seconds: Optional[float] = None) -> None:
        self.deadline = wall_deadline_seconds
        self.total_timer = StageTimer()
        self.stage_timers: Dict[str, StageTimer] = {
            "preprocessing": StageTimer(),
            "lp_relaxation": StageTimer(),
            "angle_optimization": StageTimer(),
            "circuit_execution": StageTimer(),
            "validation": StageTimer(),
        }
        self.circuit_evaluations = 0
        self.peak_memory_bytes = 0

    def start_overall(self) -> None:
        tracemalloc.start()
        self.total_timer.start()

    def time_stage(self, stage_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        timer = self.stage_timers[stage_name]
        timer.start()
        try:
            return fn(*args, **kwargs)
        finally:
            timer.stop()

    def record_circuit_evaluations(self, count: int) -> None:
        self.circuit_evaluations += count

    def finish(self) -> StageResourceBreakdown:
        self.total_timer.stop()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_bytes = peak

        tot_wall = self.total_timer.wall_elapsed
        tot_cpu = self.total_timer.cpu_elapsed
        budget_exceeded = (self.deadline is not None and tot_wall > self.deadline)

        return StageResourceBreakdown(
            preprocessing_wall_seconds=round(self.stage_timers["preprocessing"].wall_elapsed, 6),
            preprocessing_cpu_seconds=round(self.stage_timers["preprocessing"].cpu_elapsed, 6),
            lp_relaxation_wall_seconds=round(self.stage_timers["lp_relaxation"].wall_elapsed, 6),
            lp_relaxation_cpu_seconds=round(self.stage_timers["lp_relaxation"].cpu_elapsed, 6),
            angle_optimization_wall_seconds=round(self.stage_timers["angle_optimization"].wall_elapsed, 6),
            angle_optimization_cpu_seconds=round(self.stage_timers["angle_optimization"].cpu_elapsed, 6),
            circuit_execution_wall_seconds=round(self.stage_timers["circuit_execution"].wall_elapsed, 6),
            circuit_execution_cpu_seconds=round(self.stage_timers["circuit_execution"].cpu_elapsed, 6),
            validation_wall_seconds=round(self.stage_timers["validation"].wall_elapsed, 6),
            validation_cpu_seconds=round(self.stage_timers["validation"].cpu_elapsed, 6),
            total_wall_seconds=round(tot_wall, 6),
            total_cpu_seconds=round(tot_cpu, 6),
            peak_memory_bytes=self.peak_memory_bytes,
            circuit_evaluations=self.circuit_evaluations,
            deadline_budget_seconds=self.deadline,
            budget_exceeded=budget_exceeded,
        )
