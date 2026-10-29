"""Decision Quality Optimization (Track A - P0) Benchmark Engine.

Implements and evaluates the 7-way comparator pipeline on frozen scenarios:
  A: Classical ML (Exact Second-Order Newton GBDT / XGBoost formulation) + Exact/MILP selection
  B: Standard VQC (predictive BCE + trainable readout) + Exact/MILP selection
  C: Decision-focused VQC (aligned financial surrogate + BCE calibration anchor) + Exact/MILP selection
  D: Standard VQC + Standard QAOA (p=1)
  E: Decision-focused VQC + Standard QAOA (p=1)
  F: Decision-focused VQC + LP-Relaxed Warm-start QAOA (p=1)
  G: Classical ML (Newton GBDT) + LP-Relaxed Warm-start QAOA (p=1)

Also evaluates V1 baselines (fixed-readout VQC, Logistic Regression, and V1 box-relaxation Warm-QAOA)
in controlled validation ablations to isolate each bottleneck and rejected optimization attempt.
"""
from __future__ import annotations

import json
import math
import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.domain import generate_routes
from backend.learning_dispatch import (
    _exposure_scale,
    _feasible_plans,
    _nominal_pressure,
    _route_id,
)
from backend.prospective_dispatch import (
    ProspectiveRoute,
    ProspectiveScenario,
    _materialize_plan,
    _probability_gradient,
    _sigmoid,
)
from backend.quantum_learning import (
    Example,
    _loss_gradient,
    expectation_z0,
    risk_probability as v1_risk_probability,
)
from backend.qubo import build_route_qubo
from backend.validator import validate_plan
from experiments_v1_1.quantum_optimization import execute_qaoa_benchmark

TRANSPORT_COST_PER_TRAVEL_MINUTE = 0.50
LABOR_COST_PER_DUTY_MINUTE = 0.25
OPERATING_COST_PER_TRAVEL_MINUTE = TRANSPORT_COST_PER_TRAVEL_MINUTE
CURRENCY = "USD"
FEATURE_MAX_TRAVEL = 30.0


def _base_scenario() -> Dict[str, Any]:
    """Reproducible base geometry with contractual SLA terms."""
    return {
        "depot": "D",
        "vehicles": [
            {"id": "R1", "capacity": 3},
            {"id": "R2", "capacity": 2},
        ],
        "orders": [
            {"id": "P", "demand": 1, "window": [0, 5.5], "service": 0.7, "penalty": 4.0},
            {"id": "Q", "demand": 1, "window": [0, 7.5], "service": 0.8, "penalty": 1.5},
            {"id": "R", "demand": 2, "window": [1, 10.0], "service": 0.6, "penalty": 3.0},
        ],
        "travel": {
            "D": {"D": 0.0, "P": 2.1, "Q": 1.5, "R": 2.8},
            "P": {"D": 2.0, "P": 0.0, "Q": 1.4, "R": 2.2},
            "Q": {"D": 1.7, "P": 1.2, "Q": 0.0, "R": 1.9},
            "R": {"D": 2.6, "P": 2.1, "Q": 2.0, "R": 0.0},
        },
    }


def _route_duty_minutes(route: Dict[str, Any]) -> float:
    """Compute driver duty duration = travel_cost + service + waiting at stops."""
    travel = float(route["travel_cost"])
    service_and_wait = 0.0
    for st in route.get("stops", ()):
        arr = float(st.get("arrival", 0.0))
        start = float(st.get("service_start", arr))
        finish = float(st.get("departure", st.get("service_end", start)))
        wait = max(0.0, start - arr)
        svc = max(0.0, finish - start)
        service_and_wait += wait + svc
    return travel + service_and_wait


def _deterministic_route_base_cost(route: Dict[str, Any]) -> float:
    """Transport + labor USD for a forecast or realized route."""
    travel = float(route["travel_cost"])
    duty = _route_duty_minutes(route)
    return travel * TRANSPORT_COST_PER_TRAVEL_MINUTE + duty * LABOR_COST_PER_DUTY_MINUTE


def generate_world_pair(
    base: Dict[str, Any], rng: random.Random
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float]:
    """Generate paired forecast and realized scenarios with correlated congestion."""
    latent = rng.uniform(0.75, 2.1)
    forecast_mult = min(2.5, max(0.5, latent + rng.gauss(0.0, 0.3)))
    forecast = json.loads(json.dumps(base))
    realized = json.loads(json.dumps(base))
    for u in sorted(base["travel"]):
        for v in sorted(base["travel"]):
            if u == v:
                continue
            raw = float(base["travel"][u][v])
            forecast["travel"][u][v] = round(raw * forecast_mult * rng.uniform(0.97, 1.03), 6)
            realized["travel"][u][v] = round(raw * latent * rng.uniform(0.90, 1.10), 6)
    return forecast, realized, forecast_mult, latent


def evaluate_plan_financial(
    realized_scenario: Dict[str, Any],
    route_ids: Sequence[str],
    actual_routes_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score plan on realized scenario with explicit transport, labor, and SLA penalty USD."""
    actual_routes = (
        actual_routes_cache
        if actual_routes_cache is not None
        else {_route_id(r): r for r in generate_routes(realized_scenario)}
    )
    plan_obj = _materialize_plan(route_ids, actual_routes)

    val = validate_plan(realized_scenario, plan_obj)
    if not val["valid"]:
        return {
            "valid": False,
            "errors": val["errors"],
            "hard_violations": 1,
            "order_coverage_rate": 0.0,
            "total_financial_cost": float("inf"),
        }

    travel_minutes = sum(float(r["travel_cost"]) for r in plan_obj["routes"])
    duty_minutes = sum(_route_duty_minutes(r) for r in plan_obj["routes"])
    transport_cost = travel_minutes * TRANSPORT_COST_PER_TRAVEL_MINUTE
    labor_cost = duty_minutes * LABOR_COST_PER_DUTY_MINUTE

    late_orders = 0
    total_orders = len(realized_scenario["orders"])
    late_minutes_list: List[float] = []
    contractual_penalties = 0.0
    order_pen_map = {o["id"]: float(o["penalty"]) for o in realized_scenario["orders"]}

    for r in plan_obj["routes"]:
        for stop in r["stops"]:
            lateness = float(stop.get("lateness", 0.0))
            if lateness > 0.0:
                late_orders += 1
                late_minutes_list.append(lateness)
                contractual_penalties += lateness * order_pen_map[stop["order_id"]]

    total_financial_cost = transport_cost + labor_cost + contractual_penalties
    breach_rate = late_orders / total_orders if total_orders > 0 else 0.0
    mean_severity = sum(late_minutes_list) / len(late_minutes_list) if late_minutes_list else 0.0
    max_severity = max(late_minutes_list) if late_minutes_list else 0.0

    return {
        "valid": True,
        "hard_violations": 0,
        "order_coverage_rate": 1.0,
        "travel_minutes": round(travel_minutes, 6),
        "duty_minutes": round(duty_minutes, 6),
        "transport_cost": round(transport_cost, 6),
        "labor_cost": round(labor_cost, 6),
        "contractual_penalties": round(contractual_penalties, 6),
        "total_financial_cost": round(total_financial_cost, 6),
        "sla_breach_rate": round(breach_rate, 6),
        "scenario_has_sla_breach": int(late_orders > 0),
        "mean_sla_breach_severity_min": round(mean_severity, 6),
        "max_sla_breach_severity_min": round(max_severity, 6),
    }


def build_experiment_records(
    n_replicates: int = 5,
    cases_per_replicate: int = 30,
    family_seed: int = 2026103001,
    split_seed: int = 2026103002,
) -> List[ProspectiveScenario]:
    """Build paired training, validation and test scenario groups (leakage-safe)."""
    base = _base_scenario()
    nominal_routes = {_route_id(r): r for r in generate_routes(base)}
    records: List[ProspectiveScenario] = []

    for rep in range(n_replicates):
        rng = random.Random(family_seed + rep * 10_007)
        case_ids = [f"rep{rep + 1:02d}-case{i:02d}" for i in range(cases_per_replicate)]
        split_indices = list(range(cases_per_replicate))
        random.Random(split_seed + rep * 1009).shuffle(split_indices)

        if cases_per_replicate == 30:
            train_n, val_n = 18, 6
        else:
            train_n = max(1, int(cases_per_replicate * 0.6))
            val_n = max(1, int(cases_per_replicate * 0.2))
        split_map = {idx: "train" for idx in split_indices[:train_n]}
        split_map.update({idx: "validation" for idx in split_indices[train_n : train_n + val_n]})
        split_map.update({idx: "test" for idx in split_indices[train_n + val_n :]})

        for case_idx, scenario_id in enumerate(case_ids):
            split = split_map[case_idx]
            forecast, realized, f_mult, _ = generate_world_pair(base, rng)
            f_routes = generate_routes(forecast)
            feasible_plans = _feasible_plans(forecast, f_routes)
            if not feasible_plans:
                continue

            r_routes = (
                {_route_id(r): r for r in generate_routes(realized)}
                if split == "train"
                else {}
            )
            labels = {}
            if split == "train":
                labels = {
                    r_id: int(any(float(st["lateness"]) > 0 for st in r["stops"]))
                    for r_id, r in r_routes.items()
                }

            route_rows = []
            for r in f_routes:
                rid = _route_id(r)
                if rid not in nominal_routes:
                    continue
                pressure = _nominal_pressure(nominal_routes[rid], base)
                features = (min(1.0, float(r["travel_cost"]) / FEATURE_MAX_TRAVEL), pressure)
                # Store full deterministic base financial cost (transport + labor) divided by OPERATING_COST_PER_TRAVEL_MINUTE
                # so forecast_travel_cost * OPERATING_COST_PER_TRAVEL_MINUTE equals exact deterministic route USD!
                det_usd = _deterministic_route_base_cost(r)
                route_rows.append(
                    ProspectiveRoute(
                        route_id=rid,
                        features=features,
                        train_label=labels.get(rid),
                        forecast_travel_cost=det_usd / OPERATING_COST_PER_TRAVEL_MINUTE,
                        exposure_scale=_exposure_scale(base, r),
                    )
                )

            plans = []
            for fp in feasible_plans:
                r_ids = [_route_id(r) for r in fp["routes"]]
                item: Dict[str, Any] = {"route_ids": r_ids, "realized_objective_train_only": None}
                if split == "train":
                    fin = evaluate_plan_financial(realized, r_ids, actual_routes_cache=r_routes)
                    item["realized_objective_train_only"] = float(fin["total_financial_cost"])
                plans.append(item)

            digest = f"{scenario_id}-{split}"
            records.append(
                ProspectiveScenario(
                    scenario_id,
                    split,
                    f_mult,
                    forecast,
                    realized,
                    digest,
                    tuple(route_rows),
                    tuple(plans),
                )
            )

    return records


# ==============================================================================
# Strong Classical ML: Second-Order Newton GBDT (XGBoost Formulation) & Logistic
# ==============================================================================

@dataclass
class _GBDTNode:
    leaf_value: float
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_GBDTNode"] = None
    right: Optional["_GBDTNode"] = None


class NewtonGBDT:
    """Exact second-order Gradient-Boosted Decision Trees (XGBoost logistic objective)."""

    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 3,
        learning_rate: float = 0.15,
        l2_reg: float = 1.5,
        min_child_hessian: float = 0.25,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.min_child_hessian = min_child_hessian
        self.base_logit = 0.0
        self.trees: List[_GBDTNode] = []

    def _fit_tree(
        self,
        xs: List[Tuple[float, float]],
        gs: List[float],
        hs: List[float],
        indices: List[int],
        depth: int,
    ) -> _GBDTNode:
        sum_g = sum(gs[i] for i in indices)
        sum_h = sum(hs[i] for i in indices)
        leaf_weight = -sum_g / (sum_h + self.l2_reg)
        node = _GBDTNode(leaf_value=leaf_weight)
        if depth >= self.max_depth or len(indices) < 4 or sum_h < 2.0 * self.min_child_hessian:
            return node

        parent_score = (sum_g * sum_g) / (sum_h + self.l2_reg)
        best_gain = 1e-7
        best_feat = None
        best_thresh = None
        best_left: List[int] = []
        best_right: List[int] = []

        for feat in (0, 1):
            sorted_idx = sorted(indices, key=lambda i: xs[i][feat])
            g_left, h_left = 0.0, 0.0
            for pos in range(len(sorted_idx) - 1):
                i = sorted_idx[pos]
                nxt = sorted_idx[pos + 1]
                g_left += gs[i]
                h_left += hs[i]
                if xs[i][feat] == xs[nxt][feat]:
                    continue
                h_right = sum_h - h_left
                if h_left < self.min_child_hessian or h_right < self.min_child_hessian:
                    continue
                g_right = sum_g - g_left
                gain = 0.5 * (
                    (g_left * g_left) / (h_left + self.l2_reg)
                    + (g_right * g_right) / (h_right + self.l2_reg)
                    - parent_score
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = 0.5 * (xs[i][feat] + xs[nxt][feat])
                    best_left = sorted_idx[: pos + 1]
                    best_right = sorted_idx[pos + 1 :]

        if best_feat is None or best_thresh is None:
            return node

        node.feature = best_feat
        node.threshold = best_thresh
        node.left = self._fit_tree(xs, gs, hs, best_left, depth + 1)
        node.right = self._fit_tree(xs, gs, hs, best_right, depth + 1)
        return node

    def fit(self, train_scenarios: Sequence[ProspectiveScenario]) -> "NewtonGBDT":
        xs: List[Tuple[float, float]] = []
        ys: List[int] = []
        for sc in train_scenarios:
            for r in sc.routes:
                if r.train_label is not None:
                    xs.append(r.features)
                    ys.append(int(r.train_label))
        pos = sum(ys)
        p0 = min(0.95, max(0.05, pos / max(1, len(ys))))
        self.base_logit = math.log(p0 / (1.0 - p0))
        logits = [self.base_logit for _ in xs]
        self.trees = []
        all_idx = list(range(len(xs)))

        for _ in range(self.n_estimators):
            ps = [_sigmoid(z) for z in logits]
            gs = [p - y for p, y in zip(ps, ys)]
            hs = [max(1e-6, p * (1.0 - p)) for p in ps]
            tree = self._fit_tree(xs, gs, hs, all_idx, 0)
            self.trees.append(tree)
            for i, x in enumerate(xs):
                logits[i] += self.learning_rate * self._eval_tree(tree, x)
        return self

    @staticmethod
    def _eval_tree(node: _GBDTNode, x: Sequence[float]) -> float:
        cur = node
        while cur.feature is not None and cur.threshold is not None:
            cur = cur.left if float(x[cur.feature]) <= cur.threshold else cur.right  # type: ignore[assignment]
        return cur.leaf_value

    def predict_proba(self, x: Sequence[float]) -> float:
        z = self.base_logit + sum(self.learning_rate * self._eval_tree(t, x) for t in self.trees)
        return _sigmoid(max(-30.0, min(30.0, z)))


def train_classical_logistic(train_scenarios: List[ProspectiveScenario], steps: int = 120) -> List[float]:
    """Train classical logistic regression predictor."""
    examples = [
        Example(r.features, int(r.train_label), "train", i)
        for i, (sc, r) in enumerate((s, rt) for s in train_scenarios for rt in s.routes if rt.train_label is not None)
    ]
    w = [0.0, 0.0, 0.0]
    lr = 0.35
    n = max(1, len(examples))
    for _ in range(steps):
        grad = [0.0, 0.0, 0.0]
        for ex in examples:
            z = w[0] * ex.features[0] + w[1] * ex.features[1] + w[2]
            err = _sigmoid(z) - ex.label
            grad[0] += err * ex.features[0]
            grad[1] += err * ex.features[1]
            grad[2] += err
        w = [w[k] - (lr / n) * grad[k] for k in range(3)]
    return w


def predict_logistic(features: Sequence[float], weights: Sequence[float]) -> float:
    return _sigmoid(weights[0] * features[0] + weights[1] * features[1] + weights[2])


# ==============================================================================
# VQC Models: De-Saturated Trainable Readout (theta0, theta1, w_obs, b_obs)
# ==============================================================================

def risk_probability(features: Sequence[float], theta: Sequence[float]) -> float:
    """Evaluate VQC probability. Supports both 2-param V1 theta and 4-param trainable-readout theta."""
    if len(theta) == 2:
        return v1_risk_probability(features, theta)
    if len(theta) == 4:
        th0, th1, w_obs, b_obs = (float(v) for v in theta)
        z0 = expectation_z0(features, (th0, th1))
        return _sigmoid(max(-30.0, min(30.0, w_obs * z0 + b_obs)))
    raise ValueError("theta must have length 2 (fixed readout) or 4 (trainable readout)")


def vqc_prob_and_grad_4param(
    features: Sequence[float], params: Sequence[float]
) -> Tuple[float, List[float], int]:
    """Compute exact parameter-shift probability and 4-parameter gradient + circuit eval count (5)."""
    th0, th1, w_obs, b_obs = (float(v) for v in params)
    z0 = expectation_z0(features, (th0, th1))
    logit = max(-30.0, min(30.0, w_obs * z0 + b_obs))
    p = _sigmoid(logit)
    dp_dlogit = p * (1.0 - p)
    dz0 = 0.5 * (
        expectation_z0(features, (th0 + math.pi / 2.0, th1))
        - expectation_z0(features, (th0 - math.pi / 2.0, th1))
    )
    dz1 = 0.5 * (
        expectation_z0(features, (th0, th1 + math.pi / 2.0))
        - expectation_z0(features, (th0, th1 - math.pi / 2.0))
    )
    return p, [dp_dlogit * w_obs * dz0, dp_dlogit * w_obs * dz1, dp_dlogit * z0, dp_dlogit], 5


def estimate_train_conditional_severity(
    train_scenarios: Sequence[ProspectiveScenario], prior_strength: float = 2.0
) -> Dict[str, float]:
    """Estimate E[SLA_penalty_USD | breach] per route ID strictly on the training split."""
    sev_sum: Dict[str, float] = {}
    sev_cnt: Dict[str, int] = {}
    prior_map: Dict[str, float] = {}

    for sc in train_scenarios:
        r_map = {_route_id(r): r for r in generate_routes(sc.realized)}
        ord_pen = {o["id"]: float(o["penalty"]) for o in sc.realized["orders"]}
        for rt in sc.routes:
            prior_map[rt.route_id] = rt.exposure_scale
            if rt.train_label == 1 and rt.route_id in r_map:
                pen = sum(
                    float(st.get("lateness", 0.0)) * ord_pen[st["order_id"]]
                    for st in r_map[rt.route_id]["stops"]
                )
                sev_sum[rt.route_id] = sev_sum.get(rt.route_id, 0.0) + pen
                sev_cnt[rt.route_id] = sev_cnt.get(rt.route_id, 0) + 1

    return {
        rid: (sev_sum.get(rid, 0.0) + prior_strength * prior_map[rid])
        / (sev_cnt.get(rid, 0) + prior_strength)
        for rid in prior_map
    }


def train_predictive_vqc(
    train_scenarios: List[ProspectiveScenario],
    steps: int = 100,
    lr: float = 0.28,
    trainable_readout: bool = True,
) -> List[float]:
    """Train Standard VQC with BCE loss."""
    if not trainable_readout:
        examples = [
            Example(r.features, int(r.train_label), "train", i)
            for i, (s, r) in enumerate((sc, rt) for sc in train_scenarios for rt in sc.routes if rt.train_label is not None)
        ]
        theta = [0.31, -0.47]
        for _ in range(steps):
            _, grad = _loss_gradient(theta, examples, "predictive")
            theta = [theta[k] - 0.08 * grad[k] for k in range(2)]
        return theta

    rows = [(rt.features, int(rt.train_label)) for sc in train_scenarios for rt in sc.routes if rt.train_label is not None]
    params = [0.31, -0.47, -3.5, -1.5]
    n = max(1, len(rows))
    for _ in range(steps):
        g = [0.0, 0.0, 0.0, 0.0]
        for feat, y in rows:
            p, pg, _ = vqc_prob_and_grad_4param(feat, params)
            dl_dp = (p - y) / max(p * (1.0 - p), 1e-6)
            for k in range(4):
                g[k] += (dl_dp * pg[k]) / n
        params = [params[k] - lr * g[k] for k in range(4)]
    return params


def decision_loss_and_gradient(
    theta: Sequence[float],
    scenarios: Sequence[ProspectiveScenario],
    temperature: float = 0.25,
    decision_weight: float = 0.85,
    severity_map: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[float]]:
    """Compute aligned financial surrogate dispatch loss + analytic parameter-shift gradient."""
    dim = len(theta)
    total_loss = 0.0
    total_grad = [0.0] * dim

    for record in scenarios:
        rows = {r.route_id: r for r in record.routes}
        probs: Dict[str, float] = {}
        p_grads: Dict[str, List[float]] = {}
        for r in record.routes:
            if dim == 4:
                pr, pg, _ = vqc_prob_and_grad_4param(r.features, theta)
            else:
                pr = v1_risk_probability(r.features, theta)
                pg = _probability_gradient(r.features, theta)
            probs[r.route_id] = pr
            p_grads[r.route_id] = pg

        costs = [
            sum(
                rows[rid].forecast_travel_cost * OPERATING_COST_PER_TRAVEL_MINUTE
                + probs[rid] * (severity_map[rid] if severity_map and rid in severity_map else rows[rid].exposure_scale)
                for rid in p["route_ids"]
            )
            for p in record.plans
        ]
        actuals = [float(p["realized_objective_train_only"]) for p in record.plans]

        base_c = min(costs)
        unnorm = [math.exp(-min(60.0, max(-60.0, (c - base_c) / temperature))) for c in costs]
        norm = sum(unnorm)
        weights = [u / norm for u in unnorm]

        expected_actual = sum(w * act for w, act in zip(weights, actuals))
        bce_loss = 0.0
        for r in record.routes:
            if r.train_label is not None:
                pr = min(1.0 - 1e-9, max(1e-9, probs[r.route_id]))
                bce_loss -= (r.train_label * math.log(pr) + (1 - r.train_label) * math.log(1.0 - pr)) / len(record.routes)

        total_loss += decision_weight * expected_actual + (1.0 - decision_weight) * bce_loss

        dloss_dcost = [
            -(w * (act - expected_actual)) / temperature
            for w, act in zip(weights, actuals)
        ]

        for r in record.routes:
            sev = severity_map[r.route_id] if severity_map and r.route_id in severity_map else r.exposure_scale
            dcost_dp = sum(
                coeff * sev
                for p, coeff in zip(record.plans, dloss_dcost)
                if r.route_id in p["route_ids"]
            )
            bce_dp = 0.0
            if r.train_label is not None:
                pr = probs[r.route_id]
                bce_dp = (pr - float(r.train_label)) / max(pr * (1.0 - pr), 1e-6) / len(record.routes)
            comb = decision_weight * dcost_dp + (1.0 - decision_weight) * bce_dp
            for k in range(dim):
                total_grad[k] += comb * p_grads[r.route_id][k]

    n = max(1, len(scenarios))
    return total_loss / n, [g / n for g in total_grad]


def train_decision_focused_vqc(
    train_scenarios: List[ProspectiveScenario],
    steps: int = 80,
    lr: float = 0.08,
    temperature: float = 0.25,
    pretrain_predictive: bool = True,
    trainable_readout: bool = True,
    decision_weight: float = 0.85,
    severity_map: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Train Decision-Focused VQC with aligned financial units and BCE calibration anchor."""
    if pretrain_predictive:
        theta = train_predictive_vqc(
            train_scenarios, steps=80 if trainable_readout else 30, lr=0.28, trainable_readout=trainable_readout
        )
    else:
        theta = [0.31, -0.47, -3.5, -1.5] if trainable_readout else [0.31, -0.47]

    for _ in range(steps):
        _, grad = decision_loss_and_gradient(
            theta,
            train_scenarios,
            temperature=temperature,
            decision_weight=decision_weight,
            severity_map=severity_map,
        )
        theta = [theta[k] - lr * grad[k] for k in range(len(theta))]
    return theta


# ==============================================================================
# Downstream Selection (Exact/MILP & Risk-Conditioned QAOA)
# ==============================================================================

def _estimated_route_cost(
    row: ProspectiveRoute, prob: float, severity_map: Optional[Dict[str, float]] = None
) -> float:
    sev = severity_map[row.route_id] if severity_map and row.route_id in severity_map else row.exposure_scale
    return row.forecast_travel_cost * OPERATING_COST_PER_TRAVEL_MINUTE + prob * sev


def select_plan_exact(
    scenario: ProspectiveScenario,
    probabilities: Dict[str, float],
    severity_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Select exact-cover plan minimizing expected total financial exposure."""
    rows = {r.route_id: r for r in scenario.routes}
    best_plan = None
    best_est_cost = float("inf")
    for p in scenario.plans:
        est = sum(_estimated_route_cost(rows[rid], probabilities[rid], severity_map) for rid in p["route_ids"])
        if est < best_est_cost:
            best_est_cost = est
            best_plan = p
    assert best_plan is not None
    return {
        "route_ids": list(best_plan["route_ids"]),
        "estimated_subproblem_cost": best_est_cost,
        "origin": "exact_milp",
    }


def select_plan_qaoa(
    scenario: ProspectiveScenario,
    probabilities: Dict[str, float],
    warm_start: bool = False,
    relaxation_mode: str = "lp_augmented",
    layers: int = 1,
    shots: int = 512,
    grid_size: int = 6,
    seed: int = 7,
    severity_map: Optional[Dict[str, float]] = None,
    wall_deadline_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Select plan using QAOA on the risk-conditioned route QUBO."""
    rows = {r.route_id: r for r in scenario.routes}
    exact_ref = select_plan_exact(scenario, probabilities, severity_map=severity_map)
    subproblem_opt = exact_ref["estimated_subproblem_cost"]

    active_rids = {rid for p in scenario.plans for rid in p["route_ids"]}
    f_routes = generate_routes(scenario.forecast)
    qubo_candidates = []
    for r in f_routes:
        rid = _route_id(r)
        if rid in rows and rid in active_rids:
            est_cost = _estimated_route_cost(rows[rid], probabilities[rid], severity_map)
            qubo_candidates.append({
                "id": rid,
                "vehicle": r["vehicle_id"],
                "stops": r["order_ids"],
                "cost": float(est_cost),
            })

    if len(qubo_candidates) > 12:
        return {
            "route_ids": exact_ref["route_ids"],
            "origin": "fallback_exact",
            "qaoa_feasible_prob": 0.0,
            "subproblem_gap": 0.0,
            "circuit_evaluations": 0,
            "shots": 0,
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
            "fallback_reason": "candidate pool exceeds 12-qubit limit",
        }

    max_plan_cost = max(c["cost"] for c in qubo_candidates) * 2.0 + 2.0
    qubo = build_route_qubo(
        qubo_candidates,
        [o["id"] for o in scenario.forecast["orders"]],
        coverage_penalty=max_plan_cost,
        vehicle_penalty=max_plan_cost,
    )

    res = execute_qaoa_benchmark(
        qubo,
        layers=layers,
        warm_start=warm_start,
        relaxation_mode=relaxation_mode,
        optimizer_method="grid" if layers == 1 else "pattern",
        grid_size=grid_size,
        shots=shots,
        seed=seed,
        wall_deadline_seconds=wall_deadline_seconds,
        certified_optimum=subproblem_opt,
    )

    if res["best_feasible_routes"] is not None:
        achieved_est = float(res["best_feasible_objective"])
        return {
            "route_ids": res["best_feasible_routes"],
            "origin": "qaoa_sample",
            "qaoa_feasible_prob": res["feasible_sample_probability"],
            "subproblem_gap": max(0.0, achieved_est - subproblem_opt),
            "circuit_evaluations": res["circuit_evaluations"],
            "shots": res["shots"],
            "cpu_seconds": res["resource_accounting"]["total_cpu_seconds"],
            "wall_seconds": res["resource_accounting"]["total_wall_seconds"],
            "peak_memory_kb": res["resource_accounting"]["peak_memory_kb"],
        }

    return {
        "route_ids": exact_ref["route_ids"],
        "origin": "fallback_exact",
        "qaoa_feasible_prob": 0.0,
        "subproblem_gap": None,
        "circuit_evaluations": res["circuit_evaluations"],
        "shots": res["shots"],
        "cpu_seconds": res["resource_accounting"]["total_cpu_seconds"],
        "wall_seconds": res["resource_accounting"]["total_wall_seconds"],
        "peak_memory_kb": res["resource_accounting"]["peak_memory_kb"],
        "fallback_reason": "zero_feasible_quantum_shots",
    }


def compute_oracle_cost(
    scenario: ProspectiveScenario,
    actual_routes_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> float:
    """Compute realized hindsight oracle financial cost across feasible plans."""
    r_routes = (
        actual_routes_cache
        if actual_routes_cache is not None
        else {_route_id(r): r for r in generate_routes(scenario.realized)}
    )
    return min(
        evaluate_plan_financial(scenario.realized, p["route_ids"], actual_routes_cache=r_routes)["total_financial_cost"]
        for p in scenario.plans
    )


# ==============================================================================
# Full 7-Way Comparator Benchmark Runner + Matched-Budget Accounting
# ==============================================================================

def _compute_brier_and_ece(pairs: List[Tuple[float, int]]) -> Dict[str, Any]:
    n = max(1, len(pairs))
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(10)]
    for p, y in pairs:
        bins[min(9, max(0, int(p * 10)))].append((p, y))
    ece = sum((len(b) / n) * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b)) for b in bins if b)
    return {
        "brier_score": round(brier, 5),
        "expected_calibration_error_ece": round(ece, 5),
        "evaluated_routes": len(pairs),
    }


def run_comparator_benchmarks(
    records: List[ProspectiveScenario],
    df_temperature: float = 0.25,
    df_learning_rate: float = 0.08,
    df_decision_weight: float = 0.85,
    use_empirical_severity: bool = False,
    qaoa_shots: int = 256,
    qaoa_grid_size: int = 6,
) -> Dict[str, Any]:
    """Execute fair 7-way comparator pipelines A-G across 5 replicates (30 test scenarios)."""
    replicate_groups: Dict[int, Dict[str, List[ProspectiveScenario]]] = {}
    for sc in records:
        rep_num = int(sc.scenario_id.split("-")[0].replace("rep", ""))
        replicate_groups.setdefault(rep_num, {"train": [], "validation": [], "test": []})[sc.split].append(sc)

    pipeline_names = [
        "A_classical_ml_exact",
        "B_std_vqc_exact",
        "C_df_vqc_exact",
        "D_std_vqc_std_qaoa",
        "E_df_vqc_std_qaoa",
        "F_df_vqc_warm_qaoa",
        "G_classical_ml_warm_qaoa",
    ]

    all_test_evals: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pipeline_names}
    calibration_data: Dict[str, List[Tuple[float, int]]] = {
        "classical_gbdt": [],
        "classical_logistic": [],
        "std_vqc": [],
        "df_vqc": [],
        "v1_fixed_readout_vqc": [],
    }
    training_resource_log: List[Dict[str, Any]] = []
    replicate_metrics: List[Dict[str, Any]] = []

    for rep_num, splits in sorted(replicate_groups.items()):
        train_set = splits["train"]
        test_set = splits["test"]
        sev_map = estimate_train_conditional_severity(train_set) if use_empirical_severity else None
        n_train_routes = sum(len(s.routes) for s in train_set)

        # 1. Fit predictors with full CPU, wall, circuit-eval & peak-memory accounting
        # Classical GBDT (XGBoost Newton formulation)
        t0_w, t0_c = time.perf_counter(), time.process_time()
        tracemalloc.start()
        gbdt = NewtonGBDT(n_estimators=40, max_depth=3, learning_rate=0.15, l2_reg=1.5).fit(train_set)
        _, gbdt_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gbdt_fit_wall, gbdt_fit_cpu = time.perf_counter() - t0_w, time.process_time() - t0_c

        # Classical Logistic reference
        log_w = train_classical_logistic(train_set, steps=120)

        # V1 Fixed-readout VQC reference (diagnostic baseline)
        v1_theta = train_predictive_vqc(train_set, steps=80, trainable_readout=False)

        # Standard VQC (trainable readout, BCE)
        t0_w, t0_c = time.perf_counter(), time.process_time()
        tracemalloc.start()
        vqc_std_params = train_predictive_vqc(train_set, steps=100, lr=0.28, trainable_readout=True)
        _, std_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        std_fit_wall, std_fit_cpu = time.perf_counter() - t0_w, time.process_time() - t0_c
        std_train_circuit_evals = 100 * n_train_routes * 5

        # Decision-Focused VQC (trainable readout, aligned financial surrogate + BCE anchor)
        t0_w, t0_c = time.perf_counter(), time.process_time()
        tracemalloc.start()
        vqc_df_params = train_decision_focused_vqc(
            train_set,
            steps=80,
            lr=df_learning_rate,
            temperature=df_temperature,
            pretrain_predictive=True,
            trainable_readout=True,
            decision_weight=df_decision_weight,
            severity_map=sev_map,
        )
        _, df_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        df_fit_wall, df_fit_cpu = time.perf_counter() - t0_w, time.process_time() - t0_c
        df_train_circuit_evals = (80 + 80) * n_train_routes * 5

        training_resource_log.append({
            "replicate": rep_num,
            "gbdt_fit_cpu_s": round(gbdt_fit_cpu, 4),
            "gbdt_fit_wall_s": round(gbdt_fit_wall, 4),
            "gbdt_peak_mem_kb": round(gbdt_mem / 1024.0, 2),
            "std_vqc_fit_cpu_s": round(std_fit_cpu, 4),
            "std_vqc_fit_wall_s": round(std_fit_wall, 4),
            "std_vqc_circuit_evals": std_train_circuit_evals,
            "std_vqc_peak_mem_kb": round(std_mem / 1024.0, 2),
            "df_vqc_fit_cpu_s": round(df_fit_cpu, 4),
            "df_vqc_fit_wall_s": round(df_fit_wall, 4),
            "df_vqc_circuit_evals": df_train_circuit_evals,
            "df_vqc_peak_mem_kb": round(df_mem / 1024.0, 2),
        })

        rep_results: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pipeline_names}

        for case_offset, sc in enumerate(test_set):
            r_routes = {_route_id(r): r for r in generate_routes(sc.realized)}
            oracle_cost = compute_oracle_cost(sc, actual_routes_cache=r_routes)
            q_seed = 1000 * rep_num + case_offset + 7

            probs_gbdt = {r.route_id: gbdt.predict_proba(r.features) for r in sc.routes}
            probs_log = {r.route_id: predict_logistic(r.features, log_w) for r in sc.routes}
            probs_std = {r.route_id: risk_probability(r.features, vqc_std_params) for r in sc.routes}
            probs_df = {r.route_id: risk_probability(r.features, vqc_df_params) for r in sc.routes}
            probs_v1 = {r.route_id: risk_probability(r.features, v1_theta) for r in sc.routes}

            for r in sc.routes:
                if r.route_id in r_routes:
                    y = int(any(float(st["lateness"]) > 0 for st in r_routes[r.route_id]["stops"]))
                    calibration_data["classical_gbdt"].append((probs_gbdt[r.route_id], y))
                    calibration_data["classical_logistic"].append((probs_log[r.route_id], y))
                    calibration_data["std_vqc"].append((probs_std[r.route_id], y))
                    calibration_data["df_vqc"].append((probs_df[r.route_id], y))
                    calibration_data["v1_fixed_readout_vqc"].append((probs_v1[r.route_id], y))

            def record_exact(pipe_key: str, probs_map: Dict[str, float], vqc_inf_evals: int) -> None:
                t_c = time.process_time()
                sel = select_plan_exact(sc, probs_map, severity_map=sev_map)
                cpu_s = time.process_time() - t_c
                ev = evaluate_plan_financial(sc.realized, sel["route_ids"], actual_routes_cache=r_routes)
                ev["regret"] = ev["total_financial_cost"] - oracle_cost
                ev["origin"] = sel["origin"]
                ev["qaoa_feasible_prob"] = 1.0
                ev["subproblem_gap"] = 0.0
                ev["decision_cpu_s"] = cpu_s
                ev["circuit_evals"] = vqc_inf_evals
                ev["shots"] = 0
                rep_results[pipe_key].append(ev)

            def record_qaoa(pipe_key: str, probs_map: Dict[str, float], warm: bool, vqc_inf_evals: int) -> None:
                sel = select_plan_qaoa(
                    sc,
                    probs_map,
                    warm_start=warm,
                    relaxation_mode="lp_augmented",
                    layers=1,
                    shots=qaoa_shots,
                    grid_size=qaoa_grid_size,
                    seed=q_seed,
                    severity_map=sev_map,
                )
                ev = evaluate_plan_financial(sc.realized, sel["route_ids"], actual_routes_cache=r_routes)
                ev["regret"] = ev["total_financial_cost"] - oracle_cost
                ev["origin"] = sel["origin"]
                ev["qaoa_feasible_prob"] = sel["qaoa_feasible_prob"]
                ev["subproblem_gap"] = sel["subproblem_gap"] if sel["subproblem_gap"] is not None else 0.0
                ev["decision_cpu_s"] = sel["cpu_seconds"]
                ev["circuit_evals"] = vqc_inf_evals + sel["circuit_evaluations"]
                ev["shots"] = sel["shots"]
                rep_results[pipe_key].append(ev)

            n_rt = len(sc.routes)
            record_exact("A_classical_ml_exact", probs_gbdt, 0)
            record_exact("B_std_vqc_exact", probs_std, n_rt)
            record_exact("C_df_vqc_exact", probs_df, n_rt)
            record_qaoa("D_std_vqc_std_qaoa", probs_std, False, n_rt)
            record_qaoa("E_df_vqc_std_qaoa", probs_df, False, n_rt)
            record_qaoa("F_df_vqc_warm_qaoa", probs_df, True, n_rt)
            record_qaoa("G_classical_ml_warm_qaoa", probs_gbdt, True, 0)

        for p in pipeline_names:
            all_test_evals[p].extend(rep_results[p])

        replicate_metrics.append({
            "replicate": rep_num,
            "test_scenarios": len(test_set),
            "means": {
                p: {
                    "total_financial_cost": round(sum(r["total_financial_cost"] for r in rep_results[p]) / len(rep_results[p]), 4),
                    "regret": round(sum(r["regret"] for r in rep_results[p]) / len(rep_results[p]), 4),
                    "sla_breach_rate": round(sum(r["sla_breach_rate"] for r in rep_results[p]) / len(rep_results[p]), 4),
                    "raw_quantum_accept_rate": round(sum(1 for r in rep_results[p] if r["origin"] != "fallback_exact") / len(rep_results[p]), 4),
                }
                for p in pipeline_names
            },
        })

    summary_table: Dict[str, Any] = {}
    for p in pipeline_names:
        rows = all_test_evals[p]
        n_sc = len(rows)
        mean_cost = sum(r["total_financial_cost"] for r in rows) / n_sc
        mean_regret = sum(r["regret"] for r in rows) / n_sc
        sd_regret = math.sqrt(sum((r["regret"] - mean_regret) ** 2 for r in rows) / max(1, n_sc - 1))
        ci95_regret = 1.96 * sd_regret / math.sqrt(n_sc)

        summary_table[p] = {
            "mean_total_financial_cost_usd": round(mean_cost, 4),
            "mean_transport_cost_usd": round(sum(r["transport_cost"] for r in rows) / n_sc, 4),
            "mean_labor_cost_usd": round(sum(r["labor_cost"] for r in rows) / n_sc, 4),
            "mean_contractual_penalties_usd": round(sum(r["contractual_penalties"] for r in rows) / n_sc, 4),
            "mean_decision_regret_usd": round(mean_regret, 4),
            "regret_sd_usd": round(sd_regret, 4),
            "regret_95ci_halfwidth_usd": round(ci95_regret, 4),
            "order_sla_breach_rate": round(sum(r["sla_breach_rate"] for r in rows) / n_sc, 4),
            "scenario_sla_breach_rate": round(sum(r["scenario_has_sla_breach"] for r in rows) / n_sc, 4),
            "mean_sla_breach_severity_min": round(sum(r["mean_sla_breach_severity_min"] for r in rows) / n_sc, 4),
            "valid_plan_rate": round(sum(1 for r in rows if r["valid"]) / n_sc, 4),
            "hard_constraint_violations": sum(r["hard_violations"] for r in rows),
            "raw_solver_success_rate": round(sum(1 for r in rows if r["origin"] != "fallback_exact") / n_sc, 4),
            "fallback_rate": round(sum(1 for r in rows if r["origin"] == "fallback_exact") / n_sc, 4),
            "mean_qaoa_shot_feasibility": round(sum(r["qaoa_feasible_prob"] for r in rows) / n_sc, 4),
            "mean_subproblem_gap_usd": round(sum(r["subproblem_gap"] for r in rows) / n_sc, 4),
            "mean_decision_cpu_seconds": round(sum(r["decision_cpu_s"] for r in rows) / n_sc, 5),
            "mean_inference_circuit_evals": round(sum(r["circuit_evals"] for r in rows) / n_sc, 1),
            "mean_shots": round(sum(r["shots"] for r in rows) / n_sc, 1),
            "n_scenarios": n_sc,
        }

    calibration_metrics = {
        name: _compute_brier_and_ece(pairs)
        for name, pairs in calibration_data.items()
    }

    return {
        "financial_contract": {
            "currency": CURRENCY,
            "transport_cost_per_travel_minute_usd": TRANSPORT_COST_PER_TRAVEL_MINUTE,
            "labor_cost_per_duty_minute_usd": LABOR_COST_PER_DUTY_MINUTE,
            "penalty_unit": "USD_per_late_minute_per_order_contract",
            "provenance": "Explicit synthetic contract-logistics parameterization preserving hard vs soft SLA semantics.",
        },
        "summary_table": summary_table,
        "calibration_metrics": calibration_metrics,
        "training_resource_accounting": training_resource_log,
        "replicates": replicate_metrics,
    }
