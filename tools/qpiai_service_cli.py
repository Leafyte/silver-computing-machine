#!/usr/bin/env python3
"""CLI utility to interact with QpiAI Quantum and Optimization services.

Usage:
  # Check connection and credentials status
  python tools/qpiai_service_cli.py status

  # Predict SLA risk using decision VQC (local simulator)
  python tools/qpiai_service_cli.py vqc --features 0.35 0.25

  # Run decision VQC and submit to QpiAI Cloud
  python tools/qpiai_service_cli.py vqc --cloud

  # Build and inspect a routing QUBO model
  python tools/qpiai_service_cli.py qubo --costs 10.5 14.2 9.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.qpiai_service import QpiAIService


def cmd_status(service: QpiAIService) -> int:
    print("=" * 60)
    print("  QpiAI Platform Connection Status")
    print("=" * 60)

    res = service.check_connection()
    q = res["quantum"]
    print("\n[1] QpiAI Quantum / Cloud Service:")
    print(f"  - Configured in .env: {q['configured']}")
    print(f"  - Authenticated:     {q['authenticated']}")
    if q.get("details"):
        d = q["details"]
        print(f"  - User:              {d['user_name']} ({d['email']})")
        print(f"  - Cloud Backends:    {', '.join(d['supported_backends'])}")
    if q.get("error"):
        print(f"  - Error:             {q['error']}")

    opt = res["opt"]
    print("\n[2] QpiAI Optimization Service:")
    print(f"  - Configured in .env: {opt['configured']}")
    print(f"  - Authenticated:     {opt['authenticated']}")
    if opt.get("details"):
        d = opt["details"]
        print(f"  - User:              {d['user_name']} ({d['email']})")
        print(f"  - Cloud Subscription:{d['subscription_status']}")
        print(f"  - Cloud Credits:     {d['cloud_credits']}")
    if opt.get("error"):
        print(f"  - Error:             {opt['error']}")

    print("\n" + "=" * 60)
    return 0


def cmd_vqc(service: QpiAIService, args: argparse.Namespace) -> int:
    print("=" * 60)
    print("  Running Decision VQC Route Risk Prediction")
    print("=" * 60)
    res = service.predict_route_risk(
        features=args.features,
        theta=args.theta,
        shots=args.shots,
        use_cloud=args.cloud,
    )
    print(f"\nBackend:        {res['execution_backend']}")
    print(f"Route Features: Slack = {res['features']['time_slack']}, Stop Density = {res['features']['density']}")
    print(f"Observable <Z0>: {res['expectation_z0']:+.4f}")
    print(f"Delay Breach:   {res['breach_risk_probability'] * 100:.2f}%")
    print(f"Recommendation: {res['dispatch_recommendation']}")
    if "cloud_job" in res:
        print(f"Cloud Status:   {res['cloud_job']}")
    print("-" * 60)
    return 0


def cmd_qubo(service: QpiAIService, args: argparse.Namespace) -> int:
    print("=" * 60)
    print("  Building Route Candidate QUBO via QpiAI Opt")
    print("=" * 60)
    costs = args.costs
    # Simple adjacent route overlap penalties
    overlap = {(i, i + 1): 15.0 for i in range(len(costs) - 1)}
    relaxation = [1.0 / len(costs)] * len(costs)

    res = service.build_routing_qubo(
        candidate_costs=costs,
        overlap_matrix=overlap,
        relaxed_solution=relaxation,
    )
    print(f"\nCandidate Routes: {res['num_candidates']}")
    print(f"QUBO Terms:       {res['terms_count']}")
    print(f"QUBO Offset:      {res['qubo_offset']}")
    print(f"Terms:            {res['qubo_terms']}")
    if res.get("warm_start_circuit"):
        print(f"Warm-Start QAOA:  {res['warm_start_circuit']}")
    print("-" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="QpiAI Service CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: status
    subparsers.add_parser("status", help="Check QpiAI API connection and accounts")

    # Subcommand: vqc
    vqc_parser = subparsers.add_parser("vqc", help="Run Decision VQC SLA risk evaluation")
    vqc_parser.add_argument("--features", type=float, nargs=2, default=[0.35, 0.25], help="Normalized features: slack density")
    vqc_parser.add_argument("--theta", type=float, nargs=2, default=[0.31, -0.47], help="Trainable angles: th0 th1")
    vqc_parser.add_argument("--shots", type=int, default=100, help="Measurement shots")
    vqc_parser.add_argument("--cloud", action="store_true", help="Submit to QpiAI Cloud simulator")

    # Subcommand: qubo
    qubo_parser = subparsers.add_parser("qubo", help="Build and evaluate route selection QUBO")
    qubo_parser.add_argument("--costs", type=float, nargs="+", default=[12.5, 15.0, 9.2], help="Route candidate costs")

    args = parser.parse_args()
    service = QpiAIService()

    if args.command == "status":
        return cmd_status(service)
    elif args.command == "vqc":
        return cmd_vqc(service, args)
    elif args.command == "qubo":
        return cmd_qubo(service, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
