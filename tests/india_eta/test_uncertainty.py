from __future__ import annotations

import math
import copy
import unittest

from backend.india_eta.errors import EtaError
from backend.india_eta.prediction import FEATURE_SCHEMA_HASH, _canonical_hash
from backend.india_eta.uncertainty import evaluate_sla, interval_coverage


PREDICTION = {"schema_version": "india_eta.prediction.response.v1", "prediction_time": "2026-01-01T10:00:00+05:30", "predicted_duration_minutes": 20.0,
              "predicted_arrival": "2026-01-01T10:20:00+05:30", "prediction_target": "travel_time",
              "model_id": "india-eta-linear-fixture", "model_version": "1.0.0",
              "feature_schema_hash": FEATURE_SCHEMA_HASH,
              "source_domain": "india_food_delivery", "geography": "IN",
              "condition_context": {"distance_km": 5.0, "traffic_index": 0.75}}


def request(**changes):
    payload = {"schema_version": "india_eta.sla.request.v1", "prediction": PREDICTION,
               "deadline": "2026-01-01T10:25:00+05:30", "late_penalty_per_minute": 6.0,
               "currency": "INR", "scenarios": [{"residual_minutes": -5, "weight": 1/3},
               {"residual_minutes": 0, "weight": 1/3}, {"residual_minutes": 10, "weight": 1/3}],
               "scenario_source": "synthetic_fixture"}
    payload.update(changes)
    return payload


class UncertaintyTests(unittest.TestCase):
    def trusted_artifact(self):
        artifact = {
            "schema_version": "india_eta.residual_distribution.v1", "uncertainty_id": "fixture-residuals",
            "uncertainty_version": "1.0.0", "model_id": "india-eta-linear-fixture", "model_version": "1.0.0",
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "source_domain": "india_food_delivery", "geography": "IN", "prediction_target": "travel_time",
            "status": "validated_source_domain",
            "condition_support": {"traffic_index": {"minimum": 0.0, "maximum": 0.8}},
            "scenarios": [{"residual_minutes": -5, "weight": 1/3}, {"residual_minutes": 0, "weight": 1/3},
                          {"residual_minutes": 10, "weight": 1/3}],
            "provenance": {"source_id": "owner-frozen-calibration", "data_hash": "sha256:" + "1" * 64,
                           "calibration_split_hash": "sha256:" + "2" * 64, "evaluation_split_hash": "sha256:" + "3" * 64,
                           "sample_count": 300, "calibration_method": "predeclared_empirical_residuals",
                           "calibration_artifact_hash": "sha256:" + "4" * 64},
        }
        artifact["content_sha256"] = _canonical_hash(artifact)
        return artifact

    def trusted_registry(self, artifact=None):
        artifact = artifact or self.trusted_artifact()
        digest = _canonical_hash(artifact)
        return {"fixture-residuals@1.0.0": {"artifact": artifact, "sha256": digest}}

    def test_hand_calculated_lateness_and_expected_exposure(self):
        result = evaluate_sla(request())
        self.assertEqual(result["status"], "synthetic_fixture")
        self.assertAlmostEqual(result["probability_late"], 1/3)
        self.assertAlmostEqual(result["expected_late_minutes"], 5/3)
        self.assertAlmostEqual(result["expected_penalty"], 10.0)
        self.assertEqual(result["risk_claim"], "synthetic_arithmetic_only")

    def test_missing_scenarios_returns_unavailable_nulls(self):
        payload = request(); payload.pop("scenarios"); payload.pop("scenario_source")
        result = evaluate_sla(payload)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["probability_late"])
        self.assertIsNone(result["expected_penalty"])

    def test_calibration_claim_cannot_be_self_attested(self):
        payload = request(); payload["scenario_source"] = "validated_source_domain"
        with self.assertRaises(EtaError) as caught:
            evaluate_sla(payload)
        self.assertEqual(caught.exception.code, "invalid_scenario_source")

    def test_rejects_negative_duration_nonfinite_residual_bad_weights_and_boolean_rate(self):
        scenarios = [
            [{"residual_minutes": -21, "weight": 1}],
            [{"residual_minutes": math.inf, "weight": 1}],
            [{"residual_minutes": 0, "weight": 0}],
            [{"residual_minutes": 0, "weight": True}],
        ]
        for rows in scenarios:
            with self.subTest(rows=rows), self.assertRaises(EtaError):
                evaluate_sla(request(scenarios=rows))
        with self.assertRaises(EtaError):
            evaluate_sla(request(late_penalty_per_minute=True))

    def test_weights_must_sum_to_one_and_service_target_is_rejected(self):
        with self.assertRaises(EtaError) as caught:
            evaluate_sla(request(scenarios=[{"residual_minutes": 0, "weight": 0.5}]))
        self.assertEqual(caught.exception.code, "invalid_weights")
        prediction = dict(PREDICTION, prediction_target="whole_delivery_time")
        with self.assertRaises(EtaError):
            evaluate_sla(request(prediction=prediction))

    def test_mismatched_prediction_arrival_is_rejected(self):
        prediction = dict(PREDICTION, predicted_arrival="2026-01-01T10:40:00+05:30")
        with self.assertRaises(EtaError) as caught:
            evaluate_sla(request(prediction=prediction))
        self.assertEqual(caught.exception.code, "invalid_prediction")

    def test_interval_coverage_is_descriptive_and_validated(self):
        self.assertEqual(interval_coverage([1, 4, 6], [[0, 2], [3, 5], [0, 5]]),
                         {"covered": 2, "count": 3, "empirical_coverage": 2/3, "status": "descriptive_only"})
        with self.assertRaises(EtaError):
            interval_coverage([1], [[2, 0]])

    def test_hash_pinned_frozen_distribution_returns_validated_risk_and_provenance(self):
        payload = request(); payload.pop("scenarios"); payload.pop("scenario_source")
        payload.update({"uncertainty_id": "fixture-residuals", "uncertainty_version": "1.0.0"})
        result = evaluate_sla(payload, registry=self.trusted_registry())
        self.assertEqual(result["status"], "validated_source_domain")
        self.assertAlmostEqual(result["probability_late"], 1/3)
        self.assertAlmostEqual(result["expected_late_minutes"], 5/3)
        self.assertAlmostEqual(result["expected_penalty"], 10.0)
        self.assertEqual(result["uncertainty_provenance"]["sample_count"], 300)

    def test_trusted_release_mismatch_and_self_attested_spoof_fail(self):
        payload = request(); payload.pop("scenarios"); payload.pop("scenario_source")
        payload.update({"uncertainty_id": "fixture-residuals", "uncertainty_version": "1.0.0"})
        artifact = self.trusted_artifact(); artifact["geography"] = "US"; artifact["content_sha256"] = _canonical_hash(artifact)
        registry = self.trusted_registry(artifact)
        with self.assertRaises(EtaError) as caught:
            evaluate_sla(payload, registry)
        self.assertEqual(caught.exception.code, "uncertainty_mismatch")
        with self.assertRaises(EtaError) as caught:
            evaluate_sla(request(scenario_source="validated_source_domain"), registry)
        self.assertEqual(caught.exception.code, "invalid_scenario_source")

    def test_tampered_trusted_artifact_is_rejected(self):
        payload = request(); payload.pop("scenarios"); payload.pop("scenario_source")
        payload.update({"uncertainty_id": "fixture-residuals", "uncertainty_version": "1.0.0"})
        registry = self.trusted_registry()
        registry["fixture-residuals@1.0.0"]["artifact"]["status"] = "out_of_domain"
        with self.assertRaises(EtaError) as caught:
            evaluate_sla(payload, registry)
        self.assertEqual(caught.exception.code, "uncertainty_integrity_error")

    def test_unconfigured_and_condition_out_of_domain_return_null_risk(self):
        payload = request(); payload.pop("scenarios"); payload.pop("scenario_source")
        payload.update({"uncertainty_id": "fixture-residuals", "uncertainty_version": "1.0.0"})
        self.assertEqual(evaluate_sla(payload)["status"], "unavailable")
        prediction = copy.deepcopy(PREDICTION); prediction["condition_context"]["traffic_index"] = 0.95
        result = evaluate_sla({**payload, "prediction": prediction}, self.trusted_registry())
        self.assertEqual(result["status"], "out_of_domain")
        self.assertIsNone(result["probability_late"])

    def test_trusted_release_marked_out_of_domain_returns_null_risk(self):
        payload = request(); payload.pop("scenarios"); payload.pop("scenario_source")
        payload.update({"uncertainty_id": "fixture-residuals", "uncertainty_version": "1.0.0"})
        artifact = self.trusted_artifact(); artifact["status"] = "out_of_domain"
        artifact["scenarios"] = []; artifact["content_sha256"] = _canonical_hash(artifact)
        result = evaluate_sla(payload, self.trusted_registry(artifact))
        self.assertEqual(result["status"], "out_of_domain")
        self.assertIsNone(result["expected_late_minutes"])


if __name__ == "__main__":
    unittest.main()
