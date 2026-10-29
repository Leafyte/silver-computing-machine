from __future__ import annotations

import unittest

from backend.india_eta.compatibility import check_compatibility
from backend.india_eta.errors import EtaError


def request(**changes):
    payload = {"schema_version": "india_eta.compatibility.request.v1", "source_domain": "india_food_delivery",
               "geography": "IN", "source_units": "minute", "prediction_target": "travel_time",
               "road_routing_provenance": "verified_route_duration", "vehicle_type": "car",
               "features": ["distance_km", "traffic_index"], "time_semantics": "timezone_aware_service_start",
               "service_semantics": "travel_only", "horizon_minutes": 45,
               "uncertainty_status": "validated_source_domain"}
    payload.update(changes)
    return payload


class CompatibilityTests(unittest.TestCase):
    def test_verified_road_duration_without_trusted_uncertainty_is_scenario_only(self):
        result = check_compatibility(request())
        self.assertEqual(result["decision"], "scenario_only")
        self.assertFalse(result["road_arc_mapping_allowed"])
        self.assertIn("trusted_uncertainty_registry_not_configured", result["reasons"])

    def test_self_asserted_calibration_does_not_upgrade_compatibility(self):
        result = check_compatibility(request(uncertainty_status="validated_source_domain"))
        self.assertEqual(result["decision"], "scenario_only")
        self.assertIn("trusted_uncertainty_registry_not_configured", result["reasons"])

    def test_whole_delivery_duration_cannot_be_used_as_road_arc(self):
        result = check_compatibility(request(prediction_target="whole_delivery_time"))
        self.assertEqual(result["decision"], "rejected")
        self.assertIn("whole_delivery_time_cannot_populate_road_arc", result["reasons"])
        self.assertFalse(result["road_arc_mapping_allowed"])

    def test_missing_road_provenance_and_unsupported_features_reject(self):
        result = check_compatibility(request(road_routing_provenance="straight_line_distance", features=["distance_km", "secret_feature"]))
        self.assertEqual(result["decision"], "rejected")
        self.assertIn("verified_road_duration_unavailable", result["reasons"])
        self.assertIn("unsupported_features", result["reasons"])

    def test_explicit_numeric_synthetic_route_overlay_is_scenario_only(self):
        result = check_compatibility(request(source_domain="synthetic_overlay", road_routing_provenance="synthetic_fixture",
                                             uncertainty_status="synthetic_fixture", duration_minutes=20.0))
        self.assertEqual(result["decision"], "scenario_only")
        self.assertFalse(result["road_arc_mapping_allowed"])
        self.assertEqual(result["scenario_duration_minutes"], 20.0)
        self.assertIn("synthetic_route_duration_not_road_verified", result["reasons"])

    def test_india_road_arc_cannot_use_synthetic_duration_provenance(self):
        result = check_compatibility(request(road_routing_provenance="synthetic_fixture", duration_minutes=20.0))
        self.assertEqual(result["decision"], "rejected")
        self.assertFalse(result["road_arc_mapping_allowed"])

    def test_synthetic_overlay_requires_explicit_synthetic_provenance(self):
        result = check_compatibility(request(source_domain="synthetic_overlay", road_routing_provenance="unknown",
                                             uncertainty_status="synthetic_fixture"))
        self.assertEqual(result["decision"], "rejected")
        self.assertIn("verified_road_duration_unavailable", result["reasons"])

    def test_rejects_nonfinite_or_invalid_horizon(self):
        with self.assertRaises(EtaError):
            check_compatibility(request(horizon_minutes=float("nan")))
        with self.assertRaises(EtaError):
            check_compatibility(request(uncertainty_status=[]))
        with self.assertRaises(EtaError):
            check_compatibility(request(duration_minutes=float("nan")))


if __name__ == "__main__":
    unittest.main()
