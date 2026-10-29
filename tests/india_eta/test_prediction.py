from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from backend.india_eta.errors import EtaError
from backend.india_eta.prediction import FEATURE_SCHEMA_HASH, _canonical_hash, load_model_registry, predict_eta


def request() -> dict:
    return {
        "schema_version": "india_eta.prediction.request.v1", "request_id": "fixture-001",
        "prediction_time": "2026-01-01T10:00:00+05:30", "source_domain": "india_food_delivery",
        "geography": "IN", "origin": {"id": "A", "latitude": 12.9, "longitude": 77.5},
        "destination": {"id": "B", "latitude": 12.91, "longitude": 77.51},
        "vehicle_type": "car", "features": {"distance_km": 5, "traffic_index": 0.75,
        "observed_at": {"distance_km": "2026-01-01T09:50:00+05:30", "traffic_index": "2026-01-01T09:55:00+05:30"}},
        "model_id": "india-eta-linear-fixture", "model_version": "1.0.0",
    }


class PredictionTests(unittest.TestCase):
    def test_fixture_predicts_classical_duration_and_discloses_checkpoint_fallback(self):
        result = predict_eta(request())
        self.assertEqual(result["predicted_duration_minutes"], 30.0)
        self.assertEqual(result["predicted_arrival"], "2026-01-01T10:30:00+05:30")
        self.assertFalse(result["qml_requested"])
        self.assertFalse(result["qml_executed"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["fallback_reason"], "duration_quantum_checkpoint_missing")
        self.assertIsNone(result["split_hash"])

    def test_feature_schema_hash_is_sha256_of_published_canonical_schema(self):
        schema_path = Path(__file__).resolve().parents[2] / "fixtures" / "india_eta" / "prediction_feature_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        self.assertEqual(FEATURE_SCHEMA_HASH, f"sha256:{digest}")

    def test_explicit_quantum_request_still_returns_classical_fallback(self):
        payload = request()
        payload["qml_requested"] = True
        result = predict_eta(payload)
        self.assertTrue(result["qml_requested"])
        self.assertFalse(result["qml_executed"])
        self.assertEqual(result["execution_type"], "classical_fixture")

    def test_rejects_unknown_fields_without_echoing_values(self):
        payload = request()
        payload["token"] = "private-sentinel"
        with self.assertRaises(EtaError) as caught:
            predict_eta(payload)
        self.assertEqual(caught.exception.code, "invalid_schema")
        self.assertNotIn("private-sentinel", str(caught.exception))

    def test_rejects_boolean_nan_naive_and_future_values(self):
        cases = []
        p = request(); p["features"]["distance_km"] = True; cases.append(p)
        p = request(); p["features"]["traffic_index"] = math.nan; cases.append(p)
        p = request(); p["prediction_time"] = "2026-01-01T10:00:00"; cases.append(p)
        p = request(); p["features"]["observed_at"]["traffic_index"] = "2026-01-01T10:01:00+05:30"; cases.append(p)
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(EtaError):
                    predict_eta(payload)

    def test_rejects_invalid_coordinates_and_unknown_model(self):
        payload = request(); payload["origin"]["latitude"] = 91
        with self.assertRaises(EtaError):
            predict_eta(payload)
        payload = request(); payload["model_version"] = "2.0"
        with self.assertRaises(EtaError) as caught:
            predict_eta(payload)
        self.assertEqual(caught.exception.code, "model_unavailable")

    def test_hash_verified_server_injected_release_and_tamper_detection(self):
        path = Path(__file__).resolve().parents[2] / "fixtures" / "india_eta" / "prediction_model.json"
        release = json.loads(path.read_text(encoding="utf-8"))
        digest = _canonical_hash(release)
        injected = {"india-eta-linear-fixture@1.0.0": {"release": release, "sha256": digest}}
        self.assertEqual(predict_eta(request(), injected)["model_sha256"], digest)
        tampered = copy.deepcopy(release); tampered["intercept"] = 999
        injected["india-eta-linear-fixture@1.0.0"]["release"] = tampered
        with self.assertRaises(EtaError) as caught:
            predict_eta(request(), injected)
        self.assertEqual(caught.exception.code, "model_integrity_error")

    def test_server_allowlisted_frozen_release_uses_its_identity_and_provenance(self):
        path = Path(__file__).resolve().parents[2] / "fixtures" / "india_eta" / "prediction_model.json"
        release = json.loads(path.read_text(encoding="utf-8"))
        release.update({"model_id": "frozen-eta-release", "model_version": "2.1.0", "model_kind": "frozen_classical"})
        release["provenance"] = {"status": "validated_source_domain", "data_hash": "sha256:data-123",
            "split_hash": "sha256:split-456", "training_data_hash": "sha256:train-789", "source": "owner-frozen-release"}
        digest = _canonical_hash(release)
        release["content_sha256"] = digest
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / ".runtime") as temp:
            Path(temp, "release.json").write_text(json.dumps(release), encoding="utf-8")
            registry = load_model_registry(temp, {"frozen-eta-release@2.1.0": {"relative_path": "release.json", "sha256": digest}})
            payload = request(); payload["model_id"] = "frozen-eta-release"; payload["model_version"] = "2.1.0"
            result = predict_eta(payload, registry)
        self.assertEqual(result["prediction_source"], "classical_frozen_model")
        self.assertEqual(result["execution_type"], "classical_json_model")
        self.assertEqual(result["model_id"], "frozen-eta-release")
        self.assertEqual(result["provenance_status"], "validated_source_domain")
        self.assertEqual(result["split_hash"], "sha256:split-456")

    def test_server_model_loader_rejects_path_escape_and_wrong_pinned_hash(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / ".runtime") as temp:
            for config in [
                {"relative_path": "../outside.json", "sha256": "0" * 64},
                {"relative_path": "missing.json", "sha256": "0" * 64},
            ]:
                with self.subTest(config=config), self.assertRaises(EtaError):
                    load_model_registry(temp, {"other@1": config})
            source = Path(__file__).resolve().parents[2] / "fixtures" / "india_eta" / "prediction_model.json"
            Path(temp, "release.json").write_bytes(source.read_bytes())
            with self.assertRaises(EtaError) as caught:
                load_model_registry(temp, {"india-eta-linear-fixture@1.0.0": {"relative_path": "release.json", "sha256": "0" * 64}})
            self.assertEqual(caught.exception.code, "model_integrity_error")

    def test_schema_hash_mismatch_and_vehicle_are_rejected(self):
        payload = request(); payload["feature_schema_hash"] = "other"
        with self.assertRaises(EtaError) as caught:
            predict_eta(payload)
        self.assertEqual(caught.exception.code, "feature_schema_mismatch")
        payload = request(); payload["vehicle_type"] = "helicopter"
        with self.assertRaises(EtaError) as caught:
            predict_eta(payload)
        self.assertEqual(caught.exception.code, "unsupported_vehicle")


if __name__ == "__main__":
    unittest.main()
