"""Unit tests for DemoService, RouteCatalog, Store, and Auth."""

import tempfile
from pathlib import Path
from backend.demo import DemoAuthManager, DemoCoordinatorService, DemoStore, RouteCatalog


def test_pairing_and_auth():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Path(tmpdir) / "demo.sqlite3"
        store = DemoStore(db)
        store.seed_pairing_codes([
            {"code": "PAIR01", "driver_id": "driver_1", "vehicle_id": "vehicle_1", "driver_name": "Ramesh Kumar"}
        ])
        auth = DemoAuthManager(store)

        # 1. Redeem valid code
        sess = auth.redeem_code("PAIR01")
        assert sess is not None
        assert sess["driver_id"] == "driver_1"
        assert sess["driver_name"] == "Ramesh Kumar"
        token = sess["token"]

        # 2. Re-using same code fails
        assert auth.redeem_code("PAIR01") is None

        # 3. Verify token
        verified = auth.verify_token(token)
        assert verified is not None
        assert verified["driver_id"] == "driver_1"

        # 4. Revocation
        assert auth.revoke_token(token) is True
        assert auth.verify_token(token) is None


def test_demo_state_and_geojson():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Path(tmpdir) / "demo.sqlite3"
        store = DemoStore(db)
        svc = DemoCoordinatorService(store)

        state = svc.get_driver_state("driver_1")
        assert state is not None
        assert state["driver_id"] == "driver_1"
        assert state["vehicle_id"] == "vehicle_1"
        assert state["route_version"] == 1
        assert state["geojson"] is not None
        assert state["geojson"]["type"] == "Feature"
        coords = state["geojson"]["geometry"]["coordinates"]
        assert len(coords) >= 4
        # Validate [lon, lat] format
        for c in coords:
            assert len(c) == 2
            assert 76.0 <= c[0] <= 78.0  # Delhi/Gurgaon longitude range
            assert 28.0 <= c[1] <= 29.0  # Delhi/Gurgaon latitude range


def test_congestion_switching_logic():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Path(tmpdir) / "demo.sqlite3"
        store = DemoStore(db)
        svc = DemoCoordinatorService(store)

        # Base expressway is 720s (12m), bypass is 960s (16m)

        # 1. Moderate congestion (+180s = 900s). 900s is still faster than 960s -> KEEP
        mod_res = svc.report_congestion(
            driver_id="driver_1",
            event_id="evt_mod_1",
            base_route_version=1,
            severity="moderate",
            note="Minor jam near toll",
        )
        assert mod_res["decision"] == "KEEP"
        assert mod_res["published_new_route"] is False
        assert mod_res["current_route_version"] == 1

        # 2. Heavy congestion (+480s = 1200s). 1200s vs 960s saves 240s (4 min) -> AUTO-PUBLISH
        hvy_res = svc.report_congestion(
            driver_id="driver_1",
            event_id="evt_hvy_1",
            base_route_version=1,
            severity="heavy",
            note="Massive jam at Mahipalpur flyover",
        )
        assert hvy_res["decision"] == "PUBLISHED_NEW_ROUTE"
        assert hvy_res["published_new_route"] is True
        assert hvy_res["current_route_version"] == 2

        # 3. Check updated state
        updated_state = svc.get_driver_state("driver_1")
        assert updated_state["route_version"] == 2
        assert updated_state["route_id"] == "route_v1_bypass"
        assert updated_state["corridor_id"] == "corridor_b_bypass"

        # 4. Driver acknowledges route
        svc.acknowledge("driver_1", version=2)
        acked_state = svc.get_driver_state("driver_1")
        assert acked_state["last_ack_version"] == 2
