"""Focused write-boundary checks for ETA assignment mutations."""

from fastapi.testclient import TestClient

from backend.api import create_app
from backend.auth import AuthConfig


TOKENS = {
    "dispatcher": "ETA_DISPATCHER_SECURITY_SENTINEL",
    "researcher": "ETA_RESEARCHER_SECURITY_SENTINEL",
    "admin": "ETA_ADMIN_SECURITY_SENTINEL",
}


class ControllerSpy:
    def __init__(self):
        self.register_calls = 0
        self.approve_calls = 0

    def register_plan(self, payload, principal):
        self.register_calls += 1
        return {"unexpected": "registration"}

    def approve_proposal(self, proposal_id, payload, principal):
        self.approve_calls += 1
        return {"unexpected": "approval"}


def test_local_mode_rejects_assignment_writes_before_controller_or_db_action(tmp_path):
    spy = ControllerSpy()
    app = create_app(eta_db_path=tmp_path / "eta.sqlite", eta_controller=spy)
    with TestClient(app) as client:
        plan = client.post("/api/v1/eta/plans", json={"private": "ETA_WRITE_SENTINEL"})
        approval = client.post("/api/v1/eta/proposals/p-1/approve", json={"private": "ETA_WRITE_SENTINEL"})
    assert plan.status_code == 403
    assert approval.status_code == 403
    assert "ETA_WRITE_SENTINEL" not in plan.text + approval.text
    assert spy.register_calls == 0
    assert spy.approve_calls == 0


def test_researcher_token_cannot_reach_assignment_mutations(tmp_path):
    spy = ControllerSpy()
    auth = AuthConfig.from_mapping("token", TOKENS)
    app = create_app(auth_config=auth, eta_db_path=tmp_path / "eta.sqlite", eta_controller=spy)
    headers = {"Authorization": f"Bearer {TOKENS['researcher']}"}
    with TestClient(app) as client:
        plan = client.post("/api/v1/eta/plans", json={"private": "ETA_WRITE_SENTINEL"}, headers=headers)
        approval = client.post("/api/v1/eta/proposals/p-1/approve", json={}, headers=headers)
    assert plan.status_code == 403
    assert approval.status_code == 403
    assert spy.register_calls == 0
    assert spy.approve_calls == 0
