import json
import time
from threading import Event

import pytest
from fastapi.testclient import TestClient

from studio.app import create_app
from studio.settings import Settings


@pytest.fixture
def app(tmp_path):
    return create_app(Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db"))


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


def completed(client):
    response = client.post("/api/runs", json={"name": "API integration test"})
    assert response.status_code == 202
    run_id = response.json()["id"]
    for _ in range(100):
        run = client.get("/api/runs/" + run_id).json()
        if run["status"] == "completed":
            return run
        time.sleep(0.01)
    pytest.fail("Evaluation did not complete")


def test_run_review_and_export(client):
    run = completed(client)
    original = run["pipelines"]["candidate"]["metrics"]
    path = "/api/runs/" + run["id"]
    review = {
        "case_id": run["pipelines"]["candidate"]["rows"][0]["case_id"],
        "verdict": "correct",
        "note": "Reviewed source evidence.",
    }
    assert client.post(path + "/reviews", json=review).status_code == 200
    exported = client.get(path + "/export")
    assert "attachment" in exported.headers["content-disposition"]
    body = exported.json()
    assert body["reviews"][0]["note"] == review["note"]
    assert body["pipelines"]["candidate"]["metrics"] == original
    assert len(client.get("/api/runs").json()) == 1
    assert client.post(path + "/reviews", json={**review, "case_id": "q99"}).status_code == 422
    assert client.post(path + "/tracking").status_code == 422


def test_review_upsert_and_storage_reopen(client, app):
    run = completed(client)
    path = "/api/runs/" + run["id"]
    review = {
        "case_id": run["pipelines"]["candidate"]["rows"][0]["case_id"],
        "verdict": "incorrect",
        "note": "First",
    }
    client.post(path + "/reviews", json=review)
    client.post(path + "/reviews", json={**review, "note": "Updated"})
    assert len(app.state.store.get_reviews(run["id"])) == 1
    from studio.storage import Store

    reopened = Store(str(app.state.store.engine.url))
    assert reopened.get_reviews(run["id"])[0]["note"] == "Updated"
    reopened.engine.dispose()


def test_cross_origin_and_untrusted_host_rejected(client):
    assert client.post("/api/runs", json={}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/runs", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/api/benchmark", headers={"Origin": "http://testserver"}).status_code == 200


def test_oversized_body_rejected(client):
    assert client.post("/api/runs", content=json.dumps({"name": "x" * 17000})).status_code == 413


def test_token_required_and_not_leaked(tmp_path):
    app = create_app(
        Settings(
            _env_file=None, database_url=f"sqlite:///{tmp_path}/auth.db", studio_api_token="test-secret-only"
        )
    )
    with TestClient(app) as client:
        assert client.get("/api/runs").status_code == 401
        assert client.get("/api/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert (
            client.get("/api/runs", headers={"Authorization": "Bearer test-secret-only"}).status_code == 200
        )
        assert "test-secret-only" not in client.get("/api/health").text


def test_remote_client_requires_token(app):
    with TestClient(app, client=("203.0.113.10", 1234)) as client:
        assert client.get("/api/runs").status_code == 401
        assert client.get("/api/health").status_code == 200


def test_live_guard_validation_and_not_found(client):
    assert client.post("/api/runs", json={"provider": "openai"}).status_code == 422
    assert client.post("/api/runs", json={"extra": True}).status_code == 422
    assert client.get("/api/runs/not-a-uuid").status_code == 404
    assert client.get("/api/health").headers["cache-control"] == "no-store"


def test_parallel_runs_rejected(client, monkeypatch):
    import studio.app as module

    original = module.run_comparison
    started, finish = Event(), Event()

    def delayed(*args):
        started.set()
        finish.wait(5)
        return original(*args)

    monkeypatch.setattr(module, "run_comparison", delayed)
    try:
        assert client.post("/api/runs", json={}).status_code == 202
        assert started.wait(2)
        assert client.post("/api/runs", json={}).status_code == 409
    finally:
        finish.set()
