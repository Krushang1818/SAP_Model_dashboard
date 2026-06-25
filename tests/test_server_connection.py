from fastapi.testclient import TestClient

import server


def test_verify_requires_valid_token(monkeypatch):
    monkeypatch.setattr(server, "model", "mock-model")
    monkeypatch.setattr(server, "tokenizer", "mock-tokenizer")
    monkeypatch.setattr(server, "model_load_error", None)
    client = TestClient(server.app)

    unauthorized = client.get("/v1/connection/verify")
    assert unauthorized.status_code == 401

    verified = client.get(
        "/v1/connection/verify",
        headers={"Authorization": f"Bearer {server.identity.token}"},
    )
    assert verified.status_code == 200
    assert verified.json()["model_ready"] is True
    assert verified.json()["node_id"] == server.identity.node_id


def test_verify_reports_model_not_ready(monkeypatch):
    monkeypatch.setattr(server, "model", None)
    monkeypatch.setattr(server, "tokenizer", None)
    monkeypatch.setattr(server, "model_load_error", "base model missing")
    client = TestClient(server.app)
    response = client.get(
        "/v1/connection/verify",
        headers={"Authorization": f"Bearer {server.identity.token}"},
    )
    assert response.status_code == 200
    assert response.json()["model_ready"] is False
    assert response.json()["load_error"] == "base model missing"


def test_health_is_sanitized(monkeypatch):
    monkeypatch.setattr(server, "model", None)
    monkeypatch.setattr(server, "model_load_error", "sensitive local path")
    response = TestClient(server.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "not_ready", "model_loaded": False}


def test_dashboard_and_local_api_are_local_only():
    remote_client = TestClient(server.app, client=("192.168.1.50", 50000))
    assert remote_client.get("/").status_code == 403
    assert remote_client.get("/v1/local/connection").status_code == 403

    forwarded_client = TestClient(server.app, client=("127.0.0.1", 50000))
    assert (
        forwarded_client.get(
            "/v1/local/connection",
            headers={"cf-connecting-ip": "203.0.113.5"},
        ).status_code
        == 403
    )

    local_client = TestClient(server.app, client=("127.0.0.1", 50000))
    response = local_client.get("/v1/local/connection")
    assert response.status_code == 200
    assert "#token=" in response.json()["local_pairing_link"]
