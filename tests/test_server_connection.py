from fastapi.testclient import TestClient

import server
from model_settings import ModelRuntimeSettings, ModelSettingsStore


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
    assert "active_tunnel_pairing_link" in response.json()
    assert "lan_links" not in response.json()
    assert "configured_public_pairing_link" not in response.json()


def test_local_model_settings_validate_and_save_mock(monkeypatch, tmp_path):
    store = ModelSettingsStore(tmp_path / "model_runtime_settings.json")
    monkeypatch.setattr(server, "model_settings_store", store)
    client = TestClient(server.app, client=("127.0.0.1", 50000))
    payload = {
        "model_dir": "versions/missing-but-mocked",
        "base_model_path": "Qwen/Qwen2.5-7B-Instruct",
        "lora_adapter_path": "",
        "mock_mode": True,
        "local_files_only": True,
        "load_in_4bit": True,
        "max_new_tokens": 512,
        "temperature": 0.2,
    }
    tested = client.post("/v1/local/settings/model/test", json=payload)
    assert tested.status_code == 200

    saved = client.put("/v1/local/settings/model", json=payload)
    assert saved.status_code == 200
    assert saved.json()["model_ready"] is True
    assert store.path.exists()
    assert store.load(ModelRuntimeSettings(**payload)).max_new_tokens == 512


def test_local_model_settings_roll_back_failed_load(monkeypatch, tmp_path):
    store = ModelSettingsStore(tmp_path / "model_runtime_settings.json")
    monkeypatch.setattr(server, "model_settings_store", store)
    previous = server.runtime_model_settings.model_copy(
        deep=True, update={"mock_mode": True}
    )
    server.apply_runtime_model_settings(previous)
    server.load_model_and_tokenizer()

    original_loader = server.load_model_and_tokenizer

    def failing_candidate_loader():
        if server.runtime_model_settings.max_new_tokens == 777:
            server.model_load_error = "candidate failed"
            server.model_reloading = False
            return
        original_loader()

    monkeypatch.setattr(server, "load_model_and_tokenizer", failing_candidate_loader)
    client = TestClient(server.app, client=("127.0.0.1", 50000))
    payload = previous.model_copy(update={"max_new_tokens": 777}).model_dump()
    response = client.put("/v1/local/settings/model", json=payload)
    assert response.status_code == 400
    assert server.runtime_model_settings.max_new_tokens == previous.max_new_tokens
    assert server.model_ready() is True
    assert not store.path.exists()
