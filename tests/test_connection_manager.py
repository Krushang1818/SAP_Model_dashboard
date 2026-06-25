from types import SimpleNamespace

from connection_manager import (
    NodeIdentityStore,
    is_local_request,
    pairing_link,
)


def test_identity_persists_and_rotates(tmp_path):
    path = tmp_path / "node_identity.json"
    first = NodeIdentityStore(path)
    original_id = first.node_id
    original_token = first.token

    second = NodeIdentityStore(path)
    assert second.node_id == original_id
    assert second.token == original_token
    assert second.verify(original_token)

    rotated = second.rotate_token()
    assert rotated != original_token
    assert second.node_id == original_id
    assert not second.verify(original_token)
    assert NodeIdentityStore(path).token == rotated


def test_pairing_link_keeps_token_in_fragment():
    link = pairing_link("https://model.example.com/", "a token/value")
    assert link == "https://model.example.com/#token=a%20token%2Fvalue"
    assert "?token=" not in link


def test_local_request_rejects_forwarded_traffic():
    local = SimpleNamespace(
        headers={},
        client=SimpleNamespace(host="127.0.0.1"),
    )
    forwarded = SimpleNamespace(
        headers={"cf-connecting-ip": "203.0.113.4"},
        client=SimpleNamespace(host="127.0.0.1"),
    )
    remote = SimpleNamespace(
        headers={},
        client=SimpleNamespace(host="192.168.1.20"),
    )
    assert is_local_request(local) is True
    assert is_local_request(forwarded) is False
    assert is_local_request(remote) is False
