import subprocess

from tunnel_manager import TunnelManager


class FakeProcess:
    def __init__(self, lines):
        self.stdout = iter(lines)
        self.return_code = None

    def poll(self):
        return self.return_code

    def terminate(self):
        self.return_code = 0

    def wait(self, timeout=None):
        return self.return_code

    def kill(self):
        self.return_code = -9


def test_quick_tunnel_parses_public_url(monkeypatch, tmp_path):
    executable = tmp_path / "cloudflared.exe"
    executable.write_bytes(b"test")
    manager = TunnelManager(server_port=8001)
    monkeypatch.setattr(manager, "cloudflared_path", lambda: str(executable))
    fake = FakeProcess(
        [
            "INF Requesting new quick Tunnel\n",
            "INF https://bright-node.trycloudflare.com ready\n",
        ]
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake)

    status = manager.start("quick")
    assert status["running"] is True
    assert status["mode"] == "quick"
    assert status["public_url"] == "https://bright-node.trycloudflare.com"

    stopped = manager.stop()
    assert stopped["running"] is False


def test_named_tunnel_is_rejected(monkeypatch, tmp_path):
    executable = tmp_path / "cloudflared.exe"
    executable.write_bytes(b"test")
    manager = TunnelManager(server_port=8001)
    monkeypatch.setattr(manager, "cloudflared_path", lambda: str(executable))
    try:
        manager.start("named")
        assert False, "named mode should be rejected"
    except ValueError as exc:
        assert "Only Quick Internet Tunnel" in str(exc)
