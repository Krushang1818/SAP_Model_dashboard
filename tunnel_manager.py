"""Manage the Cloudflare Quick Tunnel as a PC2 child process."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
QUICK_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)


class TunnelManager:
    def __init__(self, server_port: int = 8001):
        self.server_port = server_port
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._url_event = threading.Event()
        self._mode: str | None = None
        self._public_url: str | None = None
        self._last_error: str | None = None
        self._logs: list[str] = []

    def cloudflared_path(self) -> str | None:
        configured = os.getenv("CLOUDFLARED_PATH", "").strip()
        candidates = [
            configured,
            str(SERVER_DIR / ".tools" / "cloudflared.exe"),
            str(SERVER_DIR / ".tools" / "cloudflared"),
            str(SERVER_DIR.parent / ".tools" / "cloudflared.exe"),
            shutil.which("cloudflared") or "",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return candidate
        return None

    def status(self) -> dict[str, object]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            if self._process is not None and not running and not self._last_error:
                self._last_error = "cloudflared exited unexpectedly."
            return {
                "available": self.cloudflared_path() is not None,
                "running": running,
                "mode": self._mode,
                "public_url": self._public_url,
                "error": self._last_error,
                "recent_logs": self._logs[-20:],
            }

    def start(self, mode: str) -> dict[str, object]:
        mode = (mode or "").strip().lower()
        if mode != "quick":
            raise ValueError("Only Quick Internet Tunnel mode is supported.")

        executable = self.cloudflared_path()
        if not executable:
            raise RuntimeError(
                "cloudflared is not installed. Run install_cloudflared.ps1 first."
            )

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                if self._mode == mode:
                    return self.status()
                raise RuntimeError("Stop the active tunnel before starting another mode.")

            self._mode = mode
            self._public_url = None
            self._last_error = None
            self._logs = []
            self._url_event.clear()

            command = [
                executable,
                "tunnel",
                "--no-autoupdate",
                "--protocol",
                "http2",
            ]
            child_env = os.environ.copy()
            command += ["--url", f"http://127.0.0.1:{self.server_port}"]

            creation_flags = (
                subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=SERVER_DIR,
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creation_flags,
                )
            except OSError as exc:
                self._process = None
                self._last_error = str(exc)
                raise RuntimeError(f"Could not start cloudflared: {exc}") from exc

            self._reader_thread = threading.Thread(
                target=self._read_output,
                name="cloudflared-output",
                daemon=True,
            )
            self._reader_thread.start()

        self._url_event.wait(timeout=25)
        current = self.status()
        if not current["running"]:
            raise RuntimeError(
                str(current["error"] or "Quick Tunnel stopped before it was ready.")
            )
        if not current["public_url"]:
            self.stop()
            raise RuntimeError(
                "Quick Tunnel did not return a public URL within 25 seconds."
            )

        return self.status()

    def stop(self) -> dict[str, object]:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            self._process = None
            self._reader_thread = None
            self._mode = None
            self._public_url = None
            self._url_event.clear()
        return self.status()

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                with self._lock:
                    self._logs.append(line)
                    self._logs = self._logs[-100:]
                    match = QUICK_URL_RE.search(line)
                    if match and self._mode == "quick":
                        self._public_url = match.group(0).rstrip("/")
                        self._url_event.set()
        finally:
            exit_code = process.poll()
            with self._lock:
                if exit_code not in {None, 0} and not self._last_error:
                    self._last_error = f"cloudflared exited with code {exit_code}."
                self._url_event.set()
