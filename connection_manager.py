"""Identity, pairing-link, and local-request helpers for the model node."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import socket
import tempfile
import uuid
from pathlib import Path
from threading import RLock
from urllib.parse import quote

from fastapi import HTTPException, Request, status

SERVER_DIR = Path(__file__).resolve().parent
IDENTITY_FILE = SERVER_DIR / "config" / "node_identity.json"
FORWARDED_HEADERS = (
    "cf-connecting-ip",
    "cf-ray",
    "x-forwarded-for",
    "forwarded",
    "x-real-ip",
)


class NodeIdentityStore:
    """Persist a stable PC2 node ID and rotatable bearer token."""

    def __init__(self, path: Path | None = None):
        self.path = path or IDENTITY_FILE
        self._lock = RLock()
        self._identity = self._load_or_create()

    def _load_or_create(self) -> dict[str, str]:
        with self._lock:
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    node_id = str(data.get("node_id") or "").strip()
                    token = str(data.get("token") or "").strip()
                    if node_id and token:
                        return {"node_id": node_id, "token": token}
                except (OSError, json.JSONDecodeError):
                    pass

            identity = {
                "node_id": str(uuid.uuid4()),
                "token": os.getenv("CUSTOM_LLM_API_KEY", "").strip()
                or secrets.token_urlsafe(32),
            }
            self._atomic_write(identity)
            return identity

    @property
    def node_id(self) -> str:
        return self._identity["node_id"]

    @property
    def token(self) -> str:
        return self._identity["token"]

    def rotate_token(self) -> str:
        with self._lock:
            self._identity = {
                "node_id": self.node_id,
                "token": secrets.token_urlsafe(32),
            }
            self._atomic_write(self._identity)
            return self.token

    def verify(self, supplied_token: str) -> bool:
        expected = self.token.encode("utf-8")
        supplied = (supplied_token or "").encode("utf-8")
        return bool(supplied) and hmac.compare_digest(expected, supplied)

    def _atomic_write(self, payload: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, self.path)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)


def bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        return ""
    return authorization.split(" ", 1)[1].strip()


def require_node_token(
    identity: NodeIdentityStore, authorization: str | None
) -> None:
    if not identity.verify(bearer_token(authorization)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Authorization token.",
        )


def is_local_request(request: Request) -> bool:
    if any(request.headers.get(name) for name in FORWARDED_HEADERS):
        return False
    if request.client is None:
        return False
    host = (request.client.host or "").split("%", 1)[0]
    return host in {"127.0.0.1", "::1", "localhost"}


def require_local_request(request: Request) -> None:
    if not is_local_request(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This management endpoint is available only on the model-server PC.",
        )


def pairing_link(base_url: str, token: str) -> str:
    return f"{base_url.rstrip('/')}/#token={quote(token, safe='')}"


def local_ipv4_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            addresses.add(probe.getsockname()[0])
    except OSError:
        pass

    try:
        for result in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(result[4][0])
    except OSError:
        pass

    return sorted(
        address
        for address in addresses
        if address
        and not address.startswith("127.")
        and not address.startswith("169.254.")
        and address != "0.0.0.0"
    )
