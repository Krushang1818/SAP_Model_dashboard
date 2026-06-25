"""Persistent local model-loading settings for PC2."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, Field

SERVER_DIR = Path(__file__).resolve().parent
MODEL_SETTINGS_FILE = SERVER_DIR / "config" / "model_runtime_settings.json"


class ModelRuntimeSettings(BaseModel):
    model_dir: str
    base_model_path: str
    lora_adapter_path: str = ""
    mock_mode: bool = False
    local_files_only: bool = True
    load_in_4bit: bool = True
    max_new_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class ModelSettingsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or MODEL_SETTINGS_FILE
        self._lock = RLock()

    def load(self, defaults: ModelRuntimeSettings) -> ModelRuntimeSettings:
        with self._lock:
            if not self.path.exists():
                return defaults
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                return ModelRuntimeSettings.model_validate(payload)
            except (OSError, ValueError, json.JSONDecodeError):
                return defaults

    def save(self, settings: ModelRuntimeSettings) -> None:
        payload = (
            json.dumps(settings.model_dump(), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                    temp_path = Path(handle.name)
                os.replace(temp_path, self.path)
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)


def resolve_server_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else SERVER_DIR / path


def validate_model_settings(settings: ModelRuntimeSettings) -> dict[str, str]:
    model_dir = resolve_server_path(settings.model_dir)
    if not settings.mock_mode and not model_dir.exists():
        raise ValueError(f"Model/adapter directory does not exist: {model_dir}")

    adapter_path = (
        resolve_server_path(settings.lora_adapter_path)
        if settings.lora_adapter_path
        else model_dir
    )
    if not settings.mock_mode and settings.lora_adapter_path and not adapter_path.exists():
        raise ValueError(f"Adapter override does not exist: {adapter_path}")

    if (
        not settings.mock_mode
        and (adapter_path / "adapter_config.json").exists()
        and not settings.base_model_path.strip()
    ):
        raise ValueError("A base model path or Hugging Face model name is required.")

    return {
        "model_dir": str(model_dir),
        "adapter_path": str(adapter_path),
        "mode": "mock" if settings.mock_mode else "model",
    }


model_settings_store = ModelSettingsStore()
