"""Trainer manager for Model Server.

Handles spawning background training subprocesses, capturing live logs,
cancelling runs, and monitoring system hardware capabilities.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("model-server-trainer")

SERVER_DIR = Path(__file__).resolve().parent
DATA_DIR = SERVER_DIR / "data"
STATUS_FILE = DATA_DIR / "llm_training_status.json"
LOG_FILE = DATA_DIR / "llm_training.log"


class ModelServerTrainer:
    _lock = threading.Lock()
    _active_process: subprocess.Popen | None = None
    _active_thread: threading.Thread | None = None
    _should_cancel = False

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (SERVER_DIR / "versions").mkdir(parents=True, exist_ok=True)

    def get_ram_gb(self) -> float:
        """Detect total physical RAM in GB."""
        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            parts = line.split()
                            if len(parts) >= 2 and parts[1].isdigit():
                                return float(parts[1]) / (1024 * 1024)
            except Exception:
                pass

        try:
            output = subprocess.check_output(
                "wmic computersystem get totalphysicalmemory", shell=True
            ).decode()
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if len(lines) > 1 and lines[1].isdigit():
                return float(lines[1]) / (1024**3)
        except Exception:
            pass

        return 16.0  # fallback

    def get_hardware_info(self) -> dict:
        """Fetch system hardware specs and telemetry."""
        vram_gb = 0.0
        gpu_name = "None"
        device_type = "CPU"
        hardware_status = "WARNING"
        sec_per_step = 60.0  # default CPU speed

        # Try to import torch to detect GPU
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_type = "GPU"
                gpu_name = torch.cuda.get_device_name(0)
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = vram_bytes / (1024**3)

                if vram_gb >= 20.0:
                    hardware_status = "EXCELLENT"
                    sec_per_step = 0.8
                elif vram_gb >= 14.0:
                    hardware_status = "GOOD"
                    sec_per_step = 2.5
                elif vram_gb >= 11.0:
                    hardware_status = "COMPATIBLE"
                    sec_per_step = 6.0
                else:
                    hardware_status = "RISKY_OOM"
                    sec_per_step = 15.0
        except Exception:
            pass

        cpu_cores = os.cpu_count() or 4
        ram_gb = self.get_ram_gb()

        if device_type == "CPU":
            sec_per_step = max(30.0, 180.0 / cpu_cores)

        return {
            "device_type": device_type,
            "gpu_name": gpu_name,
            "vram_gb": round(vram_gb, 2),
            "cpu_cores": cpu_cores,
            "ram_gb": round(ram_gb, 2),
            "hardware_status": hardware_status,
            "sec_per_step": sec_per_step,
        }

    def estimate_seconds(self, reviewed_count: int, epochs: int = 3) -> float:
        """Estimate total training duration based on review count and hardware."""
        info = self.get_hardware_info()
        sec_per_step = info["sec_per_step"]

        num_examples = reviewed_count * 2
        batch_size = 2
        grad_accum = 4
        samples_per_step = batch_size * grad_accum
        steps_per_epoch = max(1, (num_examples + samples_per_step - 1) // samples_per_step)
        total_steps = steps_per_epoch * epochs

        return round(total_steps * sec_per_step, 1)

    def read_status(self) -> dict:
        """Read the contents of status.json, handling stale states."""
        if not STATUS_FILE.exists():
            return self._idle_status()

        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # If marked as training but subprocess is actually dead
            if data.get("status") == "TRAINING" and not self._is_process_active():
                data["status"] = "FAILED"
                data.setdefault("logs", []).append("Training process terminated unexpectedly.")
                self.write_status(data)
            return data
        except Exception:
            return self._idle_status()

    def _is_process_active(self) -> bool:
        process = self._active_process
        if process is not None and process.poll() is None:
            return True
        thread = self._active_thread
        if thread is not None and thread.is_alive():
            return True
        return False

    def _idle_status(self) -> dict:
        return {
            "status": "IDLE",
            "progress": 0.0,
            "epoch": 1,
            "loss": None,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "logs": [],
        }

    def write_status(self, data: dict):
        try:
            STATUS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to write training status: %s", exc)

    def write_log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} [SYSTEM] {message}\n"
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as exc:
            logger.error("Failed to write to training log: %s", exc)

    def get_status_response(self) -> dict:
        """Returns the training status combined with tail log lines."""
        data = self.read_status()
        logs = []

        if LOG_FILE.exists():
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                logs = [line.strip() for line in lines[-50:]]
            except Exception:
                logs = data.get("logs", [])

        return {
            "status": data.get("status", "IDLE"),
            "progress": data.get("progress", 0.0),
            "epoch": data.get("epoch", 1),
            "loss": data.get("loss"),
            "elapsed_seconds": data.get("elapsed_seconds", 0.0),
            "estimated_remaining_seconds": data.get("estimated_remaining_seconds"),
            "logs": logs,
        }

    def trigger_training(self, dataset_list: list[dict], epochs: int = 3) -> bool:
        with self._lock:
            status_data = self.read_status()
            if status_data.get("status") == "TRAINING":
                logger.warning("Training is already running.")
                return False

            self._should_cancel = False

            # Clear status and log files
            if STATUS_FILE.exists():
                STATUS_FILE.unlink()
            if LOG_FILE.exists():
                LOG_FILE.unlink()

            self.write_status(
                {
                    "status": "TRAINING",
                    "progress": 0.0,
                    "epoch": 1,
                    "loss": None,
                    "elapsed_seconds": 0.0,
                    "estimated_remaining_seconds": None,
                    "logs": [],
                }
            )

            # Write the received dataset list to dataset.json
            dataset_path = DATA_DIR / "dataset.json"
            try:
                dataset_path.write_text(json.dumps(dataset_list, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.error("Failed to save dataset: %s", exc)
                self.write_status(
                    {
                        "status": "FAILED",
                        "progress": 0.0,
                        "epoch": 1,
                        "loss": None,
                        "elapsed_seconds": 0.0,
                        "estimated_remaining_seconds": None,
                        "logs": [f"Failed to save training dataset: {exc}"],
                    }
                )
                return False

            ai_mock = os.getenv("OFFLINE_LLM_MOCK", os.getenv("AI_MOCK", "true")).lower() == "true"
            if ai_mock:
                self._active_thread = threading.Thread(
                    target=self._run_mock_training, args=(epochs,), daemon=True
                )
                self._active_thread.start()
                logger.info("Launched mock model training thread.")
            else:
                self._active_thread = threading.Thread(
                    target=self._run_real_training, args=(epochs,), daemon=True
                )
                self._active_thread.start()
                logger.info("Launched real model training process.")

            return True

    def cancel_training(self) -> bool:
        with self._lock:
            status_data = self.read_status()
            if status_data.get("status") != "TRAINING":
                return False

            self._should_cancel = True

            if self._active_process:
                try:
                    self._active_process.terminate()
                    self._active_process.wait(timeout=3)
                except Exception:
                    pass
                self._active_process = None

            self.write_log("Training was cancelled by the administrator.")

            current_status = self.read_status()
            current_status["status"] = "FAILED"
            self.write_status(current_status)

            return True

    def _run_mock_training(self, epochs: int):
        progress = 0.0
        epoch = 1
        start_time = time.time()
        self.write_log("Initializing mock LLM retraining cockpit...")
        time.sleep(1)

        total_mock_steps = 10
        mock_logs = [
            "Loaded dataset JSON containing custom conversation examples.",
            "Loading base model Qwen/Qwen2.5-7B-Instruct in 4-bit...",
            "Tokenizer pad token configured successfully.",
            "LoRA parameters successfully added (r=16, alpha=32).",
            "Starting LLM fine-tuning training loop...",
            "Step 1/9 (Epoch 1/3): Loss = 0.6421",
            "Step 2/9 (Epoch 1/3): Loss = 0.5312",
            "Step 3/9 (Epoch 1/3): Loss = 0.4285",
            "Step 4/9 (Epoch 2/3): Loss = 0.3150",
            "Step 5/9 (Epoch 2/3): Loss = 0.2314",
            "Step 6/9 (Epoch 2/3): Loss = 0.1795",
            "Step 7/9 (Epoch 3/3): Loss = 0.1241",
            "Step 8/9 (Epoch 3/3): Loss = 0.0894",
            "Step 9/9 (Epoch 3/3): Loss = 0.0521",
            "Saving fine-tuned LoRA adapters to local folder...",
            "Retraining complete! Model updated successfully.",
        ]

        try:
            for i in range(total_mock_steps):
                if self._should_cancel:
                    return

                elapsed = time.time() - start_time
                progress = (i / total_mock_steps) * 100
                epoch = 1 if i < 3 else (2 if i < 7 else 3)
                loss = 0.6 - (i * 0.06)

                log_idx = min(i, len(mock_logs) - 1)
                self.write_log(mock_logs[log_idx])
                if i == 5:
                    self.write_log(mock_logs[6])
                elif i == 8:
                    self.write_log(mock_logs[11])
                elif i == 9:
                    self.write_log(mock_logs[14])

                self.write_status(
                    {
                        "status": "TRAINING",
                        "progress": progress,
                        "epoch": epoch,
                        "loss": round(loss, 4),
                        "elapsed_seconds": round(elapsed, 1),
                        "estimated_remaining_seconds": round((total_mock_steps - i) * 1.5, 1),
                    }
                )
                time.sleep(1.5)

            if self._should_cancel:
                return

            # Simulate writing adapter version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            adapter_dir = SERVER_DIR / "versions" / f"v_{timestamp}"
            adapter_dir.mkdir(parents=True, exist_ok=True)

            with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                        "peft_type": "LORA",
                        "r": 16,
                        "lora_alpha": 32,
                    },
                    f,
                    indent=2,
                )

            with open(adapter_dir / "adapter_model.safetensors", "wb") as f:
                f.write(b"MOCK_LORA_WEIGHTS_SAFE_TENSORS")

            self.write_log(mock_logs[-2])
            self.write_log(mock_logs[-1])

            elapsed = time.time() - start_time
            self.write_status(
                {
                    "status": "COMPLETED",
                    "progress": 100.0,
                    "epoch": epochs,
                    "loss": 0.0521,
                    "elapsed_seconds": round(elapsed, 1),
                    "estimated_remaining_seconds": 0.0,
                    "active_path": str(adapter_dir),
                }
            )
        except Exception as exc:
            self.write_log(f"Mock training failed with error: {exc}")
            self.write_status(
                {
                    "status": "FAILED",
                    "progress": progress,
                    "epoch": epoch,
                    "loss": None,
                    "elapsed_seconds": round(time.time() - start_time, 1),
                    "estimated_remaining_seconds": None,
                }
            )

    def _run_real_training(self, epochs: int):
        start_time = time.time()
        self.write_log("Starting actual model retraining subprocess...")

        # Find python executable
        python_exe = sys.executable

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = SERVER_DIR / "versions" / f"v_{timestamp}"
        version_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["OFFLINE_LLM_MODEL_DIR"] = str(version_dir)

        run_script = str(SERVER_DIR / "train.py")
        dataset_path = str(DATA_DIR / "dataset.json")
        status_path = str(STATUS_FILE)

        base_model_name = os.getenv("OFFLINE_BASE_MODEL_PATH") or os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

        cmd = [
            python_exe,
            run_script,
            "--dataset_path",
            dataset_path,
            "--output_dir",
            str(version_dir),
            "--base_model",
            base_model_name,
            "--epochs",
            str(epochs),
            "--status_path",
            status_path,
        ]

        try:
            self.write_log(f"Spawning command: {' '.join(cmd)}")
            self._active_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(SERVER_DIR),
                env=env,
            )

            # Read logs line by line and append to LOG_FILE
            while True:
                line = self._active_process.stdout.readline()
                if not line:
                    break

                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp_str} [HF-TRAIN] {line}")

            ret_code = self._active_process.wait()
            self._active_process = None

            if self._should_cancel:
                return

            if ret_code == 0:
                self.write_log("Subprocess completed training successfully!")
                elapsed = time.time() - start_time
                self.write_status(
                    {
                        "status": "COMPLETED",
                        "progress": 100.0,
                        "epoch": epochs,
                        "loss": None,
                        "elapsed_seconds": round(elapsed, 1),
                        "estimated_remaining_seconds": 0.0,
                        "active_path": str(version_dir),
                    }
                )
            else:
                self.write_log(f"Subprocess failed with exit code: {ret_code}")
                self.write_status(
                    {
                        "status": "FAILED",
                        "progress": 0.0,
                        "epoch": 1,
                        "loss": None,
                        "elapsed_seconds": round(time.time() - start_time, 1),
                        "estimated_remaining_seconds": None,
                    }
                )
        except Exception as exc:
            self.write_log(f"Real training process failed: {exc}")
            self.write_status(
                {
                    "status": "FAILED",
                    "progress": 0.0,
                    "epoch": 1,
                    "loss": None,
                    "elapsed_seconds": round(time.time() - start_time, 1),
                    "estimated_remaining_seconds": None,
                }
            )
