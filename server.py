"""FastAPI Server for Standalone Model Server.

Serves LLM inference (orchestration) and model training APIs.
Renders the premium dark-themed Dashboard at GET /.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

SERVER_DIR = Path(__file__).resolve().parent

# Load environment configuration (try local .env, fallback to main app .env)
local_env = SERVER_DIR / ".env"
if local_env.exists():
    load_dotenv(local_env)
else:
    load_dotenv(SERVER_DIR.parent / "SAP_S4hana" / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sap-model-server")

def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

API_KEY = os.getenv("CUSTOM_LLM_API_KEY", "")

from trainer import ModelServerTrainer
trainer = ModelServerTrainer()

# Active version tracking config file
ACTIVE_VERSION_FILE = SERVER_DIR / "config" / "active_model_version.json"
MODEL_DIR = None

if ACTIVE_VERSION_FILE.exists():
    try:
        active_data = json.loads(ACTIVE_VERSION_FILE.read_text(encoding="utf-8"))
        active_path = active_data.get("active_path")
        if active_path:
            MODEL_DIR = Path(active_path)
    except Exception as exc:
        logger.error("Failed to load active model configuration: %s", exc)

if not MODEL_DIR:
    MODEL_DIR = Path(
        os.getenv(
            "OFFLINE_LLM_MODEL_DIR",
            str(SERVER_DIR / "versions" / "custom-invoice-llm"),
        )
    )

if not MODEL_DIR.is_absolute():
    MODEL_DIR = SERVER_DIR / MODEL_DIR

BASE_MODEL_PATH = os.getenv("OFFLINE_BASE_MODEL_PATH") or os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "")
LOCAL_FILES_ONLY = env_bool("LLM_LOCAL_FILES_ONLY", True)
LOAD_IN_4BIT = env_bool("LLM_LOAD_IN_4BIT", True)
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Offline SAP-Compatible Model Server", version="1.0.0")
templates = Jinja2Templates(directory=str(SERVER_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(SERVER_DIR / "static")), name="static")

model = None
tokenizer = None
model_load_error: str | None = None

class Message(BaseModel):
    role: str
    content: str

class Template(BaseModel):
    messages: list[Message]

class LLM(BaseModel):
    name: str

class BtpOrchestrationRequest(BaseModel):
    template: Template
    llm: LLM

class ChoiceMessage(BaseModel):
    content: str

class Choice(BaseModel):
    message: ChoiceMessage

class OrchestrationResult(BaseModel):
    choices: list[Choice]

class BtpOrchestrationResponse(BaseModel):
    orchestration_result: OrchestrationResult

class ModelReloadRequest(BaseModel):
    model_dir: str

class TrainRequest(BaseModel):
    dataset: list[dict]
    epochs: int = 3

class ActivateRequest(BaseModel):
    version_name: str

def model_source() -> tuple[str, str | None]:
    """Return (base_source, adapter_source) for loading adapter layers or full model."""
    adapter_config = MODEL_DIR / "adapter_config.json"
    model_config = MODEL_DIR / "config.json"

    if LORA_ADAPTER_PATH:
        adapter = Path(LORA_ADAPTER_PATH)
        if not adapter.is_absolute():
            adapter = SERVER_DIR / adapter
        base = BASE_MODEL_PATH or str(MODEL_DIR)
        return base, str(adapter)

    if adapter_config.exists():
        base_source = BASE_MODEL_PATH
        if not base_source:
            try:
                adapter_data = json.loads(adapter_config.read_text(encoding="utf-8"))
                base_source = str(adapter_data.get("base_model_name_or_path") or "")
            except (OSError, json.JSONDecodeError):
                base_source = ""
        if not base_source:
            raise RuntimeError(
                "The configured model folder looks like a LoRA adapter. Set OFFLINE_BASE_MODEL_PATH "
                "to the local base model folder."
            )
        return base_source, str(MODEL_DIR)

    if model_config.exists():
        return str(MODEL_DIR), None

    if BASE_MODEL_PATH:
        return BASE_MODEL_PATH, None

    raise RuntimeError(
        f"No offline model found in {MODEL_DIR}. Put a complete Hugging Face model there, "
        "or put a LoRA adapter there and set OFFLINE_BASE_MODEL_PATH."
    )

@app.on_event("startup")
def load_model_and_tokenizer() -> None:
    global model, tokenizer, model_load_error
    if os.getenv("OFFLINE_LLM_MOCK", "false").lower() == "true":
        logger.info("OFFLINE_LLM_MOCK=true. Bypassing real Hugging Face weights loading.")
        model = "mock_model"
        tokenizer = "mock_tokenizer"
        model_load_error = None
        return

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        base_source, adapter_source = model_source()
        logger.info("Loading offline tokenizer from %s", base_source)
        tokenizer = AutoTokenizer.from_pretrained(
            base_source,
            trust_remote_code=True,
            local_files_only=LOCAL_FILES_ONLY,
        )

        quantization_config = None
        if LOAD_IN_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        logger.info("Loading offline base model from %s", base_source)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_source,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=LOCAL_FILES_ONLY,
        )

        if adapter_source:
            logger.info("Loading LoRA adapter from %s", adapter_source)
            model = PeftModel.from_pretrained(
                base_model,
                adapter_source,
                local_files_only=LOCAL_FILES_ONLY,
            )
        else:
            model = base_model

        model.eval()
        model_load_error = None
        logger.info("Offline LLM weights successfully loaded and evaluated.")
    except Exception as exc:
        model_load_error = str(exc)
        logger.exception("Offline LLM loading failed: %s", exc)

def clean_and_repair_json(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    raw_text = raw_text.strip()

    try:
        json.loads(raw_text)
        return raw_text
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", raw_text, re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return raw_text

def require_api_key(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    token = ""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Authorization token.",
        )

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Render the HTML Standalone Retraining and Resource dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "api_key": API_KEY,
        }
    )

@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "healthy" if model is not None else "not_ready",
        "model_loaded": model is not None,
        "model_dir": str(MODEL_DIR),
        "load_error": model_load_error,
    }

@app.post("/v1/orchestration", response_model=BtpOrchestrationResponse)
async def orchestration(
    request: BtpOrchestrationRequest,
    authorization: Optional[str] = Header(None),
) -> BtpOrchestrationResponse:
    require_api_key(authorization)
    ai_mock = os.getenv("OFFLINE_LLM_MOCK", os.getenv("AI_MOCK", "true")).lower() == "true"
    if ai_mock or (model is None and os.getenv("APP_ENV", "development") == "development"):
        import json
        content = json.dumps({
            "vendor_name": "ACME Corp",
            "vendor_tax_id": "DE1234567",
            "vendor_id": "1000001",
            "invoice_number": "INV-2026-999",
            "invoice_date": "2026-06-24",
            "po_number": "4500000001",
            "currency": "EUR",
            "total_amount": 500.0,
            "tax_amount": 50.0,
            "bank_account": "12345678",
            "iban": "DE89370400440532013000",
            "payment_terms": "NT30",
            "line_items": [
                {
                    "line_number": 1,
                    "material": "MAT01",
                    "description": "Office Supplies",
                    "quantity": 10.0,
                    "unit_price": 50.0,
                    "amount": 500.0,
                    "uom": "EA"
                }
            ]
        })
        all_messages_content = " ".join([m.content for m in request.template.messages])
        if "classify" in all_messages_content.lower() or "classification" in all_messages_content.lower():
            content = json.dumps({
                "classification": "VALID",
                "confidence": 0.99,
                "summary": "Invoice INV-2026-999 passed validation and matches purchase order, vendor has clean history.",
                "details": {"po_number": "4500000001"}
            })
        return BtpOrchestrationResponse(
            orchestration_result=OrchestrationResult(
                choices=[Choice(message=ChoiceMessage(content=content))]
            )
        )

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not initialized. {model_load_error or ''}".strip(),
        )

    try:
        import torch

        messages_payload = [
            {"role": message.role, "content": message.content}
            for message in request.template.messages
        ]
        text_prompt = tokenizer.apply_chat_template(
            messages_payload,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1] :]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        content = clean_and_repair_json(completion)
        return BtpOrchestrationResponse(
            orchestration_result=OrchestrationResult(
                choices=[Choice(message=ChoiceMessage(content=content))]
            )
        )
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc

@app.post("/v1/model/reload")
async def reload_model(
    request: ModelReloadRequest,
    authorization: Optional[str] = Header(None),
):
    require_api_key(authorization)
    global MODEL_DIR, model, tokenizer, model_load_error

    new_dir = Path(request.model_dir)
    if not new_dir.is_absolute():
         new_dir = SERVER_DIR / new_dir

    if not new_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model directory does not exist: {request.model_dir}",
        )

    logger.info("Hot-reloading model from: %s", new_dir)
    MODEL_DIR = new_dir
    load_model_and_tokenizer()

    if model_load_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {model_load_error}",
        )

    return {"status": "success", "model_dir": str(MODEL_DIR)}

@app.get("/v1/model/hardware")
def get_hardware(authorization: Optional[str] = Header(None)):
    require_api_key(authorization)
    return trainer.get_hardware_info()

@app.get("/v1/model/info")
def get_model_info(reviewed_count: int = 0, authorization: Optional[str] = Header(None)):
    require_api_key(authorization)
    from datetime import datetime
    hw = trainer.get_hardware_info()
    status_data = trainer.read_status()
    est_seconds = trainer.estimate_seconds(reviewed_count)

    last_trained = None
    if ACTIVE_VERSION_FILE.exists():
        try:
            active_data = json.loads(ACTIVE_VERSION_FILE.read_text(encoding="utf-8"))
            active_path = active_data.get("active_path")
            if active_path:
                adapter_path = Path(active_path)
                if not adapter_path.is_absolute():
                     adapter_path = SERVER_DIR / adapter_path
                config_file = adapter_path / "adapter_config.json"
                if config_file.exists():
                    mtime = os.path.getmtime(config_file)
                    last_trained = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    return {
        "reviewed_invoices_count": reviewed_count,
        "estimated_seconds": est_seconds,
        "status": status_data.get("status", "IDLE"),
        "device_type": hw["device_type"],
        "gpu_name": hw["gpu_name"],
        "vram_gb": hw["vram_gb"],
        "cpu_cores": hw["cpu_cores"],
        "ram_gb": hw["ram_gb"],
        "last_trained_at": last_trained,
        "hardware_status": hw["hardware_status"],
    }

@app.get("/v1/model/status")
def get_training_status(authorization: Optional[str] = Header(None)):
    require_api_key(authorization)
    return trainer.get_status_response()

@app.post("/v1/model/train")
async def trigger_training(request: TrainRequest, authorization: Optional[str] = Header(None)):
    require_api_key(authorization)
    success = trainer.trigger_training(request.dataset, request.epochs)
    if not success:
        raise HTTPException(status_code=400, detail="Training is already in progress.")
    return {"status": "started"}

@app.post("/v1/model/cancel")
async def cancel_training(authorization: Optional[str] = Header(None)):
    require_api_key(authorization)
    success = trainer.cancel_training()
    if not success:
        raise HTTPException(status_code=400, detail="Training is not running.")
    return {"status": "cancelled"}

@app.get("/v1/model/versions")
def list_versions(authorization: Optional[str] = Header(None)):
    require_api_key(authorization)
    from datetime import datetime

    versions_dir = SERVER_DIR / "versions"
    active_version = None
    if ACTIVE_VERSION_FILE.exists():
        try:
            active_data = json.loads(ACTIVE_VERSION_FILE.read_text(encoding="utf-8"))
            active_version = active_data.get("active_version")
        except Exception:
            pass

    versions = []
    if versions_dir.exists():
        for item in versions_dir.iterdir():
            if item.is_dir() and (item / "adapter_config.json").exists():
                mtime = os.path.getmtime(item / "adapter_config.json")
                dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                versions.append({
                    "version_name": item.name,
                    "created_at": dt,
                    "active": item.name == active_version,
                    "path": str(item),
                })

    versions.sort(key=lambda x: x["created_at"], reverse=True)
    return {"versions": versions, "active_version": active_version}

@app.post("/v1/model/activate")
async def activate_version(request: ActivateRequest, authorization: Optional[str] = Header(None)):
    require_api_key(authorization)
    global MODEL_DIR

    versions_dir = SERVER_DIR / "versions"
    target_dir = versions_dir / request.version_name
    if not target_dir.exists() or not (target_dir / "adapter_config.json").exists():
        raise HTTPException(status_code=400, detail="Invalid version name or files missing.")

    ACTIVE_VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_VERSION_FILE.write_text(
        json.dumps(
            {
                "active_version": request.version_name,
                "active_path": str(target_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Activating version and reloading: %s", target_dir)
    MODEL_DIR = target_dir
    load_model_and_tokenizer()

    if model_load_error:
        raise HTTPException(
            status_code=500,
            detail=f"Model activated but failed to reload: {model_load_error}",
        )

    return {"status": "success", "active_version": request.version_name}
