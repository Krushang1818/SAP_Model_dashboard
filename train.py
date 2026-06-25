"""Standalone offline QLoRA training script for the Model Server.

Loads a JSON dataset, prepares the base model, adds LoRA adapter layers,
and performs fine-tuning. Updates progress in a JSON file via a callback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("model-training-run")


class ProgressCallback(TrainerCallback):
    """Callback to record progress and loss on each training step."""

    def __init__(self, status_path: Path, epochs: int):
        self.status_path = status_path
        self.epochs = epochs
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        total_steps = state.max_steps
        loss = None

        if state.log_history:
            for log in reversed(state.log_history):
                if "loss" in log:
                    loss = log["loss"]
                    break

        progress = min(99.0, (step / total_steps) * 100.0) if total_steps > 0 else 0.0
        epoch = int(state.epoch) if state.epoch is not None else 1
        elapsed = time.time() - self.start_time
        remaining = None
        if progress > 0:
            remaining = (elapsed / progress) * (100.0 - progress)

        status_data = {
            "status": "TRAINING",
            "progress": round(progress, 1),
            "epoch": epoch,
            "loss": round(loss, 4) if loss is not None else None,
            "elapsed_seconds": round(elapsed, 1),
            "estimated_remaining_seconds": round(remaining, 1) if remaining is not None else None,
        }

        try:
            self.status_path.write_text(json.dumps(status_data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to write progress status: %s", exc)


def main():
    parser = argparse.ArgumentParser(description="Standalone Model Server QLoRA Training")
    parser.add_argument("--dataset_path", type=str, default="data/dataset.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--status_path", type=str, default="data/llm_training_status.json")

    args = parser.parse_args()

    # Resolve paths relative to current script
    script_dir = Path(__file__).resolve().parent
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = script_dir / dataset_path
        
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
        
    status_path = Path(args.status_path)
    if not status_path.is_absolute():
        status_path = script_dir / status_path

    logger.info("Starting training script execution...")
    logger.info("Dataset path: %s", dataset_path)
    logger.info("Output adapter directory: %s", output_dir)
    logger.info("Base model name: %s", args.base_model)
    logger.info("Epochs: %d", args.epochs)

    # 1. Load Dataset from JSON
    if not dataset_path.exists():
        logger.error("Dataset file not found: %s", dataset_path)
        sys.exit(1)

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_list = json.load(f)
        dataset = Dataset.from_list(dataset_list)
        logger.info("Loaded %d conversations from dataset JSON.", len(dataset))
    except Exception as exc:
        logger.error("Failed to parse dataset file: %s", exc)
        sys.exit(1)

    # 2. Setup Device and Quantization Config
    cuda_available = torch.cuda.is_available()
    logger.info("CUDA/GPU detection: %s", "Available" if cuda_available else "Not Available")

    logger.info("Loading tokenizer for %s...", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if cuda_available:
        logger.info("Setting up 4-bit QLoRA Quantization for GPU...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        logger.info("Loading base model %s in 4-bit...", args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Loading base model %s on CPU in float32...", args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True,
        )

    # 3. PEFT/LoRA Adapter Config
    logger.info("Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Training Arguments Config
    os.makedirs(output_dir, exist_ok=True)
    if cuda_available:
        training_args = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=1,
            num_train_epochs=args.epochs,
            save_strategy="no",
            optim="paged_adamw_8bit",
            fp16=True,
            bf16=False,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to="none",
            max_seq_length=512,
        )
    else:
        training_args = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=1,
            num_train_epochs=args.epochs,
            save_strategy="no",
            optim="adamw_torch",
            fp16=False,
            bf16=False,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to="none",
            max_seq_length=512,
            use_cpu=True,
        )

    # 5. Initialize SFTTrainer
    progress_callback = ProgressCallback(status_path=status_path, epochs=args.epochs)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
        tokenizer=tokenizer,
        callbacks=[progress_callback],
    )

    # Force trainable parameters to float32
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    logger.info("Starting fine-tuning...")
    trainer.train()

    logger.info("Saving fine-tuned adapter weights to %s...", output_dir)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Retraining complete! Model updated successfully.")


if __name__ == "__main__":
    main()
