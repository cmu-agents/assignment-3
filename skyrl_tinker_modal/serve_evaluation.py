"""Serve a saved LoRA adapter with vLLM on two H100s (no training workers).

From this directory, use a local directory containing adapter_config.json
and adapter_model.safetensors (an exported Hugging Face repo also works):

    LORA_ADAPTER=../runs/p5_adapter \
        uv run --extra modal modal deploy serve_evaluation.py

Evaluate with --server <printed-url> --model a3, or --model
Qwen/Qwen3.5-4B for the base model. Both use the same endpoint.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal
from skyrl_server import HF_CACHE, MODEL_NAME, VENV, hf_cache, image

# Reuse the training deployment's cached image, but start only vLLM.
adapter = os.environ.get("LORA_ADAPTER", "")
if modal.is_local():
    if not adapter:
        raise ValueError("Set LORA_ADAPTER to your exported HF repo or local adapter directory")
    if Path(adapter).is_dir():
        if not (Path(adapter) / "adapter_config.json").is_file() or not any(
            (Path(adapter) / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            raise ValueError(f"{adapter} must contain adapter_config.json and adapter weights")
        image = image.add_local_dir(adapter, remote_path="/adapter")
        adapter = "/adapter"

app = modal.App("scottyrl-evaluation")
PORT = 8000


@app.function(
    image=image,
    gpu="H100:2",
    cpu=8,
    memory=32 * 1024,
    timeout=12 * 60 * 60,
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    volumes={HF_CACHE: hf_cache},
    secrets=[modal.Secret.from_dict({"LORA_ADAPTER": adapter, "SKYRL_BASE_MODEL": MODEL_NAME})],
)
@modal.concurrent(max_inputs=128)
@modal.web_server(PORT, startup_timeout=20 * 60)
def serve() -> None:
    subprocess.Popen([
        f"{VENV}/bin/vllm", "serve", os.environ["SKYRL_BASE_MODEL"],
        "--host", "0.0.0.0", "--port", str(PORT),
        "--data-parallel-size", "2",
        "--enable-lora", "--max-lora-rank", "32",
        "--lora-modules", f"a3={os.environ['LORA_ADAPTER']}",
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
        "--max-model-len", "65000", "--max-num-seqs", "64",
        "--max-num-batched-tokens", "8192",
        "--gpu-memory-utilization", "0.85",
        "--enable-prefix-caching", "--gdn-prefill-backend", "triton",
        "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        "--generation-config", "vllm",
    ])
