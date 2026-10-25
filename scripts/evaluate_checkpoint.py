"""Evaluate a checkpoint, an exported adapter, or the base model on a
held-out split (given).

Graders run this on the shipped val splits against your exported Part 5
adapter, served with vLLM.

The model settings (base model, chat template, thinking mode, max_tokens,
temperature, stop sequences) come from the same YAML config the run was
trained with, so evaluation renders prompts exactly like training did. The
environment and agent come from the config's `train` section. Any field can
be overridden with `--set`, as in the training scripts.

Samples come from an OpenAI-compatible completions server (vLLM, SGLang).
Pass its URL with --server and the served model name with --model. A trailing
/v1 is added automatically if needed.

Examples:
  # An adapter exported with scripts/export_checkpoint.py, served by vLLM:
  vllm serve Qwen/Qwen3.5-4B --enable-lora --max-lora-rank 32 \\
      --lora-modules a3=<hf_model_id>
  python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \\
      --server http://localhost:8000/v1 --model a3 --k 4
  # The same with SGLang (the adapter is named base:adapter in --model):
  python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --enable-lora \\
      --lora-paths a3=<hf_model_id> --max-lora-rank 32 --port 8000
  python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \\
      --server http://localhost:8000/v1 --model Qwen/Qwen3.5-4B:a3 --k 4

The printed JSON carries the results plus `model_path`, `server` and
`model`, so a saved output says what was evaluated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from scottyrl.agents import AGENT_REGISTRY
from scottyrl.config import load_config
from scottyrl.envs import make_env_factory
from scottyrl.evaluate import evaluate_policy
from scottyrl.policy import CompletionsServerPolicy, load_chat_template
from scottyrl.types import load_tasks

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="YAML config (see configs/)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="SECTION.KEY=VALUE", help="override a config field")
    p.add_argument("--server", required=True, metavar="URL",
                   help="OpenAI-compatible server URL, e.g. http://localhost:8000/v1")
    p.add_argument("--model", required=True, metavar="NAME",
                   help="served adapter or base-model name, e.g. a3 or Qwen/Qwen3.5-4B")
    p.add_argument("--tasks", default=None, help="jsonl path; defaults to the env's val split")
    p.add_argument("--k", type=int, default=None, help="rollouts per task (default: train.eval_k)")
    p.add_argument("--num-tasks", type=int, default=None, help="evaluate only the first N tasks")
    p.add_argument("--concurrency", type=int, default=64)
    args = p.parse_args(argv)
    args.server = args.server.rstrip("/")
    if not args.server.endswith("/v1"):
        args.server += "/v1"
    return args


async def main() -> None:
    args = parse_args()

    config = load_config(args.config, args.overrides)
    model_cfg, train_cfg = config.model, config.train
    k = args.k if args.k is not None else train_cfg.eval_k

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.base_model)
    rendering = dict(
        max_tokens=model_cfg.max_tokens,
        temperature=model_cfg.temperature,
        chat_template=load_chat_template(model_cfg.chat_template),
        enable_thinking=model_cfg.enable_thinking,
        preserve_thinking_history=model_cfg.preserve_thinking_history,
        stop_sequences=model_cfg.stop_sequences,
    )
    policy = CompletionsServerPolicy(args.server, args.model, tokenizer, **rendering)

    tasks = load_tasks(args.tasks or DATA_DIR / f"{train_cfg.env}_val.jsonl")
    if args.num_tasks:
        tasks = tasks[: args.num_tasks]

    results = await evaluate_policy(
        policy,
        tasks,
        make_env_factory(train_cfg.env),
        k=k,
        concurrency=args.concurrency,
        agent_cls=AGENT_REGISTRY[train_cfg.agent],
    )
    results["model_path"] = None
    results["server"] = args.server
    results["model"] = args.model
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
