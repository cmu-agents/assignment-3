# Self-Hosting the Training Backend with SkyRL and Modal

`scottyrl` trains against any Tinker-compatible service. In this assignment that service is [SkyRL's Tinker server](https://docs.skyrl.ai/docs/tinker/overview), which you deploy yourself on [Modal](https://modal.com/) GPUs. This directory contains one launch script for each kind of run in the assignment. Each script requests the GPUs its runs need and splits them between training and sampling to suit those runs.

| Launch script | Use it for | GPUs | Layout |
|---|---|---|---|
| `serve_number_search_sync.py` | Part 2 (`p2a`, `p2b`) and Part 4.1 (`p41`) | 2x H100 | colocated: both GPUs train *and* sample |
| `serve_number_search_async.py` | Part 3 (`p3`) and Part 4.3 (`p43_*`) | 2x H100 | split: 1 trains, 1 samples |
| `serve_textcraft.py` | Part 5 (`p5`) training and checkpoint export | 4x H100 | split: 1 trains, 3 sample |
| `serve_evaluation.py` | Part 5 exported-adapter and base-model evaluation | 2x H100 | inference only |

The layouts differ because the two trainers use GPUs differently. Synchronous training alternates between sampling and updating, so colocating the two keeps both GPUs busy in both phases (SkyRL puts the sampling engines to sleep during an update and wakes them afterwards). Asynchronous training samples *while* it updates, which a colocated server cannot do, so it needs a GPU for each role. Part 5 is asynchronous too, and TextCraft-Synth episodes are long and spend most of their time sampling, so three of its four GPUs go to the sampler.

The shared parts of the three deployments (the container image, the SkyRL version, the optimizer settings) live in `skyrl_server.py`. All three train with SkyRL's Megatron backend, which is the one SkyRL supports Qwen3.5 on, and sample with vLLM. The image pins a revision of [our SkyRL fork](https://github.com/adityasoni9998/SkyRL) containing the Megatron token-batched output-ordering fix and adapter-only sampler checkpoints, so that a saved LoRA snapshot is your adapter rather than a copy of the whole model. These changes are part of the pinned source; no local patch is applied during the image build. You should not need to edit any of these files.

## One-Time Setup

Create a Modal account, then install and authenticate the Modal CLI:

```bash
uv sync --extra modal
uv run modal setup
```

Modal is an optional extra, so a plain `uv sync` later on removes it again. If `uv run modal` stops working, run `uv sync --extra modal` once more.

Run all the commands below from this directory:

```bash
cd skyrl_tinker_modal
```

The first deployment builds a container image with SkyRL, Megatron, and their CUDA dependencies. This is a download of many gigabytes, so expect it to take 20 minutes or more. The build runs on one H100, so it costs about as much as that GPU for the same time. The image is shared by all three scripts and cached by Modal, so you only wait and pay once. If you want to build it ahead of time and check that it works, run the optional smoke test, which builds the image if needed and then uses one H100 for a few more minutes:

```bash
uv run modal run serve_number_search_sync.py::smoke_install
```

## Running a Part

**1. Deploy the server for the part you are working on.** For example, for Part 2:

```bash
uv run modal deploy serve_number_search_sync.py
```

Modal prints the URL of the deployment. Apart from the one-time image build, deploying costs nothing by itself: the server scales to zero, and GPUs are only allocated once a training run connects.

**2. Point `scottyrl` at it and train.** From the repo root:

```bash
export TINKER_BASE_URL=https://<your-workspace>--scottyrl-ns-sync-tinker-server.modal.run
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml
```

Copy the exact URL that `modal deploy` printed. It normally has the form above.

The first request starts the container, loads the model, and brings up the sampling engines. Expect to wait several minutes before the step-0 evaluation begins. Later runs against a warm server start right away.

**3. Stop the server when you are done.** An idle server shuts down by itself after five minutes, but you pay for those minutes. To stop it immediately:

```bash
uv run modal app stop scottyrl-ns-sync
```

Stopping removes the deployment, so run `uv run modal deploy ...` again before your next run. This takes seconds because the image is cached, and the URL stays the same. Your checkpoints live on a volume and are not affected.

The app names are `scottyrl-ns-sync`, `scottyrl-ns-async`, and `scottyrl-textcraft`. You can see what is running, and what it has cost so far, in the Modal dashboard.

## Standalone Evaluation with vLLM

Use `serve_evaluation.py` for Part 5 evaluation. It serves the base model or the fine-tuned checkpoint on **two H100s**. From this directory, deploy the local adapter downloaded in Part 5:

```bash
LORA_ADAPTER=../runs/p5_adapter \
    uv run --extra modal modal deploy serve_evaluation.py
```

Pass the printed URL directly to the evaluator's `--server` option.
Use `--model a3` for the adapter and `--model Qwen/Qwen3.5-4B` for the base
model. [Part 5](../ASSIGNMENT.md#part-5-textcraft-synth-10-points) lists all
four required commands. `TINKER_BASE_URL` is not used in `--server` mode, and
the training deployment can be stopped once the adapter has been downloaded.

The local adapter directory must contain
`adapter_config.json` and `adapter_model.safetensors`. The launcher uploads
that directory to the inference container. For a sampler archive downloaded
from your training volume, extract it with `tar -xf` (SkyRL's `.tar.gz` files
may be uncompressed tar archives). Use the **sampler** archive, such as
`sampler_weights/v50.tar.gz`, not a `resume_*.tar.gz` training checkpoint.

An exported Hugging Face adapter repository also works:

```bash
LORA_ADAPTER=<hf-username>/scottyrl-a3-p5_textcraft_grpo \
    uv run --extra modal modal deploy serve_evaluation.py
```

The evaluator supplies temperature, per-turn token limit, stop tokens, and
prompts using the YAML config. The vLLM launcher sets capacity: 65,000 context
tokens (matching TextCraft training), up to 64 concurrent sequences per GPU, and LoRA
rank capacity 32. The default base model is `Qwen/Qwen3.5-4B`; the existing
`SKYRL_BASE_MODEL` environment override also applies to this launcher.

Stop the evaluation deployment when done:

```bash
uv run --extra modal modal app stop scottyrl-evaluation
```

## Running Parts in Parallel

The three launch scripts are separate Modal apps, each with its own URL, GPUs, and checkpoint volume, so you can run one part on each of them at the same time without any extra setup. You only need a second app name when you want two runs on the *same* launcher, for example `p2a` and `p41`, which both use the synchronous server. Set `SKYRL_APP_NAME` when deploying either Number Search launcher:

```bash
SKYRL_APP_NAME=scottyrl-p2a uv run modal deploy serve_number_search_sync.py
SKYRL_APP_NAME=scottyrl-p41 uv run modal deploy serve_number_search_sync.py
```

Each app has its own endpoint and checkpoint/database volume. In separate terminals, set `TINKER_BASE_URL` to the corresponding deployment URL and launch one training run per app. Without `SKYRL_APP_NAME`, the default app names are unchanged. Use the same app name when resuming a run, and stop each app with `uv run modal app stop <app name>` when you are done.

## Trainer and Inference Settings

Each `serve_*.py` file combines `BASE_BACKEND_CONFIG` from [skyrl_server.py](skyrl_server.py) with its profile overrides. The settings below are documented for reference; you should not need to change them for any of the required runs. A server's settings live in these files, not in the client's training YAML, and a change only takes effect when you redeploy (a change to `BASE_BACKEND_CONFIG` affects all three servers).

These are the SkyRL backend configuration flags:

| Flag | Meaning: value used here |
|---|---|
| `trainer.placement.policy_num_nodes` | no. of GPU nodes to use: `1` |
| `trainer.policy.megatron_config.{tensor,pipeline,context}_model_parallel_size` | Megatron model, pipeline and context parallelism: `1` (a 4B model with LoRA fits on one GPU, so we only use data parallelism) |
| `trainer.placement.policy_num_gpus_per_node` | no. of GPUs to allocate to the Megatron training backend: `2` for synchronous RL + number search, `1` for the others. |
| `trainer.placement.colocate_all` | share the same GPUs between trainer and inference engines: `True` for synchronous RL where training and sampling alternate, `False` for async RL. |
| `generator.inference_engine.num_engines` | no. of vLLM inference engines to host: `2` for synchronous RL + number search, `1` for async RL + number search, `3` for TextCraft. Equal to the no. of inference GPUs as we use data parallelism. |
| `trainer.policy.language_model_only` | train only the text backbone as Qwen3.5 is a multimodal LLM: `True` (our tasks are text-only) |
| `generator.inference_engine.language_model_only` | load only the text backbone for vLLM: `True` (our tasks are text-only) |
| `trainer.policy.megatron_config.lora_config.merge_lora` | merge the LoRA adapter into the base model: `False` (keep it separate for sampling and adapter checkpoints) |
| `trainer.policy.optimizer_config.max_grad_norm` | clip the gradient norms to this value: `1.0`. |
| `trainer.policy.optimizer_config.adam_betas` | decay rates for Adam's first- and second-moment estimates: `[0.9, 0.95]` |
| `trainer.policy.optimizer_config.weight_decay` | weight decay in Adam optimizer: `0.0` (disabled) |
| `trainer.use_expandable_segments` | PyTorch's expandable-segments allocator on the training workers, which reduces memory fragmentation: `True` |
| `trainer.max_tokens_per_microbatch` | max. number of tokens per training micro-batch for [dynamic batching](https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html): `32768` (SkyRL bin-packs a variable number of whole sequences into each micro-batch by token count instead of using a fixed number of sequences; this is a soft cap, a single sequence longer than this runs alone in its own micro-batch) |
| `trainer.fused_lm_head_logprob` | compute logprobs in chunks inside the output layer without storing the full-vocabulary logits: `True` for all parts (the unfused backward pass runs out of GPU memory at this micro-batch size) |
| `trainer.logprobs_chunk_size` | no. of tokens per logprob chunk: `1024`, explicitly set for the Number Search servers and the SkyRL default for TextCraft. |
| `generator.inference_engine.gpu_memory_utilization` | fraction of each inference GPU's memory that vLLM may use for weights, KV cache and activations: `0.7` for colocated setups (leave room for the trainer), `0.85` otherwise. |
| `generator.inference_engine.engine_init_kwargs.max_model_len` | context window in vLLM (max allowed prompt tokens + output tokens): `32768` for number search, `65000` for TextCraft. A request over this limit is rejected; the framework then abandons that episode with reward 0 and the run carries on (`scottyrl/rollout.py`). |
| `generator.inference_engine.max_num_batched_tokens` | max. allowed tokens in a single inference batch for each vLLM engine: `50000` (50K) for all server configs (independent of the training microbatch budget) |
| `generator.inference_engine.max_num_seqs` | max. allowed sequences in a single inference batch for each vLLM engine: `256` |
| `generator.inference_engine.enable_prefix_caching` | enable KV-caching in vLLM for faster inference: `True` |
| `generator.inference_engine.use_expandable_segments` | the same allocator setting for the vLLM processes: `False` |
| `generator.inference_engine.enforce_eager` | disable [CUDA graphs](https://github.com/vllm-project/vllm/issues/4449) and use eager PyTorch execution for inference (can reduce memory usage but is slower): `False` |
| `generator.inference_engine.engine_init_kwargs.gdn_prefill_backend` | kernel backend to use for Qwen3.5's Gated-DeltaNet prefill: `"triton"` |
| `generator.inference_engine.engine_init_kwargs.compilation_config` | CUDA graph configuration: `{"cudagraph_mode": "FULL_DECODE_ONLY"}` (decoding only) |
| `generator.inference_engine.enable_ray_prometheus_stats` | expose inference statistics through Ray's Prometheus integration: `True` |
| `generator.inference_engine.engine_init_kwargs.enable_mfu_metrics` | log model FLOPs utilization during inference: `True` |

Tinker requests supply the learning rate; the optimizer flags above are configured on the server.

Settings for the three parts:

| Profile | GPU placement | vLLM memory fraction | Context limit | Fused logprobs | Rationale |
|---|---|---|---|---|---|
| Number Search sync | 2 H100s shared | `0.7` | `32768` | `True` | Training and sampling alternate; leave memory for both. |
| Number Search async | 1 training + 1 inference H100 | `0.85` | `32768` | `True` | Separate GPUs for concurrent training and sampling. |
| TextCraft | 1 training + 3 inference H100s | `0.85` | `65000` | `True` | Longer episodes need more context and sampling capacity. |

If you ever change GPU placement, keep it consistent with Modal's GPU allocation. All three servers also request 8 CPUs and 32 GiB RAM, with a 12-hour server timeout and a five-minute idle scaledown window.

## What It Costs

At Modal's list price of roughly $4 per H100-hour, each Number Search server costs about $8 per hour and the TextCraft server about $16 per hour while a container is up. Please check [Modal's pricing](https://modal.com/pricing) for current numbers. A few habits keep the bill small:

- Get your code passing the tests locally before you deploy anything. The tests need no GPU.
- Do a [short probe](../ASSIGNMENT.md#do-a-short-probe-first) before every full run. A crash at step 2 of a probe is much cheaper than a crash at step 40 of the real thing.
- Stop the app when you step away, and never leave the TextCraft server up overnight.
