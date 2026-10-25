# Assignment 3: Training Agents with Reinforcement Learning

11-768 AI Agents, CMU, Fall 2026.

## Overview

In this assignment, you will help build `scottyrl`, a minimal training framework for training agents with reinforcement learning. In particular, you will:

- Build the trajectory-processing and advantage-computation pipeline.
- Implement synchronous REINFORCE and GRPO training.
- Implement asynchronous GRPO with stale-rollout handling.
- Compare algorithms and system designs through controlled ablations.
- Train and evaluate an agent on TextCraft-Synth.

**Start early!** This assignment is heavier and is expected to take longer than previous assignments both in terms of implementation and wait time for the various training runs you will launch.

## Tinker API Compatibility

`scottyrl` builds on top of and is fully compatible with the Tinker API. We recommend looking through its documentation and referencing it as you work through the assignment to familiarize yourself with the training and sampling primitives that the Tinker SDK exposes: https://tinker-docs.thinkingmachines.ai/tinker/quickstart/.

Targeting the Tinker API allows great flexibility by decoupling the higher-level aspects of your RL framework (training loop, agent harness, environments, data processing, reward design, advantage computation, etc.) from lower-level distributed GPU execution and state by encapsulating the latter into a service that implements a scoped API (`forward_backward`, `optim_step`, `sample`, etc.).

Importantly, this allows the framework to be modular and highly flexible: `scottyrl` can work with any training backend that implements a Tinker-like service. For example:

Hosted Services (Requires API Credits):
- [Thinking Machines' official Tinker service](https://tinker-docs.thinkingmachines.ai/tinker/)
- [Fireworks' Tinker-compatible Training Service](https://docs.fireworks.ai/fine-tuning/training-api/introduction)

Self-Hosted Services (Bring Your Own Compute and Host a Tinker-compatible backend service on your own GPU nodes):
- [SkyRL's Tinker-compatible Training Service](https://docs.skyrl.ai/docs/tinker/overview) (supports full fine-tuning and LoRA)
- [Miles' Tinker-compatible Training Service](https://github.com/radixark/miles/issues/3284); experimental -- supports LoRA only for now

For Assignment 3, you will self-host using SkyRL's open-source Tinker service implementation and [Modal](https://modal.com/) compute.

If you would like to use this framework or build on top of it for your final projects, you are free to experiment with other service options depending on the compute or API credits you have access to.

## Self-Hosting with SkyRL and Modal

Please see the instructions [here](skyrl_tinker_modal/README.md) on how to set up and configure a backend training service for each of the problems.

Make sure to read this before starting training for the different problems, as we recommend different configurations for the different training runs you will perform.

## Setup

Install the project with [uv](https://docs.astral.sh/uv/) and Python 3.12:

```bash
uv python install 3.12
uv sync
uv run pytest
```

On the starter code, 12 of the 25 tests pass and the rest fail with `NotImplementedError`. They will start passing as you fill in your implementation. Run all commands through `uv run` from the repo root. The tests and the entire data pipeline run locally, without a GPU or a backend service.

When you are ready to launch a training run, deploy the SkyRL service for that part and set `TINKER_BASE_URL` as described in the [self-hosting guide](skyrl_tinker_modal/README.md). There are three launch scripts, each sized for the runs it serves, and every part below tells you which one to use. Each server takes one run at a time, but the three servers are independent, so you can run different parts [at the same time](skyrl_tinker_modal/README.md#running-parts-in-parallel). Get your code passing the tests before you train: a server bills for its GPUs from the moment a run connects, so a crash you could have caught locally still costs money.

All training runs in this assignment use `Qwen/Qwen3.5-4B` with LoRA (rank 32). Every required run has a ready-made config in [configs/](configs/).

## Environments

You will train agents on two environments. In both, the agent acts by emitting one tool call per turn, written as `<tool>name(args)</tool>`, and receives a text observation in return. An episode ends when the task is solved, the agent calls `finish()` (TextCraft-Synth), or the step budget runs out. The reward is binary: 1 for success and 0 otherwise.

**Number Search** is a small, cheap environment that you will use for most of the assignment. The agent has to find a hidden number in `[low, high]` with a single tool, `guess(n)`, which replies too low, too high, or correct. It gets $\lceil \log_2(\text{high}-\text{low}+2) \rceil$ guesses: the exact worst-case binary-search budget, including the final correct guess, with no slack. There are 50,000 training tasks and 1,000 validation tasks. All parts use the same `xml_toolcall` agent. Reasoning is controlled by the YAML chat-template settings: `model.enable_thinking: false` for Number Search and `true` for TextCraft. Both set `model.preserve_thinking_history: true` to retain prior thinking blocks, including empty ones.

**TextCraft-Synth** is a harder crafting environment that you will use in the final part. The agent starts with an inventory and a list of target items, and has four tools: `craft`, `get_info`, `view_inventory`, and `finish`. Recipes must be followed exactly, and raw materials only come from the initial inventory. The agent is rewarded only if it crafts all the target items *and* calls `finish()` before it runs out of steps. There are 852 training tasks, 213 validation tasks of medium difficulty, and 136 hard validation tasks that we use to test generalization in Part 5. This environment is variant of the version first introduced in the paper [Recursive Agent Optimization (RAO)](https://arxiv.org/abs/2605.06639). 

A big difference in your training runs for these environments is that we will disable chain-of-thought (CoT) for Number Search but enable it for Textcraft-Synth, making the Textcraft runs much longer.

The datasets live in [data/](data/) (see [`data/README.md`](data/README.md)). Please do not edit or regenerate them.

## Expected Cost and Time per Run

The two environments have very different budgets, so plan your runs around them:

| Environment | Approx. cost per run | Approx. time per run | Intended use |
|---|---|---|---|
| Number Search | ~$10 | ~40 min | Cheap and fast: use it for debugging, testing your implementation, and the ablations. |
| TextCraft-Synth | ~$100 | 6-8 hours | Expensive and long: a single **hero run** for Part 5. |

**Treat TextCraft-Synth as a hero run.** You should aim to launch it only once, after you have used Number Search to debug and validate your trainers end to end. Do not use TextCraft-Synth for debugging or repeated attempts. Once the run starts, keep an eye on its metrics and transcripts (see [Standard Diagnostic Plots](#standard-diagnostic-plots) and [Sanity Warnings](#sanity-warnings)) so that you catch errors early and stop a broken run before it burns through your budget.

## Codebase Overview

A good place to start reading is `scottyrl/types.py`, which defines the objects that flow through the framework:

1. An agent (`scottyrl/agents/xml_tool_call.py`) interacts with an environment for one episode (`scottyrl/rollout.py`). This produces a `Trajectory`: every LLM call of the episode, recorded in token space, plus the final reward.
2. The data pipeline (`scottyrl/advantages.py`, `scottyrl/datagen.py`) turns trajectories into `TokenSequence`s: training sequences with a loss mask, sampling logprobs, and advantages.
3. A trainer (`scottyrl/train/`) collects rollouts, builds batches, and sends them to the backend through `TinkerSession` (`scottyrl/tinker_adapter.py`), which wraps the Tinker training API.

The [README](README.md) has a full map of the repository. You will implement the following six files, in this order:

| File | What you implement | Part |
|---|---|---|
| `scottyrl/advantages.py` | `compute_group_advantages` | 1 |
| `scottyrl/datagen.py` | `extension_suffix`, `stitch_trajectory`, `trajectory_to_training_sequences` | 1 |
| `scottyrl/train/common.py` | `collect_group`, `groups_to_batch` | 2 |
| `scottyrl/tinker_adapter.py` | `TinkerSession.submit_update` (the rest of the file is given) | 2 |
| `scottyrl/train/sync.py` | `SyncTrainer`: `collect_batch`, `build_batch`, `update`, `evaluate_now`, `train` | 2 |
| `scottyrl/train/async_grpo.py` | `AsyncGRPOTrainer`: `generate_group`, `rollout_worker`, `collect_fresh_batch`, `update`, `evaluate_now`, `train` | 3 |

Everything else is given code, and we restore it before grading, so only your changes to these six files count. Treat each stub's docstring as the specification. Keep the function signatures unchanged; you are free to add private helpers.

The public tests in `tests/` check that your code plugs into the framework correctly. Passing them does not mean your code is fully correct: we grade with a more thorough hidden test suite.

### Chat Template and Sequence Extension

Prompts are rendered with a chat template (`scottyrl/chat_templates/qwen3_5.jinja`) adapted from Qwen3.5's own. An episode involves several LLM calls, and we would like to train on the whole episode as *one* sequence rather than one sequence per call. This is possible when each call's prompt is an exact extension of the previous prompt plus its completion (see [Sequence Extension](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/sequence-extension/)). The agent builds its context in token space to preserve it.

Two config fields control how prompts are rendered:

- `model.enable_thinking`: whether CoT is enabled. It is off for Number Search and on for TextCraft-Synth.
- `model.preserve_thinking_history`: whether the CoT from earlier turns stays in the prompt. When `true` (the default), sequence extension normally holds and an episode becomes a single training sequence (it can still break occasionally, for example when the model stops on `<|endoftext|>` instead of `<|im_end|>`). When `false`, earlier `<think>` blocks are stripped to save context. This breaks the extension, so each turn becomes its own training sequence. Your data pipeline needs to handle both cases.

### Running Experiments

Each run is described by a YAML config with a `model:` and a `train:` section. You can override any field from the command line by adding `--set section.field=value`, for example `--set train.lr=2e-5` (see [Do a Short Probe First](#do-a-short-probe-first) for a full command).

A run is identified by `train.run_name` and writes everything under `runs/`:

| Artifact | Path |
|---|---|
| Per-step metrics | `runs/<run_name>.jsonl` |
| Rollout transcripts | `runs/<run_name>_trajectories/step_XXXXXX/` |
| Saved checkpoints | `runs/<run_name>.checkpoints.json` |

Step 0 of every log holds a baseline evaluation of the untrained model. The provided configs save transcripts every five steps. Read them! They are the fastest way to see what your agent is actually doing.

Keep the provided run names for the required runs, since grading looks for them. A run name that already has output is rejected so that you cannot overwrite results by accident. Pick a new name for a new experiment, or pass `--set train.resume=true` to continue a crashed run from its last checkpoint.

### Logging to Weights & Biases (Optional)

Every run writes its metrics to the JSONL log, and that log is what we grade. We still recommend also logging to [Weights & Biases](https://wandb.ai): it gives you live charts of every metric while a run is going, which makes it much easier to keep an eye on long runs and to compare runs side by side. To enable it, log in once and add `--set train.wandb_project=<project>` to your training command:

```bash
uv run wandb login
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml \
    --set train.wandb_project=11768-a3
```

Each run appears under its `run_name`, and a resumed run continues the same W&B run. If W&B cannot start (for example, you are not logged in), the run prints `[metrics] wandb disabled` and continues with the JSONL log only, so check for that line at startup.

### Standard Diagnostic Plots

Every experimental report section (Part 2, Part 3, each Part 4 subsection, and Part 5) should include plots for the following metrics against training step for every training run it discusses:

- reward quality: `train/success_rate`, `eval/success_rate`, and every logged `train/pass@*` and `eval/pass@*`;
- optimization: `train/grad_norm`;
- policy drift: `train/importance_weight_mean`, `train/importance_weight_max`, and `train/importance_weight_std`;
- data health: `train/frac_degenerate_groups` and `train/entropy_proxy`.

You may combine related metrics and runs into readable multi-panel figures. Label each pass@ curve with its exact key: training and evaluation often use different values of $k$, so they are distinct quantities.

### Timing Metrics

Every training record also carries timing metrics, in seconds, for the interval since the previous training record. They are logged by the given code, so you do not need to implement anything, and they are not part of the standard diagnostic plots. Use them to understand where a run spends its time and to spot slowdowns early.

| Metric | What it measures |
|---|---|
| `train/step_time_s` | Wall-clock length of the step. |
| `train/update_time_s` | Time spent in the optimizer update, including the periodic resumable checkpoint. |
| `train/learner_wait_s` | Step time minus the update and the sampler snapshot save: time the learner spent on everything else. |
| `train/learner_wait_frac` | `learner_wait_s` as a fraction of `step_time_s`. |
| `train/rollout_time_s` | From the start of the earliest to the end of the latest training episode that finished during the step. |
| `train/episode_time_mean_s` | Mean duration of those episodes. |


## Background

### From an Episode to a Gradient

Let a stitched episode contain tokens $t_0,\ldots,t_{n-1}$, and let $m_j \in \{0,1\}$ mark whether target token $t_{j+1}$ was sampled by the policy, as opposed to coming from the prompt or the environment. With an episode-level advantage $A$, the policy gradient is

$$\nabla_\theta J(\theta) = \mathbb{E}\Big[\, A \sum_{j:m_j=1} \nabla_\theta \log \pi_\theta(t_{j+1}\mid t_{\le j}) \Big].$$

Only the tokens the policy sampled belong in the loss. If the mask includes observation tokens, the model learns to predict the environment instead of learning to act.

### REINFORCE and GRPO

REINFORCE uses $A = r$: every sampled token of a successful episode is reinforced, and failed episodes contribute nothing. The estimate is unbiased but noisy, especially with sparse rewards.

GRPO samples a group of $G$ rollouts for the same task and uses the group's mean reward as a baseline:

$$A_i = r_i - \frac{1}{G}\sum_{k=1}^{G} r_k.$$

Subtracting the baseline removes the effect of task difficulty, so that a rollout is reinforced only if it did better than the policy's typical attempt at the same task. If every reward in a group is identical, all advantages are zero and the group carries no learning signal. We call such a group *degenerate* and skip it. Following [Dr.GRPO](https://arxiv.org/abs/2503.20783), we do not divide by the group's reward standard deviation, and we do not add a KL penalty to a reference policy.

In the configs, `train.algo` chooses the advantage rule and `train.loss_fn` separately chooses the token-level loss described next.

### Asynchronous Training and Importance Weighting

Synchronous training alternates between collecting rollouts and updating the model, so every batch is on-policy, but the learner sits idle during rollouts and the rollout workers sit idle during updates. Asynchronous training overlaps the two. This is faster, but a group may now have been sampled by a policy that is several updates old. We deal with this *staleness* in two ways: groups older than `train.max_staleness` versions are dropped, and the rest are corrected with importance weighting.

For a token sampled from $\pi_{\text{old}}$ and trained under $\pi_\theta$, the importance ratio is

$$\rho_j = \exp\big(\log \pi_\theta(t_{j+1}\mid t_{\le j}) - \log \pi_{\text{old}}(t_{j+1}\mid t_{\le j})\big).$$

The framework supports three losses:

- `ppo`, the clipped surrogate: $-\sum_{j:m_j=1} \min(\rho_j A_j,\ \mathop{\text{clip}}(\rho_j, 1-\epsilon, 1+\epsilon)\,A_j)$
- `importance_sampling`, the same without clipping: $-\sum_{j:m_j=1} \rho_j A_j$
- `cross_entropy`, advantage-weighted log-likelihood with no ratio at all: $-\sum_{j:m_j=1} A_j \log \pi_\theta(t_{j+1}\mid t_{\le j})$

The GRPO runs use `ppo`, the REINFORCE run uses `importance_sampling`, and `cross_entropy` is only used for the "no correction" ablation in Part 4.

Statistics of $\rho$ are logged for every update. In a synchronous run they should stay very close to 1. If they do not, your sampling logprobs are most likely attached to the wrong tokens.

## Part 0: Warmup (0 points)

Before writing any code, familiarize yourself with the agent, rollout and environment implementations: `scottyrl/agents/xml_tool_call.py`, `scottyrl/rollout.py`, and `scottyrl/envs/`.

## Part 1: The Data Pipeline (20 points)

In this part you will turn agent trajectories into training data.

**Implement** `compute_group_advantages` in `scottyrl/advantages.py` and the three functions in `scottyrl/datagen.py`, following their docstrings.

Be careful about next-token alignment: A sequence of $n$ tokens has $n-1$ prediction targets, so `mask`, `logprobs`, and `advantages` all have length $n-1$, and position `j` describes the target `tokens[j+1]`. If a completion of length $c$ follows a context of length $L$, its positions are `L-1 : L-1+c`, not `L : L+c`.

When stitching, keep extending the current sequence for as long as the next call's prompt extends what you have accumulated, and start a new sequence when it does not. Only completion tokens get a mask of 1, together with their sampling logprob and the episode's advantage.

**Test** with `uv run pytest tests/test_interfaces.py -k "advantages or datagen"` (the other tests in that file belong to Part 2). The `ThinkStrippingPolicy` in `tests/test_envs.py` gives you an episode that breaks the extension, which is handy for testing that case locally.

## Part 2: Synchronous REINFORCE and GRPO (25 points)

Next, you will build the synchronous training loop and use it to train your first agents.

**Implement** the following:

- `collect_group` and `groups_to_batch` in `scottyrl/train/common.py`. The first runs the rollouts of one group concurrently, and the second computes advantages and flattens the groups into one training batch.
- `TinkerSession.submit_update` in `scottyrl/tinker_adapter.py`. Refer to the tinker documentation to understand how to use its API. 
- `SyncTrainer` in `scottyrl/train/sync.py`: four small helper methods and the `train` loop that ties them together. Make sure to refresh the sampling policy after every update so that the next batch is on-policy.

**Test** with `uv run pytest tests/test_interfaces.py tests/test_contracts.py -k "not async"`. The remaining `test_async_*` tests will pass once you finish Part 3.

**Run** both Number Search experiments with the provided configs unchanged, against the `serve_number_search_sync.py` deployment. We recommend a [short probe](#do-a-short-probe-first) as a sanity check before you do your full runs.

```bash
uv run python scripts/train_sync.py --config configs/p2a_number_search_reinforce.yaml
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml
```

`p2a` is REINFORCE with a single rollout per task, and `p2b` is GRPO with eight rollouts per task. Both train for 50 steps with eight tasks per step, and evaluate every 10 steps on 50 validation tasks with four rollouts each.

**Report:**

- Include the [standard diagnostic plots](#standard-diagnostic-plots) for both runs.
- Compare REINFORCE and GRPO in 3 to 5 sentences, in terms of learning speed and explain whether the difference is expected or not, and why. Working run rewards should rise clearly above its step-0 baseline within 50 steps. If your curve is flat, please debug your implementation before writing the analysis.

## Part 3: Asynchronous GRPO (25 points)

In this part you will reduce idle time from training with a producer-consumer design. Rollout workers continuously generate groups with the latest published policy and push them into a bounded queue. The learner pulls groups off the queue, trains on them, and publishes a new policy version.

**Implement** the six student methods of `AsyncGRPOTrainer` in `scottyrl/train/async_grpo.py`, following their docstrings. A few things to keep in mind:

- Each group should be generated with a single `(policy, version)` pair captured up front, so that the version it carries is still correct if a new policy is published halfway through. Staleness is then checked by the learner when it consumes the group, not by the workers.
- The queue is bounded on purpose. Let [`asyncio.Queue`](https://docs.python.org/3/library/asyncio-queue.html) provide the backpressure through `await queue.put(...)` and `await queue.get()`, rather than polling `full()` or `empty()`.
- Neither evaluation nor shutdown should be able to stall training. Run each evaluation as a background task on a captured `(policy, version)` (overlapping it with training). When training ends, [cancel](https://docs.python.org/3/library/asyncio-task.html#task-cancellation) and await the workers, and if anything fails, cancel everything else and let the original error propagate instead of hanging.

**Test** with `uv run pytest`: the whole public suite should now pass. The `test_async_*` tests in `tests/test_contracts.py` drive your trainer with fake rollouts and updates, so you can get the concurrency right without spending any compute.

**Run** the provided config unchanged, against the `serve_number_search_async.py` deployment (asynchronous training needs separate GPUs for training and sampling, so the Part 2 server will not do):

```bash
uv run python scripts/train_async.py --config configs/p3_number_search_async.yaml
```

`p3` matches `p2b` (eight groups of eight rollouts per step, 50 steps), but generates them with eight rollout workers and allows a staleness of up to 2. When the run finishes, it prints the totals returned by `train()` and also writes them to the last record of `runs/p3_number_search_async.jsonl` as `summary/train_steps`, `summary/groups_consumed` and `summary/groups_dropped_stale`. You will need `summary/groups_dropped_stale` in Part 4.2.

**Report:**

- Include the [standard diagnostic plots](#standard-diagnostic-plots) for `p3`. In your analysis, compare the importance-weight mean, minimum, maximum, and standard deviation; a mean near 1 can hide a heavy tail.
- Compare the wall-clock time per step with `p2b`, using `train/step_time_s` (or the difference between the `time` fields of consecutive training records), and quantify the speedup.
- Pick one early, one middle, and one late training rollout from the transcripts (give the step, task id, and the `policy_version` from the file header of each). Describe at least two concrete ways the agent's behavior changed, for example how it uses feedback, whether it bisects or scans, or how it recovers from an invalid call.

## Part 4: Ablations on Number Search (20 points)

Now use your trainers to run three small studies. For each one, write a short section in the report with the requested plots and a paragraph of analysis. Use the provided configs unchanged.

### 4.1 Group Size

Compare `p2a` ($G=1$, REINFORCE), `p41` ($G=2$, GRPO), and `p2b` ($G=8$, GRPO). The only new run is `p41`, which uses the same deployment as Part 2:

```bash
uv run python scripts/train_sync.py --config configs/p41_number_search_grpo_g2.yaml
```

**Report:** Include the [standard diagnostic plots](#standard-diagnostic-plots) for all three runs. Use `eval/success_rate` and `train/frac_degenerate_groups` to explain what a smaller group does to the variance of the advantages and to the number of useful sequences per step. Report the degenerate fraction as N/A for REINFORCE: it has no group baseline, so its logged value of 0 is not meaningful. Keep in mind that `p2a` also differs from the GRPO runs in its advantage rule and loss, so this is not a pure group-size ablation.

### 4.2 Synchronous vs. Asynchronous Training

**Report:** Include the [standard diagnostic plots](#standard-diagnostic-plots) for `p2b` and `p3`. Reuse your wall-clock analysis from Part 3 and add each run's final `eval/success_rate` (a small table is enough). Explain where the speedup comes from, and whether training on stale data cost anything in sample quality. Use the importance-weight statistics and the `summary/groups_dropped_stale` count from the last record of the `p3` log as evidence.

### 4.3 Importance Weighting under Higher Staleness

Run two async experiments that differ only in their loss, against the same deployment as Part 3:

```bash
uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_iw.yaml
uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_noiw.yaml
```

Both allow a staleness of up to 4 to make the effect visible. The `iw` run uses PPO, while `noiw` uses advantage-weighted cross-entropy, which ignores the importance ratio entirely.

**Report:** Include the [standard diagnostic plots](#standard-diagnostic-plots) for both runs.


## Part 5: TextCraft-Synth (10 points)

For the final part, you will train an agent on TextCraft-Synth with your asynchronous GRPO trainer in a single run with the provided config, then evaluate how well it does on both the medium and the hard validation split.

> **This is a hero run (~$100, 6-8 hours).** Launch it only after your async trainer works on Number Search, and monitor the metrics so you can stop early if something is wrong. See [Expected Cost and Time per Run](#expected-cost-and-time-per-run).

**Run** the provided config unchanged. It enables thinking with up to 512 tokens per turn. Each of its 50 steps trains on 16 groups of eight rollouts, generated by 32 rollout workers with a staleness of up to 4, and it evaluates every 10 steps. TextCraft episodes are much longer than Number Search ones and spend most of their time sampling, so this part has its own, larger deployment (`serve_textcraft.py`):

```bash
uv run python scripts/train_async.py --config configs/p5_textcraft_grpo.yaml
```

**Evaluate** the final adapter and the base model on the medium and hard
validation splits, with four rollouts per task. Read `final_sampler_path` from
`runs/p5_textcraft_grpo.checkpoints.json` to find the model ID and snapshot name
(e.g. `tinker://model_abc123/v50`). Download that sampler checkpoint and deploy it:

```bash
# From the repository root (substitute your model ID and snapshot name):
mkdir -p runs/p5_adapter
uv run --extra modal modal volume get scottyrl-textcraft-state \
    checkpoints/<model-id>/sampler_weights/<snapshot>.tar.gz runs/p5_adapter.tar.gz
tar -xf runs/p5_adapter.tar.gz -C runs/p5_adapter

cd skyrl_tinker_modal
LORA_ADAPTER=../runs/p5_adapter \
    uv run --extra modal modal deploy serve_evaluation.py
cd ..

# Copy the URL printed by deploy, for example https://your-workspace--scottyrl-evaluation-serve.us-east.modal.run:
export EVAL_SERVER=<printed_url>

# Fine-tuned model: medium, then hard
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --server "$EVAL_SERVER" --model a3 --k 4
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --server "$EVAL_SERVER" --model a3 --k 4 --tasks data/textcraft_val_hard.jsonl

# Base model: medium, then hard
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --server "$EVAL_SERVER" --model Qwen/Qwen3.5-4B --k 4
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --server "$EVAL_SERVER" --model Qwen/Qwen3.5-4B --k 4 --tasks data/textcraft_val_hard.jsonl

# Stop the inference deployment when finished:
uv run --extra modal modal app stop scottyrl-evaluation
```

For other serving options, see the
[serving guide](skyrl_tinker_modal/README.md#standalone-evaluation-with-vllm).

**Export** your final adapter to Hugging Face so that we can evaluate it ourselves. The script downloads the checkpoint through your `serve_textcraft.py` deployment, so it must be up. If you have already stopped it, redeploy it with the same command and keep `TINKER_BASE_URL` pointing at it; your checkpoints persist on its volume. Exporting right after training saves you that extra spin-up. The first command is a one-time login:

```bash
uv run hf auth login
uv run python scripts/export_checkpoint.py --run p5_textcraft_grpo
```

The script uploads the LoRA adapter to a public model repo named `<hf-username>/scottyrl-a3-p5_textcraft_grpo` and records it as `hf_model_id` in the checkpoints file. Add `--exist-ok` to update an existing export, and please keep the model public until grades are released.

**Report:**
- Include the [standard diagnostic plots](#standard-diagnostic-plots) for the `p5` training run.
- One table with four rows (fine-tuned and base model, on each split) showing `success_rate` and `pass@4`.
- Your analysis of the run. Describe your results in a few sentences.

## Tips and Debugging

### Do a Short Probe First

Before committing to a long run, try a short one under a throwaway name:

```bash
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml \
    --set train.num_steps=10 --set train.eval_every=2 --set train.eval_tasks=20 \
    --set train.run_name=probe_p2b
```

Note: If you only want to check that training runs without running evaluation, you can add `--set train.eval_every=0` (or, equivalently, `--set train.eval_tasks=0`) to disable evaluation entirely, including the step-0 baseline. The log will then contain only training records, so skip the evaluation parts of check 1 below.

Then check that:

1. The log has a baseline evaluation at step 0, one training record per step, and an evaluation every two steps.
2. The transcripts under `step_000005/` read like real episodes: a completion, a parsed tool call, and the environment's reply. A `null` tool call means the completion failed to parse. The one exception is a last turn that has a `stop_reason` and no completion: that episode lost a sampling call, for example because it outgrew the server's context window, and counts as a failure.
3. `train/importance_weight_mean` stays close to 1, as it should for a synchronous run.
4. You can explain every `[sanity]` line in the terminal output.

Note that a probe only checks that everything is wired up correctly. Ten steps are too few to tell whether the agent is learning.

### Sanity Warnings

During training, the framework prints a `[sanity]` warning when something looks off: collapsed entropy, importance weights far from 1, a gradient spike, or almost all groups being degenerate (the exact thresholds are in `scottyrl/metrics.py`). 

## Report

Submit a `report.pdf`, placed at the root of the repository. What to include is listed under each part. 
Make your plots readable (axes, legends, captions) and discuss every curve you show.

## AI Usage

Submit a file `AI_USAGE.md` detailing your use of any AI technologies for this assignment. List all the tools you used, and provide a clear description of how you used each of these tools. If you did not use any AI assistance, declare that in this file.

Place the file at the root of the repository, next to `report.pdf`. The packaging script includes it in your `submission.zip` and refuses to package without it.

## Submission

Once all seven required runs and the Part 5 export are done, package your submission:

```bash
uv run python scripts/package_submission.py
```

The script prints a checklist and writes `submission.zip`, which contains your six source files, the config, log, transcripts, and checkpoints file of every required run, your `report.pdf`, your `AI_USAGE.md`, and a manifest that records the `hf_model_id` of your Part 5 adapter. It refuses to package an incomplete submission and tells you what is missing.

Upload `submission.zip` to Canvas. Late days follow the course syllabus.

## Grading

| Part | Points |
|---|---|
| 1 | 20 |
| 2 | 25 |
| 3 | 25 |
| 4 | 20 |
| 5 | 10 |
| | **100** |
