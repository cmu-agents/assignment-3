# scottyrl

<p align="center">
  <img
    src="assets/scottyrl-logo.webp"
    alt="ScottyRL mascot: a robotic Scottish terrier in a red tartan bandana leaping to catch a frisbee"
    width="240"
  >
</p>

A minimal framework for training LLM agents with reinforcement learning,
built for **11-768 (AI Agents), Assignment 3**. 

Authors: Apurva Gandhi, Aditya Soni and Demi Wang, with feedback from 11-768 course staff (instructors Daniel Fried and Graham Neubig, and TAs Saujas Vaduguru and Andy Liu).

**Instructions for the assignment are in [ASSIGNMENT.md](ASSIGNMENT.md)**:

## Setup

[uv](https://docs.astral.sh/uv/) with Python 3.12 (the project is tested
with 3.12; `pyproject.toml` accepts 3.11 or newer):

```bash
uv python install 3.12
uv sync                        # installs everything needed to code and test
```

Run every command through `uv run` from the repo root. The public tests,
the environments, `MockPolicy` rollouts and the whole data pipeline run
offline. Training and evaluation need a Tinker-compatible backend: for this
assignment you deploy your own SkyRL server on Modal and point the scripts
at it with `TINKER_BASE_URL` (see `skyrl_tinker_modal/README.md`). wandb is optional
(`--set train.wandb_project=<project>` with `WANDB_API_KEY` exported); the
jsonl logs under `runs/` are always written and are what is graded.

## Repo tour

```
scottyrl/
├── ASSIGNMENT.md           the handout: parts, background, grading, tips
├── README.md               this file
├── pyproject.toml          dependencies (tinker is pinned to 0.24.0)
├── scottyrl/
│   ├── types.py            Task / LLMCall / Trajectory / TokenSequence; read it first
│   ├── config.py           ModelConfig / TrainConfig, the YAML loader, --set overrides, the run-name guard
│   ├── policy.py           TinkerPolicy (real), CompletionsServerPolicy (vLLM/SGLang, exported adapters) and MockPolicy (tests, no key)
│   ├── chat_templates/
│   │   └── qwen3_5.jinja   the Qwen3.5 chat template with the enable_thinking / preserve_thinking_history knobs
│   ├── agents/
│   │   ├── base.py         Agent interface (async reset / step / close) + AgentAction
│   │   ├── xml_tool_call.py  XMLToolCallAgent: one <tool>...</tool> call per turn, prompt kept in token space
│   │   └── __init__.py     agent registry (xml_toolcall)
│   ├── envs/
│   │   ├── base.py         async Env interface; the env owns the step budget
│   │   ├── number_search.py  Number Search: guess a hidden integer with too-low / too-high feedback
│   │   ├── textcraft/
│   │   │   ├── env.py        TextCraft-Synth: craft target items up a recipe DAG
│   │   │   └── recipe_book.py  loads the shipped recipe book (data/textcraft_recipes.json)
│   │   └── __init__.py     env registry (number_search, textcraft)
│   ├── rollout.py          one episode: agent <-> env, and the transcript buffer the logger drains
│   ├── evaluate.py         success rate / pass@k on held-out tasks
│   ├── metrics.py          MetricsLogger (jsonl, transcripts, wandb), pass_at_k, sanity_warnings
│   ├── advantages.py     ★ REINFORCE / GRPO advantages
│   ├── datagen.py        ★ trajectory -> stitched, masked token sequences
│   ├── tinker_adapter.py ★ submit_update only; the rest of TinkerSession is given
│   └── train/
│       ├── common.py     ★ group rollouts + batch assembly (shared by both trainers)
│       ├── sync.py       ★ synchronous trainer (SyncTrainer)
│       └── async_grpo.py ★ asynchronous producer-consumer trainer (AsyncGRPOTrainer)
├── configs/
│   ├── p2a_number_search_reinforce.yaml       one YAML per required run: p2a, p2b, p3,
│   ├── ...                                    p41, p43_*_iw, p43_*_noiw, p5
│   └── required_runs.yaml  run_name -> config for the seven required runs
├── scripts/
│   ├── train_sync.py       entry point for the synchronous runs (p2a, p2b, p41)
│   ├── train_async.py      entry point for the asynchronous runs (p3, p43_*, p5)
│   ├── evaluate_checkpoint.py  evaluate a checkpoint, a served adapter, or the base model on a val split
│   ├── export_checkpoint.py    export the Part 5 adapter to a public Hugging Face repo
│   └── package_submission.py   check the required runs and write submission.zip
├── data/                   shipped task splits + recipe book; provenance in data/README.md
├── tests/                  the public suite: test_envs.py, test_interfaces.py, test_contracts.py
├── assets/                 the logo
└── skyrl_tinker_modal/     the SkyRL Tinker servers you deploy on Modal, one launch script per kind of run
```

**★ marks the six files with code you implement.** In `tinker_adapter.py`
only `TinkerSession.submit_update` is yours. Every stub is marked
`TODO(student)` and raises `NotImplementedError`; keep the signatures (the
hidden suite calls the stubs directly) and add private helpers as you like.
Everything else is given: the autograder replaces given files with pristine
copies, so please do not edit other parts of the codebase for the assignment.


## Running

```bash
uv run pytest                                    # the public suite

# Part 2: synchronous REINFORCE and GRPO on Number Search
uv run python scripts/train_sync.py  --config configs/p2a_number_search_reinforce.yaml
uv run python scripts/train_sync.py  --config configs/p2b_number_search_grpo.yaml

# Part 3: asynchronous GRPO
uv run python scripts/train_async.py --config configs/p3_number_search_async.yaml

# Part 4: ablations
uv run python scripts/train_sync.py  --config configs/p41_number_search_grpo_g2.yaml
uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_iw.yaml
uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_noiw.yaml

# Part 5: asynchronous GRPO on TextCraft-Synth, then download and serve the local adapter (see ASSIGNMENT.md)
uv run python scripts/train_async.py --config configs/p5_textcraft_grpo.yaml

# Deploy serve_evaluation.py as described in ASSIGNMENT.md, then:
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --server "$EVAL_SERVER" --model a3 --k 4
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --server "$EVAL_SERVER" --model a3 --k 4 --tasks data/textcraft_val_hard.jsonl
# Repeat both with --model Qwen/Qwen3.5-4B for the base-model results.

# Export the final Part 5 adapter to a public Hugging Face repo (after `hf auth login`)
uv run python scripts/export_checkpoint.py --run p5_textcraft_grpo

# When all seven runs are done: check them and write submission.zip
uv run python scripts/package_submission.py
```

`EVAL_SERVER` is the URL printed by `serve_evaluation.py`; see
[the deployment instructions](skyrl_tinker_modal/README.md#standalone-evaluation-with-vllm).
This inference-only server loads your saved adapter via vLLM. Use `--model Qwen/Qwen3.5-4B` for the base model.
Graders cannot read your `tinker://` checkpoints, so the Part 5 adapter is
submitted as a public Hugging Face repo named
`<hf-username>/scottyrl-a3-p5_textcraft_grpo`: `scripts/export_checkpoint.py`
creates it and records the id in the checkpoints file as `hf_model_id`,
and `scripts/package_submission.py` puts it in `submission/manifest.json`
(and refuses to package without it). To evaluate the exported adapter
yourself, serve it with vLLM and pass `--server http://localhost:8000/v1
--model a3` to `scripts/evaluate_checkpoint.py` (the exact commands are in
that script's docstring). Any config field can be overridden with
`--set section.key=value` (repeatable), for example
`--set train.num_steps=10 --set train.run_name=probe_p2b` for a cheap probe.
Keep the provided `run_name` values for the required runs:
`scripts/package_submission.py` and the graders look for exactly the files
named by `configs/required_runs.yaml`.

## Resume

Everything a run writes is named after its `train.run_name`. Its per-step
log goes to `runs/<run_name>.jsonl`, transcripts of the captured steps go to
`runs/<run_name>_trajectories/step_XXXXXX/`, what it saved on Tinker is
listed in `runs/<run_name>.checkpoints.json`, and its last usable checkpoint
is recorded in `runs/<run_name>.resume.json`. To protect those results, a
`run_name` that already has output is refused. The error message offers
three ways forward: pick a new name (`--set train.run_name=<name>`) for a
new experiment, pass `--set train.resume=true` to continue the same run
after a crash, or delete the run's files under `runs/` to redo it from
scratch.

Resuming relies on checkpoints the run saves as it goes. Every
`train.resume_save_every` optimizer steps (10 in the provided configs), it
saves its weights and optimizer state, and that checkpoint becomes usable
once the step that took it has been logged.

When you resume, training picks up from the last usable checkpoint. Any
log records and transcript directories written after that checkpoint are
removed, and those steps are trained and logged again, so the final log
still has every step exactly once. The resumed run otherwise carries on as
if nothing happened: step numbers stay absolute, the step-0 baseline
evaluation is not repeated, and periodic evaluations keep their cadence (a
run resumed at step 45 with `eval_every: 10` evaluates next at 50). The one
visible difference is in snapshot names. The crashed process may already
have used the plain names on the service, so the resumed process adds an
`_r1` suffix (`step_000041_r1`, and `_r2` after a second resume).

## Training backend

`skyrl_tinker_modal/README.md` describes how to deploy the SkyRL Tinker
server for each part on Modal. `export TINKER_BASE_URL=<endpoint>` points
the training, evaluation and export scripts at your deployment.
For parallel Number Search experiments, deploy separate apps with
`SKYRL_APP_NAME` and run one trainer per app. See
[Running Parts in Parallel](skyrl_tinker_modal/README.md#running-parts-in-parallel)
for deployment commands.
