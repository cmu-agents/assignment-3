"""Synchronous RL training loop. *** STUDENT IMPLEMENTATION FILE ***

Parts 1 & 2 of the assignment. The same loop handles REINFORCE
(algo="reinforce", group_size=1) and GRPO (algo="grpo", group_size>=2); the
only difference is the advantage computation you already wrote.

The trainer is split into small helper methods with fixed signatures (the
grading suite calls them individually); `train()` composes them.
"""

from __future__ import annotations

import asyncio  # noqa: F401  (student dependency)
import random
from collections.abc import Callable

from scottyrl.config import ModelConfig, TrainConfig
from scottyrl.envs.base import Env
from scottyrl.evaluate import evaluate_policy  # noqa: F401  (student dependency)
from scottyrl.metrics import (
    MetricsLogger,
    pass_at_k,
    sanity_warnings,
    sequence_batch_stats,
)
from scottyrl.policy import TinkerPolicy, load_chat_template
from scottyrl.tinker_adapter import TinkerSession
from scottyrl.train.common import collect_group, groups_to_batch  # noqa: F401  (student dependency)
from scottyrl.types import Task, TokenSequence, Trajectory


class SyncTrainer:
    def __init__(
        self,
        session: TinkerSession,
        env_factory: Callable[[Task], Env],
        train_tasks: list[Task],
        val_tasks: list[Task],
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        logger: MetricsLogger,
    ):
        self.session = session
        self.env_factory = env_factory
        self.train_tasks = train_tasks
        self.val_tasks = val_tasks
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.logger = logger
        self.rng = random.Random(train_cfg.seed)

    # ------------------------------------------------------ given helpers --

    def make_policy(self, sampling_client) -> TinkerPolicy:
        """Wrap a sampling client in a TinkerPolicy (given)."""
        return TinkerPolicy(
            sampling_client,
            self.session.tokenizer,
            max_tokens=self.model_cfg.max_tokens,
            temperature=self.model_cfg.temperature,
            chat_template=load_chat_template(self.model_cfg.chat_template),
            enable_thinking=self.model_cfg.enable_thinking,
            preserve_thinking_history=self.model_cfg.preserve_thinking_history,
            stop_sequences=self.model_cfg.stop_sequences,
        )

    def agent_cls(self):
        """The agent class named by cfg.agent (given)."""
        from scottyrl.agents import AGENT_REGISTRY

        return AGENT_REGISTRY[self.cfg.agent]

    def eval_due(self, step: int) -> bool:
        """Whether `step` is a periodic evaluation boundary (given).

        `step` is this process's iteration index (1, 2, ...). The logger's
        `step_offset` (the iterations an earlier process of this run
        completed; 0 in a fresh run) is added first, so evaluations stay
        at absolute multiples of cfg.eval_every: a run resumed at
        iteration 45 with eval_every=10 evaluates next at 50, not at 55.
        Returns False when cfg.eval_every is 0 (periodic evaluation
        disabled). In a fresh run this is `step % cfg.eval_every == 0`.
        """
        if not self.cfg.eval_every:
            return False
        return (step + getattr(self.logger, "step_offset", 0)) % self.cfg.eval_every == 0

    def record_training(
        self,
        step: int,
        groups: list[list[Trajectory]],
        seqs: list[TokenSequence],
        batch_stats: dict,
        train_metrics: dict,
    ) -> None:
        """Build and emit the training record for one step (given).

        Call this exactly once per training step, including a step whose
        batch had no sequences (pass seqs=[] and train_metrics={}). It
        merges `batch_stats` (from build_batch), `train_metrics` (from
        update), the batch's sequence statistics
        (metrics.sequence_batch_stats) and train-time success_rate /
        pass@k over `groups` (metrics.pass_at_k with k=cfg.group_size),
        adds "skipped": 1 when there was nothing to train on, prints every
        sanity warning, and logs each key prefixed "train/" so that train
        and eval curves for the same quantity share a base name.

        `step` is the training-iteration index, not a policy version. One
        record mixes metrics of different provenance, so at step t:

          - rollout-derived metrics (success_rate, pass@k, mean_reward)
            come from trajectories sampled BEFORE update t;
          - optimizer metrics (loss, grad_norm, importance weights)
            describe update t itself;
          - sequence statistics describe the batch update t consumed;
          - eval/* at step t, when evaluation runs after the update,
            describes the policy AFTER update t.

        Rollout-derived train metrics and eval metrics at the same step are
        therefore offset by one policy update (except on a skipped step,
        which performs no update). No train record is emitted at step 0;
        step 0 holds only the baseline evaluation of the initial policy,
        which the entry script records before train() runs.
        """
        seq_stats = sequence_batch_stats(seqs)
        # build_batch's stats and sequence_batch_stats both report these two
        # counts. A disagreement means build_batch is miscounting, so say so
        # instead of letting the merge below hide it.
        for key in ("num_sequences", "num_action_tokens"):
            if key in batch_stats and batch_stats[key] != seq_stats[key]:
                print(
                    f"[sanity] step {step}: {key} disagrees: build_batch reports "
                    f"{batch_stats[key]}, the batch actually has {seq_stats[key]}"
                )
        record = {**batch_stats, **train_metrics, **seq_stats}
        if not seqs:
            record["skipped"] = 1
        try:
            record.update(
                pass_at_k([[t.reward for t in group] for group in groups], k=self.cfg.group_size)
            )
        except ValueError as e:  # a metrics problem must never kill a run
            print(f"[metrics] step {step}: skipping pass@k: {e}")
        # sanity_warnings expects unprefixed keys, so check before prefixing.
        for warning in sanity_warnings(record):
            print(f"[sanity] step {step}: {warning}")
        self.logger.log(step, {f"train/{key}": value for key, value in record.items()})

    def record_evaluation(self, step: int, eval_metrics: dict) -> None:
        """Log `eval_metrics` at `step` with every key prefixed "eval/" (given)."""
        self.logger.log(
            step,
            {f"eval/{key}": value for key, value in eval_metrics.items()},
        )

    # ---------------------------------------------------- student helpers --

    async def collect_batch(self, policy) -> list[list[Trajectory]]:
        """Collect one training batch of trajectory groups.

        Contract: cfg.tasks_per_batch distinct tasks, drawn from
        self.train_tasks using self.rng (so runs are reproducible); one
        group of cfg.group_size rollouts per task, using the agent class
        from cfg; groups for different tasks are collected concurrently,
        not one task after another. 

        Returns: list of groups - one group (list of Trajectory) per sampled task.
        """
        # TODO(student): implement this method.
        raise NotImplementedError

    def build_batch(
        self, groups: list[list[Trajectory]]
    ) -> tuple[list[TokenSequence], dict]:
        """Assemble the training batch from the collected groups, with
        cfg.algo as the advantage method.

        Returns: (sequences, stats) with the same contract as your
        `groups_to_batch`. `train()` hands both to the given
        `record_training`, which merges `stats` into the step's record.
        """
        # TODO(student): implement this method.
        raise NotImplementedError

    async def update(self, seqs: list[TokenSequence]) -> dict:
        """Run one optimizer step on `seqs` through the session:
        `self.session.train_step`.

        Returns: the metrics dict reported by the training step, unchanged.
        `train()` hands it to the given `record_training`; do not log it
        here.
        """
        # TODO(student): implement this method.
        raise NotImplementedError

    async def evaluate_now(self, policy) -> dict:
        """Held-out evaluation of `policy`: the first cfg.eval_tasks
        validation tasks, cfg.eval_k rollouts each, using the agent class
        from cfg (see scottyrl.evaluate.evaluate_policy).

        Returns: the raw evaluation metrics dict. The given
        `record_evaluation` owns the "eval/" prefix and the logging; the
        entry script also calls this method for the step-0 baseline.
        """
        # TODO(student): implement this method.
        raise NotImplementedError

    # ------------------------------------------------------------- driver --

    async def train(self) -> None:
        """Run cfg.num_steps synchronous training steps, t = 1, ..., cfg.num_steps.

        A *weight snapshot* saves a copy of the current weights on the
        Tinker server and returns a sampling client that serves that frozen
        copy: `client = await self.session.new_sampling_client(name)`,
        wrapped by `self.make_policy(client)` to get the policy to sample
        from. Later updates do not change an existing weight snapshot, and
        every new_sampling_client call saves a new one, so take exactly the
        weight snapshots listed below. Weight snapshots are not resumable
        checkpoints: do not call `save_resumable_state`, which
        `session.train_step` handles itself.

        Before the loop: take a weight snapshot of the initial weights,
        named "step_000000". Its policy is the one step 1 samples from.

        Each step t:
          1. Collect a batch with the current policy (`collect_batch`) and
             build it (`build_batch`).
          2. If the batch has sequences: `update` on it, then take a weight
             snapshot named f"step_{t:06d}" and switch to its policy, so
             step t+1 samples from the weights this update produced
             (on-policy).
             If it has none (every group degenerate), the step is
             *skipped*: no update and no weight snapshot, so the policy is
             unchanged and step t+1 samples from it again. Snapshot names
             can therefore have gaps and are not a count of updates.
          3. Call the given `record_training` exactly once, skipped or not
             (pass seqs=[] and train_metrics={} for a skipped step).
          4. If the given `self.eval_due(t)` is true, `evaluate_now` the
             current policy, i.e. the one after this step's update (or the
             unchanged one on a skipped step), and pass the result to the
             given `record_evaluation(t, metrics)`. `eval_due` counts on
             absolute step numbers so a resumed run keeps the cadence, and
             returns False when cfg.eval_every is 0.

        `new_sampling_client`, `collect_batch`, `update` and `evaluate_now`
        are coroutines; await them. The step-0 baseline evaluation of the
        untrained policy is done for you by scripts/train_sync.py before
        train() is called; do not repeat it here.
        """
        # TODO(student): implement this method.
        raise NotImplementedError
