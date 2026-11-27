"""Executable examples (given) of conventions the hidden suite grades.

Each test pins one documented contract using a small fake in place of the
Tinker service. A test fails with NotImplementedError until the method it
exercises is implemented; after that it shows what a correct implementation
must do. Passing here is necessary, not sufficient: the hidden suite checks
the same contracts more thoroughly.
"""

import asyncio

import pytest

from scottyrl.config import ModelConfig, TrainConfig
from scottyrl.tinker_adapter import TinkerSession
import scottyrl.train.sync as sync_module
from scottyrl.train.async_grpo import AsyncGRPOTrainer, GroupResult
from scottyrl.train.sync import SyncTrainer
from scottyrl.types import LLMCall, Task, TokenSequence, Trajectory

# ------------------------------------------------------- submit_update --


class SdkFailure(Exception):
    """A fake SDK error; must reach submit_update's caller unchanged."""


class RecordingFuture:
    """Stands in for the APIFuture the SDK returns from a submission."""

    def __init__(self, client, operation, result, fail):
        self.client = client
        self.operation = operation
        self.result = result
        self.fail = fail

    async def result_async(self):
        # Checked at the moment a result is resolved, so it does not depend
        # on timing and allows either resolution order.
        assert self.client.submitted == {"forward", "optim"}, (
            "submit both operations before resolving either result"
        )
        self.client.events.append(f"resolve_{self.operation}")
        if self.fail:
            raise SdkFailure(f"{self.operation}_result")
        return self.result


class RecordingTrainingClient:
    """Records submissions and resolutions; can fail at any of four stages."""

    def __init__(self, failure_stage=None):
        self.failure_stage = failure_stage
        self.fwd_result = object()
        self.opt_result = object()
        self.submitted = set()
        self.events = []
        self.calls = {}

    async def forward_backward_async(self, data, loss_fn):
        self.events.append("submit_forward")
        self.calls["forward"] = (data, loss_fn)
        if self.failure_stage == "forward_submit":
            raise SdkFailure("forward_submit")
        self.submitted.add("forward")
        return RecordingFuture(
            self, "forward", self.fwd_result, self.failure_stage == "forward_result"
        )

    async def optim_step_async(self, adam_params):
        self.events.append("submit_optim")
        self.calls["optim"] = adam_params
        if self.failure_stage == "optim_submit":
            raise SdkFailure("optim_submit")
        self.submitted.add("optim")
        return RecordingFuture(
            self, "optim", self.opt_result, self.failure_stage == "optim_result"
        )


def make_session(client) -> TinkerSession:
    session = TinkerSession("fake-model")
    session.training_client = client
    return session


def test_submit_update_submits_both_operations_before_resolving_either():
    client = RecordingTrainingClient()
    session = make_session(client)
    datums = [object(), object()]

    fwd_result, opt_result = asyncio.run(
        session.submit_update(datums, loss_fn="ppo", lr=3e-5, grad_clip_norm=1.0)
    )

    # Forward/backward is queued first, then the optimizer step; only then
    # is either result awaited (RecordingFuture asserts that as well).
    assert client.events[:2] == ["submit_forward", "submit_optim"]
    assert sorted(client.events[2:]) == ["resolve_forward", "resolve_optim"]
    assert client.calls["forward"] == (datums, "ppo")  # passed through unchanged
    assert client.calls["optim"].learning_rate == 3e-5
    assert client.calls["optim"].grad_clip_norm == 1.0
    assert fwd_result is client.fwd_result
    assert opt_result is client.opt_result


@pytest.mark.parametrize("failure_stage", ["forward_submit", "optim_result"])
def test_submit_update_lets_sdk_errors_propagate(failure_stage):
    """One failure at submission, one at resolution; the hidden suite covers
    all four stages."""
    session = make_session(RecordingTrainingClient(failure_stage))

    # SdkFailure is not a RuntimeError, so an unimplemented stub surfaces as
    # NotImplementedError here instead of being mistaken for the SDK error.
    with pytest.raises(SdkFailure, match=failure_stage):
        asyncio.run(
            session.submit_update([object()], loss_fn="ppo", lr=1e-5, grad_clip_norm=1.0)
        )


# ------------------------------------------------------ training loops --
#
# The trainers are driven with their student methods replaced by tiny fakes,
# so each test isolates the one convention of `train()` it pins. The fake
# session records the snapshot names it is asked for; the fake logger keeps
# every (step, metrics) record.


class FakeSession:
    tokenizer = None

    def __init__(self):
        self.snapshot_names = []

    async def new_sampling_client(self, name):
        self.snapshot_names.append(name)
        return name


class FakeLogger:
    def __init__(self):
        self.records = []

    def log(self, step, metrics):
        self.records.append((step, metrics))


def make_sequence() -> TokenSequence:
    return TokenSequence(tokens=[1, 2], mask=[1], logprobs=[-0.5], advantages=[1.0])


def make_trajectory(reward: float) -> Trajectory:
    return Trajectory(
        task=Task(goal="g", id="t", max_steps=4),
        calls=[LLMCall([1, 2], [3, 4], [-0.1, -0.2])],
        reward=reward,
        num_steps=1,
    )


def make_sync_trainer(**config_overrides):
    config_values = {"group_size": 2, "eval_every": 0}
    config_values.update(config_overrides)
    session = FakeSession()
    logger = FakeLogger()
    trainer = SyncTrainer(
        session=session,
        env_factory=lambda task: None,
        train_tasks=[],
        val_tasks=[],
        model_cfg=ModelConfig(),
        train_cfg=TrainConfig(**config_values),
        logger=logger,
    )
    trainer.make_policy = lambda sampling_client: sampling_client
    return trainer, session, logger


def make_async_trainer(**config_overrides):
    config_values = {
        "num_workers": 1,
        "groups_per_batch": 1,
        "group_size": 2,
        "queue_capacity": 2,
        "num_steps": 1,
        "eval_every": 0,
    }
    config_values.update(config_overrides)
    session = FakeSession()
    logger = FakeLogger()
    trainer = AsyncGRPOTrainer(
        session=session,
        env_factory=lambda task: None,
        train_tasks=[Task(goal="a", id="a", max_steps=1)],
        val_tasks=[],
        model_cfg=ModelConfig(),
        train_cfg=TrainConfig(**config_values),
        logger=logger,
    )
    trainer.make_policy = lambda sampling_client: sampling_client
    return trainer, session, logger


def train_records(logger: FakeLogger) -> list[tuple[int, dict]]:
    return [
        (step, metrics)
        for step, metrics in logger.records
        if any(key.startswith("train/") for key in metrics)
    ]


def test_sync_snapshot_names_are_step_000000_to_num_steps():
    trainer, session, _ = make_sync_trainer(num_steps=2)

    async def collect_batch(policy):
        return []

    def build_batch(groups):
        return [make_sequence()], {}

    async def update(seqs):
        return {}

    trainer.collect_batch = collect_batch
    trainer.build_batch = build_batch
    trainer.update = update

    asyncio.run(trainer.train())
    # step_000000 is the initial policy; step_N follows update N (six digits).
    assert session.snapshot_names == ["step_000000", "step_000001", "step_000002"]


def test_sync_empty_batch_is_a_skipped_step_with_a_record_but_no_update_or_snapshot():
    trainer, session, logger = make_sync_trainer(num_steps=2)
    batches = iter([([], {"num_groups": 1}), ([make_sequence()], {"num_groups": 1})])

    async def collect_batch(policy):
        return [[make_trajectory(1.0), make_trajectory(1.0)]]

    def build_batch(groups):
        return next(batches)

    updates = 0

    async def update(seqs):
        nonlocal updates
        updates += 1
        return {"loss": 0.1}

    trainer.collect_batch = collect_batch
    trainer.build_batch = build_batch
    trainer.update = update

    asyncio.run(trainer.train())

    assert updates == 1
    assert session.snapshot_names == ["step_000000", "step_000002"]
    # record_training (given) logs the skipped step too, marked train/skipped.
    records = train_records(logger)
    assert [step for step, _ in records] == [1, 2]
    assert records[0][1]["train/skipped"] == 1
    assert "train/skipped" not in records[1][1]
    assert records[1][1]["train/loss"] == 0.1


# The tests below call the real collect_batch / evaluate_now with
# collect_group / evaluate_policy replaced by fakes (sync.py looks both up as
# module globals). They pin contracts whose violations do not raise on a
# real run.


def make_tasks(n, prefix="t"):
    return [Task(goal=f"g{i}", id=f"{prefix}{i}", max_steps=1) for i in range(n)]


def make_sync_trainer_with_tasks(num_tasks=8, **config_overrides):
    trainer, _, _ = make_sync_trainer(**config_overrides)
    trainer.train_tasks = make_tasks(num_tasks, prefix="train")
    trainer.val_tasks = make_tasks(num_tasks, prefix="val")
    return trainer


@pytest.fixture
def collect_group_calls(monkeypatch):
    """Replace collect_group with a fake that records its arguments."""
    recorded = []

    async def fake_collect_group(policy, task, env_factory, group_size, agent_cls=None):
        recorded.append({"task": task, "group_size": group_size})
        await asyncio.sleep(0)
        return [make_trajectory(1.0) for _ in range(group_size)]

    monkeypatch.setattr(sync_module, "collect_group", fake_collect_group)
    return recorded


def test_sync_collect_batch_tasks_are_distinct(collect_group_calls):
    # With tasks_per_batch equal to the number of tasks, any repeated task
    # means sampling with replacement; over several seeds that produces a
    # repeat essentially always.
    for seed in range(5):
        collect_group_calls.clear()
        trainer = make_sync_trainer_with_tasks(num_tasks=6, tasks_per_batch=6, seed=seed)
        asyncio.run(trainer.collect_batch("policy"))
        ids = [call["task"].id for call in collect_group_calls]
        assert len(ids) == 6 and len(set(ids)) == 6, f"seed {seed}: {ids}"


def test_sync_collect_batch_is_reproducible_with_group_size(collect_group_calls):
    # Same seed -> same tasks (draws come from self.rng), and every group
    # has cfg.group_size rollouts.
    def sampled(seed):
        collect_group_calls.clear()
        trainer = make_sync_trainer_with_tasks(
            num_tasks=20, tasks_per_batch=4, group_size=3, seed=seed
        )
        asyncio.run(trainer.collect_batch("policy"))
        assert [call["group_size"] for call in collect_group_calls] == [3] * 4
        return sorted(call["task"].id for call in collect_group_calls)

    assert sampled(1) == sampled(1)


def test_sync_collect_batch_runs_groups_concurrently(monkeypatch):
    # Every fake group waits until all groups have started, so collecting
    # one task after another never gets past the first group and times out.
    tasks_per_batch = 4
    started = 0
    all_started = None

    async def fake_collect_group(policy, task, env_factory, group_size, agent_cls=None):
        nonlocal started
        started += 1
        if started == tasks_per_batch:
            all_started.set()
        await all_started.wait()
        return [make_trajectory(1.0) for _ in range(group_size)]

    monkeypatch.setattr(sync_module, "collect_group", fake_collect_group)
    trainer = make_sync_trainer_with_tasks(tasks_per_batch=tasks_per_batch)

    async def run():
        nonlocal all_started
        all_started = asyncio.Event()
        return await asyncio.wait_for(trainer.collect_batch("policy"), timeout=2.0)

    try:
        groups = asyncio.run(run())
    except (asyncio.TimeoutError, TimeoutError):
        pytest.fail(
            f"only {started}/{tasks_per_batch} groups started before the first "
            "finished: collect_batch is not running groups concurrently"
        )
    assert len(groups) == tasks_per_batch


def test_sync_evaluate_now_uses_first_eval_tasks_val_tasks_and_eval_k(monkeypatch):
    recorded = {}

    async def fake_evaluate_policy(policy, tasks, env_factory, k=1, **kwargs):
        recorded.update(policy=policy, tasks=list(tasks), k=k)
        return {"success_rate": 0.5}

    monkeypatch.setattr(sync_module, "evaluate_policy", fake_evaluate_policy)
    trainer = make_sync_trainer_with_tasks(num_tasks=8, eval_tasks=3, eval_k=5, group_size=2)
    metrics = asyncio.run(trainer.evaluate_now("the-policy"))

    assert recorded["policy"] == "the-policy"
    assert [task.id for task in recorded["tasks"]] == ["val0", "val1", "val2"]
    assert recorded["k"] == 5
    assert metrics == {"success_rate": 0.5}


def test_async_groups_consumed_excludes_stale_dequeues():
    calls = 0

    async def generate_group(task_id, policy, version):
        nonlocal calls
        calls += 1
        # The first fake result is stale at version 0; the second is fresh.
        generated_at = version - 1 if calls == 1 else version
        return GroupResult(
            task_id=task_id,
            policy_version=generated_at,
            sequences=[make_sequence()],
            rewards=[1.0, 0.0],
        )

    async def update(batch):
        return {}

    trainer, session, _ = make_async_trainer(max_staleness=0)
    trainer.generate_group = generate_group
    trainer.update = update
    stats = asyncio.run(trainer.train())
    assert stats == {
        "train_steps": 1,
        "groups_consumed": 1,
        "groups_dropped_stale": 1,
    }
    assert session.snapshot_names == ["v0", "v1"]


def test_async_batch_without_sequences_is_a_skipped_step_without_update_or_publish():
    async def generate_group(task_id, policy, version):
        # Every group is degenerate: rewards are kept, sequences are empty.
        return GroupResult(task_id, version, [], [1.0, 1.0])

    async def update(batch):
        raise AssertionError("update must not run on a batch with no sequences")

    trainer, session, logger = make_async_trainer(num_steps=2)
    trainer.generate_group = generate_group
    trainer.update = update
    stats = asyncio.run(trainer.train())

    assert session.snapshot_names == ["v0"]  # nothing published after v0
    assert trainer.version == 0
    assert stats["train_steps"] == 2
    records = train_records(logger)
    assert [step for step, _ in records] == [1, 2]  # keyed by iteration, not version
    for _, metrics in records:
        assert metrics["train/skipped"] == 1
        assert metrics["train/policy_version"] == 0


def test_async_evaluation_records_the_captured_version_without_blocking_training():
    second_update_finished = asyncio.Event()
    num_updates = 0

    async def generate_group(task_id, policy, version):
        return GroupResult(
            task_id=task_id,
            policy_version=version,
            sequences=[make_sequence()],
            rewards=[1.0, 0.0],
        )

    async def update(batch):
        nonlocal num_updates
        num_updates += 1
        if num_updates == 2:
            second_update_finished.set()
        return {}

    async def evaluate_now(policy):
        # If step-1 evaluation blocks the learner, step 2 can never release it.
        if policy == "v1":
            await second_update_finished.wait()
        return {"policy": policy}

    trainer, _, logger = make_async_trainer(num_steps=2, eval_every=1)
    trainer.generate_group = generate_group
    trainer.update = update
    trainer.evaluate_now = evaluate_now

    async def run():
        return await asyncio.wait_for(trainer.train(), timeout=1.0)

    asyncio.run(run())
    # Each evaluation is recorded at the step that scheduled it, with the
    # policy and version captured then, not whatever is current when it ends.
    evaluations = sorted(
        (step, metrics["eval/policy"], metrics["eval/policy_version"])
        for step, metrics in logger.records
        if "eval/policy" in metrics
    )
    assert evaluations == [(1, "v1", 1), (2, "v2", 2)]


class WorkerFailure(Exception):
    """A rollout worker's own error; must reach train()'s caller unchanged."""


def test_async_worker_failure_propagates_instead_of_deadlocking_the_consumer():
    trainer, _, _ = make_async_trainer()

    async def failing_worker(results, task_iter):
        await asyncio.sleep(0)
        raise WorkerFailure("rollout worker failed")

    trainer.rollout_worker = failing_worker

    async def run():
        # WorkerFailure is not a NotImplementedError, so an unimplemented
        # train() fails here for its own reason, not by accident.
        with pytest.raises(WorkerFailure, match="rollout worker failed"):
            await asyncio.wait_for(trainer.train(), timeout=1.0)

    asyncio.run(run())
