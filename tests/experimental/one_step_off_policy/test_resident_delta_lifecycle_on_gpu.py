# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
from pathlib import Path

import pytest
import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, Qwen3Config

from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.engine_workers import ActorRolloutRefWorker

PHYSICAL_WORLD_SIZE = 4
PROBE_KEY = "ownership-changing-parameter-slice"
PROBE_NUMEL = 4096
RESIZE_ROUND_TRIPS = 10
MEMORY_GROWTH_TOLERANCE_BYTES = 4 * 1024 * 1024
STABLE_IDENTITY_FIELDS = (
    "actor_id",
    "pid",
    "physical_rank",
    "probe_key",
    "object_id",
    "data_ptr",
    "shape",
    "dtype",
    "device",
    "numel",
    "nbytes",
    "owns_storage",
    "content_sha256",
    "expected_sha256",
)


@ray.remote
class _NoCheckpointActorRolloutRefWorker(ActorRolloutRefWorker):
    """Fail any checkpoint handoff while allowing immutable HF safetensor reloads."""

    def __init__(self, *args, **kwargs):
        self._forbidden_checkpoint_io_attempts: list[dict[str, str]] = []

        def guarded_torch_load(source, *load_args, **load_kwargs):
            self._forbidden_checkpoint_io_attempts.append({"operation": "torch.load", "path": repr(source)})
            raise AssertionError(f"resident delta lifecycle must not call torch.load: {source!r}")

        def guarded_torch_save(obj, destination, *save_args, **save_kwargs):
            self._forbidden_checkpoint_io_attempts.append({"operation": "torch.save", "path": repr(destination)})
            raise AssertionError(f"resident delta lifecycle must not call torch.save: {destination!r}")

        torch.load = guarded_torch_load
        torch.save = guarded_torch_save
        super().__init__(*args, **kwargs)

    def _forbid_checkpoint_operation(self, operation: str, args) -> None:
        self._forbidden_checkpoint_io_attempts.append({"operation": operation, "path": repr(args[:1])})
        raise AssertionError(f"resident delta lifecycle must not call {operation}")

    def _install_inner_checkpoint_guards(self) -> None:
        if self.actor is None:
            return

        def guard(operation):
            return lambda *args, **kwargs: self._forbid_checkpoint_operation(operation, args)

        for method_name in ("save_checkpoint", "load_checkpoint", "stage_optimizer_for_dynamic_resize"):
            setattr(self.actor, method_name, guard(f"TrainingWorker.{method_name}"))
        checkpoint_manager = getattr(self.actor.engine, "checkpoint_manager", None)
        if checkpoint_manager is not None:
            for method_name in ("save_checkpoint", "load_checkpoint"):
                if hasattr(checkpoint_manager, method_name):
                    setattr(checkpoint_manager, method_name, guard(f"FSDPCheckpointManager.{method_name}"))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self._install_inner_checkpoint_guards()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reinitialize_actor_runtime(self):
        super().reinitialize_actor_runtime()
        self._install_inner_checkpoint_guards()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, *args, **kwargs):
        self._forbid_checkpoint_operation("ActorRolloutRefWorker.save_checkpoint", args)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, *args, **kwargs):
        self._forbid_checkpoint_operation("ActorRolloutRefWorker.load_checkpoint", args)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stage_optimizer_for_dynamic_resize(self, *args, **kwargs):
        self._forbid_checkpoint_operation("ActorRolloutRefWorker.stage_optimizer_for_dynamic_resize", args)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def forbidden_checkpoint_io_attempts(self):
        return list(self._forbidden_checkpoint_io_attempts)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def probe_actor_runtime_state(self):
        engine = self.actor.engine
        return {
            "physical_rank": self.rank,
            "has_module": getattr(engine, "module", None) is not None,
            "has_optimizer": getattr(engine, "optimizer", None) is not None,
            "has_process_group": getattr(engine, "process_group", None) is not None,
        }


def _create_tiny_model(root: Path) -> str:
    model_path = root / "tiny-qwen3"
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=True,
        use_cache=False,
    )
    torch.manual_seed(20260829)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(model_path, safe_serialization=True)
    config.save_pretrained(model_path)
    return str(model_path)


def _create_worker_config(model_path: str):
    return OmegaConf.create(
        {
            "actor": {
                "_target_": "verl.workers.config.FSDPActorConfig",
                "strategy": "fsdp2",
                "rollout_n": 1,
                "ppo_mini_batch_size": PHYSICAL_WORLD_SIZE,
                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_max_token_len_per_gpu": 128,
                "use_dynamic_bsz": False,
                "use_torch_compile": False,
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "strategy": "fsdp2",
                    "fsdp_size": -1,
                    "model_dtype": "fp32",
                    "param_offload": False,
                    "optimizer_offload": False,
                    "use_torch_compile": False,
                    "ulysses_sequence_parallel_size": 1,
                },
                "optim": {
                    "_target_": "verl.workers.config.FSDPOptimizerConfig",
                    "lr": 1e-6,
                },
                "checkpoint": {
                    "_target_": "verl.trainer.config.CheckpointConfig",
                },
            },
            "rollout": {
                "name": "vllm",
                "log_prob_use_dynamic_bsz": False,
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 128,
                "checkpoint_engine": {
                    "_target_": "verl.workers.config.CheckpointEngineConfig",
                    "backend": "naive",
                    "update_weights_bucket_megabytes": 1,
                    "engine_kwargs": {},
                },
            },
            "model": {
                "_target_": "verl.workers.config.HFModelConfig",
                "path": model_path,
                "load_tokenizer": False,
                "enable_gradient_checkpointing": False,
                "use_remove_padding": False,
            },
        }
    )


def _snapshots_by_rank(snapshots: list[dict]) -> dict[int, dict]:
    by_rank = {snapshot["physical_rank"]: snapshot for snapshot in snapshots}
    assert set(by_rank) == set(range(PHYSICAL_WORLD_SIZE))
    return by_rank


def _assert_stable_snapshots(baseline_by_rank: dict[int, dict], snapshots: list[dict]) -> None:
    current_by_rank = _snapshots_by_rank(snapshots)
    for rank, baseline in baseline_by_rank.items():
        current = current_by_rank[rank]
        assert current["exact_match"], current
        assert current["content_sha256"] == current["expected_sha256"], current
        for field in STABLE_IDENTITY_FIELDS:
            assert current[field] == baseline[field], (
                f"rank {rank} changed held-delta field {field}: before={baseline[field]!r}, after={current[field]!r}"
            )


def _worker_transition(physical_rank: int, old_ranks: list[int], new_ranks: list[int]) -> str:
    was_train = physical_rank in old_ranks
    is_train = physical_rank in new_ranks
    if was_train and is_train:
        return "persistent_train"
    if was_train:
        return "train_to_non_train"
    if is_train:
        return "non_train_to_train"
    return "persistent_non_train"


def _assert_expected_transition_roles(old_ranks: list[int], new_ranks: list[int]) -> None:
    expected_by_topology = {
        ((0, 1, 2, 3), (0, 1)): {
            0: "persistent_train",
            1: "persistent_train",
            2: "train_to_non_train",
            3: "train_to_non_train",
        },
        ((0, 1), (0, 1, 2)): {
            0: "persistent_train",
            1: "persistent_train",
            2: "non_train_to_train",
            3: "persistent_non_train",
        },
        ((0, 1, 2), (0, 1)): {
            0: "persistent_train",
            1: "persistent_train",
            2: "train_to_non_train",
            3: "persistent_non_train",
        },
    }
    topology = (tuple(old_ranks), tuple(new_ranks))
    assert topology in expected_by_topology, f"unexpected lifecycle topology: {topology}"
    actual = {rank: _worker_transition(rank, old_ranks, new_ranks) for rank in range(PHYSICAL_WORLD_SIZE)}
    assert actual == expected_by_topology[topology]


def _log_snapshots(
    *,
    cycle: int,
    transition: str,
    stage: str,
    old_ranks: list[int],
    new_ranks: list[int],
    snapshots: list[dict],
) -> None:
    for snapshot in snapshots:
        event = {
            "event": "resident_delta_snapshot",
            "cycle": cycle,
            "transition": transition,
            "stage": stage,
            "old_active_ranks": old_ranks,
            "new_active_ranks": new_ranks,
            "worker_transition": _worker_transition(snapshot["physical_rank"], old_ranks, new_ranks),
            **snapshot,
        }
        print(f"RESIDENT_DELTA_LIFECYCLE {json.dumps(event, sort_keys=True)}", flush=True)


def _run_transition(
    *,
    physical_wg: RayWorkerGroup,
    baseline_by_rank: dict[int, dict],
    cycle: int,
    transition: str,
    old_ranks: list[int],
    new_ranks: list[int],
) -> tuple[RayWorkerGroup, list[dict]]:
    _assert_expected_transition_roles(old_ranks, new_ranks)
    timings: dict[str, float] = {}
    transition_start = time.perf_counter()

    before = physical_wg.probe_inspect_held_delta_tensor(PROBE_KEY)
    _assert_stable_snapshots(baseline_by_rank, before)
    _log_snapshots(
        cycle=cycle,
        transition=transition,
        stage="old_runtime_alive",
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        snapshots=before,
    )

    stage_start = time.perf_counter()
    physical_wg.shutdown_actor_runtime()
    timings["shutdown_runtime_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    physical_wg.destroy_actor_process_group()
    timings["destroy_process_group_s"] = time.perf_counter() - stage_start
    after_destroy = physical_wg.probe_inspect_held_delta_tensor(PROBE_KEY)
    _assert_stable_snapshots(baseline_by_rank, after_destroy)
    for state in physical_wg.probe_actor_runtime_state():
        assert not state["has_module"], state
        assert not state["has_optimizer"], state
        assert not state["has_process_group"], state
    _log_snapshots(
        cycle=cycle,
        transition=transition,
        stage="old_process_group_destroyed",
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        snapshots=after_destroy,
    )

    stage_start = time.perf_counter()
    membership = physical_wg.prepare_actor_process_group(new_ranks)
    timings["prepare_process_group_s"] = time.perf_counter() - stage_start
    assert membership == [rank in new_ranks for rank in range(PHYSICAL_WORLD_SIZE)]
    after_prepare = physical_wg.probe_inspect_held_delta_tensor(PROBE_KEY)
    _assert_stable_snapshots(baseline_by_rank, after_prepare)
    _log_snapshots(
        cycle=cycle,
        transition=transition,
        stage="new_process_group_prepared",
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        snapshots=after_prepare,
    )

    active_wg = physical_wg.subgroup(new_ranks, name_prefix=f"resident_delta_{cycle}_{transition}")
    assert active_wg.world_size == len(new_ranks)
    for subgroup_index, physical_rank in enumerate(new_ranks):
        assert active_wg.workers[subgroup_index] is physical_wg.workers[physical_rank]

    stage_start = time.perf_counter()
    active_wg.reinitialize_actor_runtime()
    timings["reinitialize_runtime_s"] = time.perf_counter() - stage_start
    after_reinitialize = physical_wg.probe_inspect_held_delta_tensor(PROBE_KEY)
    _assert_stable_snapshots(baseline_by_rank, after_reinitialize)
    for state in physical_wg.probe_actor_runtime_state():
        expected_active = state["physical_rank"] in new_ranks
        assert state["has_module"] is expected_active, state
        assert state["has_optimizer"] is expected_active, state
        assert state["has_process_group"] is expected_active, state
    _log_snapshots(
        cycle=cycle,
        transition=transition,
        stage="new_runtime_alive",
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        snapshots=after_reinitialize,
    )

    process_group_states = physical_wg.probe_check_actor_process_group(run_collective=False)
    for state in process_group_states:
        physical_rank = state["physical_rank"]
        expected_member = physical_rank in new_ranks
        assert state["is_member"] is expected_member
        if expected_member:
            assert state["group_rank"] == new_ranks.index(physical_rank)
            assert state["group_world_size"] == len(new_ranks)
        else:
            assert state["group_rank"] is None
            assert state["group_world_size"] is None

    collective = active_wg.probe_check_actor_process_group()
    for group_rank, result in enumerate(collective):
        assert result["physical_rank"] == new_ranks[group_rank]
        assert result["group_rank"] == group_rank
        assert result["group_world_size"] == len(new_ranks)
        assert result["all_reduce_exact"], result

    timings["total_s"] = time.perf_counter() - transition_start
    summary = {
        "event": "resident_delta_transition",
        "cycle": cycle,
        "transition": transition,
        "old_active_ranks": old_ranks,
        "new_active_ranks": new_ranks,
        "collective": collective,
        "timings_s": timings,
    }
    print(f"RESIDENT_DELTA_LIFECYCLE {json.dumps(summary, sort_keys=True)}", flush=True)
    return active_wg, after_destroy


def _assert_equivalent_stage_memory_stable(samples_by_rank: dict[int, list[int]], metric: str) -> None:
    for rank, samples in samples_by_rank.items():
        assert len(samples) == RESIZE_ROUND_TRIPS
        assert samples[-1] <= samples[0] + MEMORY_GROWTH_TOLERANCE_BYTES, (
            f"rank {rank} {metric} grew from {samples[0]} to {samples[-1]} bytes"
        )
        assert max(samples) - min(samples) <= MEMORY_GROWTH_TOLERANCE_BYTES, (
            f"rank {rank} {metric} varied beyond tolerance: {samples}"
        )
        assert not all(after > before for before, after in zip(samples, samples[1:], strict=False)), (
            f"rank {rank} {metric} increased on every equivalent lifecycle sample: {samples}"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < PHYSICAL_WORLD_SIZE,
    reason="requires at least four CUDA devices",
)
def test_resident_delta_tensor_survives_ten_resize_round_trips_without_checkpoint_handoff(tmp_path):
    model_path = _create_tiny_model(tmp_path)
    assert not list(tmp_path.rglob("*.pt"))
    initial_test_files = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())

    test_module_dir = str(Path(__file__).resolve().parent)
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    worker_pythonpath = os.pathsep.join(path for path in (test_module_dir, inherited_pythonpath) if path)
    ray.init(
        num_cpus=8,
        num_gpus=PHYSICAL_WORLD_SIZE,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": worker_pythonpath}},
    )
    physical_wg = None
    resource_pool = None
    try:
        worker_cls = RayClassWithInitArgs(
            cls=_NoCheckpointActorRolloutRefWorker,
            config=_create_worker_config(model_path),
            role="actor",
        )
        resource_pool = RayResourcePool(
            process_on_nodes=[PHYSICAL_WORLD_SIZE],
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="resident_delta_lifecycle",
        )
        physical_wg = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_cls,
            name_prefix="resident_delta_physical",
        )
        physical_wg.init_model()

        baseline = physical_wg.probe_install_held_delta_tensor(PROBE_KEY, PROBE_NUMEL)
        baseline_by_rank = _snapshots_by_rank(baseline)
        _assert_stable_snapshots(baseline_by_rank, baseline)

        active_wg, _ = _run_transition(
            physical_wg=physical_wg,
            baseline_by_rank=baseline_by_rank,
            cycle=0,
            transition="initial_4_to_2",
            old_ranks=[0, 1, 2, 3],
            new_ranks=[0, 1],
        )
        assert active_wg.world_size == 2

        shrink_allocated_samples = {rank: [] for rank in range(PHYSICAL_WORLD_SIZE)}
        shrink_reserved_samples = {rank: [] for rank in range(PHYSICAL_WORLD_SIZE)}
        for cycle in range(1, RESIZE_ROUND_TRIPS + 1):
            active_wg, _ = _run_transition(
                physical_wg=physical_wg,
                baseline_by_rank=baseline_by_rank,
                cycle=cycle,
                transition="expand_2_to_3",
                old_ranks=[0, 1],
                new_ranks=[0, 1, 2],
            )
            assert active_wg.world_size == 3

            active_wg, after_destroy = _run_transition(
                physical_wg=physical_wg,
                baseline_by_rank=baseline_by_rank,
                cycle=cycle,
                transition="shrink_3_to_2",
                old_ranks=[0, 1, 2],
                new_ranks=[0, 1],
            )
            assert active_wg.world_size == 2
            for snapshot in after_destroy:
                allocated = snapshot["device_memory_allocated_bytes"]
                reserved = snapshot["device_memory_reserved_bytes"]
                assert isinstance(allocated, int) and isinstance(reserved, int)
                rank = snapshot["physical_rank"]
                shrink_allocated_samples[rank].append(allocated)
                shrink_reserved_samples[rank].append(reserved)

        _assert_equivalent_stage_memory_stable(shrink_allocated_samples, "memory_allocated")
        _assert_equivalent_stage_memory_stable(shrink_reserved_samples, "memory_reserved")
        forbidden_io_attempts = physical_wg.forbidden_checkpoint_io_attempts()
        assert forbidden_io_attempts == [[] for _ in range(PHYSICAL_WORLD_SIZE)]
        pre_release_summary = {
            "event": "resident_delta_verification_summary",
            "resize_round_trips": RESIZE_ROUND_TRIPS,
            "lifecycle_transition_count": 1 + 2 * RESIZE_ROUND_TRIPS,
            "snapshot_count": PHYSICAL_WORLD_SIZE * 4 * (1 + 2 * RESIZE_ROUND_TRIPS),
            "forbidden_checkpoint_io_attempts": forbidden_io_attempts,
            "equivalent_stage_memory_allocated_bytes": shrink_allocated_samples,
            "equivalent_stage_memory_reserved_bytes": shrink_reserved_samples,
        }
        print(f"RESIDENT_DELTA_LIFECYCLE {json.dumps(pre_release_summary, sort_keys=True)}", flush=True)

        released = physical_wg.probe_release_held_delta_tensor(PROBE_KEY)
        release_evidence = []
        for release_snapshot in released:
            rank = release_snapshot["physical_rank"]
            for field in STABLE_IDENTITY_FIELDS:
                assert release_snapshot[field] == baseline_by_rank[rank][field]
            assert release_snapshot["released"]
            assert PROBE_KEY not in release_snapshot["remaining_probe_keys"]
            before = release_snapshot["device_memory_allocated_bytes"]
            after = release_snapshot["memory_after_release"]["device_memory_allocated_bytes"]
            assert isinstance(before, int) and isinstance(after, int)
            assert after <= before
            release_evidence.append(
                {
                    "physical_rank": rank,
                    "released_nbytes": release_snapshot["nbytes"],
                    "allocated_before_release": before,
                    "allocated_after_release": after,
                }
            )
        print(
            "RESIDENT_DELTA_LIFECYCLE "
            + json.dumps({"event": "resident_delta_release_summary", "workers": release_evidence}, sort_keys=True),
            flush=True,
        )
    finally:
        if physical_wg is not None:
            for worker in physical_wg.workers:
                ray.kill(worker, no_restart=True)
        if resource_pool is not None and resource_pool.pgs is not None:
            for placement_group in resource_pool.pgs:
                ray.util.remove_placement_group(placement_group)
        ray.shutdown()

    pt_files = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*.pt"))
    assert pt_files == []
    final_test_files = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    assert final_test_files == initial_test_files
    print(
        "RESIDENT_DELTA_LIFECYCLE "
        + json.dumps({"event": "resident_delta_test_complete", "pt_files": pt_files}, sort_keys=True),
        flush=True,
    )
