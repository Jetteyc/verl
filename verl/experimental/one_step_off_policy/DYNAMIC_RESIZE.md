# Dynamic resize in the hybrid one-step-off trainer

This document describes how `verl.experimental.one_step_off_policy` moves GPUs between rollout and
training while a run is in progress ("dynamic resize"): what a resize is, how to run the single-node
example, what happens during a hard switch, and how the result is verified. It is the reader's guide to
the code; the commit messages carry the per-change rationale.

Related files:

* `README.md` -- the one-step-off recipe itself (pipeline, parameter sync, configuration).
* `DYNAMIC_RESIZE_DIRECT_RESHARD_PLAN.md` -- the original design plan; its section 0 tracks which plan
  items are implemented on this branch. Sections 1-18 of that file predate the implementation.
* `config/one_step_off_ppo_trainer_dynamic_resize.yaml` -- the configuration the example uses.
* `shell/run_dynamic_resize.sh` -- the single-node example launcher.

## What a resize is

A one-step-off run has two kinds of GPU work: **rollout** (vLLM generating responses) and **training**
(the actor, plus critic/reference if configured). Their ideal GPU shares change over a run, so the trainer
can re-partition a **shared pool** of GPUs between them at step boundaries.

* `hybrid_engine=true`: one `ActorRolloutRefWorker` process per GPU covers the whole pool. Each process
  holds a training engine, a rollout adapter and a weight-sync (checkpoint-engine) client. Processes are
  never killed by a resize; only their *roles* change.
* Layout: training ranks are always the prefix `range(actor_size)` of the pool, rollout ranks the suffix.
  `trainer.dynamic_resize.shared_pool.n_gpus_per_node` is the pool size; every schedule stage must satisfy
  `actor_size + rollout_size == pool size`.
* Rollout is organised in **replicas**: one vLLM engine per `tensor_model_parallel_size` consecutive GPUs
  (`rollout.yaml` defaults to TP=2, so a replica is a GPU pair). `replica_rank = physical_rank // TP`;
  the server actor is named `server_{replica_rank}_{node_rank}`. The split point must be a multiple of the
  replica size (`_validate_hybrid_split`), so a replica is either wholly rollout or wholly training.
* `trainer.dynamic_resize.mode=schedule`: `schedule.stageN` entries (`step`, `actor_pool.size`,
  `rollout_pool.size`, ...) say which topology applies from which step; `stage0` with `step: -1` is the
  initial split. `hard_switch.enable=true` makes the trainer actually move resources; otherwise the switch
  is only recorded as deferred.
* The **hard switch** runs at the end of a training step, after `update_actor`, with no rollout in flight
  (the step that resizes does not pre-launch the next generation). Training state that has to survive it:
  fp32 parameters, optimizer state, LR-scheduler state and module buffers, all FSDP2-sharded along dim 0,
  so a change of world size changes every shard boundary.

## Running the single-node example

```bash
bash verl/experimental/one_step_off_policy/shell/run_dynamic_resize.sh
```

The script converts a GSM8K parquet pair into the verl format (under `$CKPT_DIR/data`), turns the
environment variables below into hydra overrides and starts
`python -m verl.experimental.one_step_off_policy.main_ppo --config-name=one_step_off_ppo_trainer_dynamic_resize.yaml`.
Any extra command-line arguments are forwarded to hydra unchanged (`"$@"`), so arbitrary overrides can be
appended, e.g. `bash .../run_dynamic_resize.sh trainer.total_training_steps=3`.

| Variable | Default | Meaning |
|---|---|---|
| `NUM_GPUS` / `SHARED_POOL_GPUS` | 8 / `NUM_GPUS` | pool size |
| `N_GPUS_ROLLOUT` | 4 | initial rollout GPUs; training gets `NUM_GPUS - N_GPUS_ROLLOUT` |
| `RESIZE_STEP` | 2 | step of the first switch; the second is `RESIZE_STEP + 2` |
| `FIRST_SPLIT_PLAN` / `SECOND_SPLIT_PLAN` | `[6,2]` / `[2,6]` | `[actor, rollout]` sizes of the two switches |
| `TOTAL_TRAINING_STEPS` | second switch + 2 | run length |
| `MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`, `CKPT_DIR` | site paths | model, data and output locations |
| `TRAIN_BATCH_SIZE` / `PPO_MINI_BATCH_SIZE` | 960 / 240 | must divide evenly by every actor size in the schedule |
| `USE_KL_LOSS` | false | GRPO without a reference model by default |
| `HARD_SWITCH_ENABLE` | true | actually move resources |
| `HOST_STAGING_ENABLE` / `HOST_STAGING_BACKEND` | true / `gpu_direct` | `handoff.enable` / `handoff.backend` |
| `HYSTERESIS_*`, `BUDGET_PROTECTION_ENABLE`, `COMM_CACHE_ENABLE` | off | optional controller, memory gate, communicator cache |
| `PROFILER_ENABLE` | true | `global_profiler.enabled`; set `false` for quick iterations |
| `VERL_RESIZE_VERIFY` | unset | `1` checksums every tensor moved by `gpu_direct` against the source |

More switches are added as further stages with the same shape as `stage0` in the yaml:

```bash
S=trainer.dynamic_resize.schedule
TOTAL_TRAINING_STEPS=8 bash .../run_dynamic_resize.sh \
  +$S.stage3.step=6 +$S.stage3.actor_pool.mode=split +$S.stage3.actor_pool.from_pool=shared_pool \
  +$S.stage3.actor_pool.index=0 +$S.stage3.actor_pool.size=[6,2] \
  +$S.stage3.rollout_pool.mode=split +$S.stage3.rollout_pool.from_pool=shared_pool \
  +$S.stage3.rollout_pool.index=1 +$S.stage3.rollout_pool.size=[6,2] +$S.stage3.release_old=false
```

## The hard switch, step by step

`OneStepOffRayTrainer._apply_dynamic_resize_hard_switch` with `handoff.backend=gpu_direct`, for a switch
to `(actor_target, rollout_target)`. Nothing is written to or read from disk, and nothing is staged on the
host; the training state moves GPU to GPU.

1. **Rollout makes room** (`_destroy_dynamic_resize_runtime(actor_target)`): abort in-flight requests on
   every replica, kill the agent-loop workers, then `LLMServerManager.retire_hybrid_replicas_below(actor_target,
   dormant=handoff.dormant_rollout_replicas)`: the replicas whose GPUs become training ranks are put to
   sleep (vLLM sleep level 2: weights and KV cache freed, server actor, engine and TP communicators kept)
   and dropped from the load balancer; with `dormant=false` they are shut down instead. Replicas that stay
   on rollout GPUs are untouched.
2. **Move the training state** (`_rebuild_hybrid_training_runtime` -> `resize_actor_runtime` on every
   physical rank): build the new FSDP2 runtime on the new ranks next to the live one, fill it shard to
   shard with one `all_to_all_single` per tensor, swap, destroy the old groups. The critic goes through the
   same `resize_runtime` call right after the actor. Details under *Training state: shard-to-shard
   transfer*.
3. **Re-point the control plane**: `actor_wg` / `critic_wg` become `RayWorkerGroup.subgroup(new actor ranks)`
   views (no Ray actors are created or destroyed), the rollout view becomes the suffix ranks, and the
   hybrid workers drop their cached server handles (`reset_rollout_server_handle`).
4. **Re-plan rollout** (`LLMServerManager.rebuild_hybrid_runtime(start_rank=actor_target // TP)`):
   `plan_hybrid_replica_reuse` classifies every new GPU pair as *keep* (a live replica already sits
   there), *wake* (a dormant replica sits there) or *launch* (the pair has never been rollout); the load
   balancer is updated in place. Details under *Rollout side*.
5. **Weight sync**: `reconfigure_checkpoint_engine(group_name, rebuild_group=True)` finalizes the old NCCL
   weight-sync engine and creates a new one on every rank; `CheckpointEngineManager.update_weights`
   pushes the current actor weights to all replicas -- kept, woken and launched alike -- and resumes
   generation. A woken replica's weights are garbage after a level-2 sleep and are filled here exactly
   like a fresh replica's, so one-step-off staleness is unchanged.
6. The trainer records the new topology and launches the next step's generation on the new rollout ranks.

Startup does not transfer anything: `init_model(actor_ranks=initial_actor_ranks)` builds the training
runtimes directly on the initial actor subgroup (see *Startup* below).

## Training state: shard-to-shard transfer

### Layers

| Layer | File | Responsibility |
|---|---|---|
| trainer | `one_step_off_policy/ray_trainer.py` | decides the topology; `_resize_actor_runtime_gpu_direct` / `_resize_critic_runtime_gpu_direct` call the workers, check that exactly `actor_target` ranks report `member` with identical tensor counts, aggregate `resize/gpu_direct_*` metrics |
| worker | `verl/workers/engine_workers.py` | `TrainingWorker.resize_runtime(old_ranks, new_ranks)` runs on *every* physical rank (`ONE_TO_ALL`): group creation, prepare, verdicts, transfer, commit, teardown; `ActorRolloutRefWorker.resize_actor_runtime` wraps it for actor and co-located ref |
| engine | `verl/workers/engine/fsdp/transformer_impl.py` | `FSDPEngine.prepare_resize_target` / `transfer_resize_state` / `commit_resize` / `abort_resize`; `check_gpu_direct_resize_supported`; `_build_without_weights` branches of `_build_module` / `_build_fsdp_module` |
| transfer | `verl/utils/fsdp_resize.py` | `FSDPResizePlanner` (row intervals), `transfer_fsdp2_runtime` / `_Transfer` (manifest, `all_to_all_single`, broadcasts, optimizer/scheduler metadata, checksums), `raise_if_any_failed`, `fsdp2_units` / `materialize_unit` / `release_unit` (low-memory pipeline) |

### What moves

`_build_manifest` on `old_ranks[0]` lists everything; the manifest is broadcast and every rank checks its
old and new runtime against it (`_check_structure`) before any data moves.

| Item | How | Why it is needed |
|---|---|---|
| parameters with `Shard(0)` placement | one `all_to_all_single` per tensor, old local shard as send buffer, new local shard as receive buffer | the weights themselves; tied weights are moved once (`named_parameters()` de-duplicates) |
| replicated parameters / plain tensors | `broadcast` from `old_ranks[0]` | same |
| **all** `named_buffers()`, including non-persistent ones such as rotary `inv_freq` | `broadcast` | the new module is built on the meta device and `to_empty`'d, so buffers are uninitialised too; `state_dict()` would miss the non-persistent ones |
| optimizer state tensors (`exp_avg`, `exp_avg_sq`) | same planner and `all_to_all_single` as the parameter (same shape, same sharding); the new side first allocates `zeros_like(param)` via `_ensure_optimizer_state` | exact training continuation; without them Adam restarts from zero moments |
| optimizer scalars (`step`), `param_groups` hyper-parameters (minus `params`), `lr_scheduler.state_dict()` | metadata in the manifest (`broadcast_object_list`), applied on the new side | bias correction, current LR / weight decay, warm-up position |

Not moved, and not needed: gradients (the switch happens after `optimizer.step()` and `zero_grad`), FSDP2's
own bookkeeping (recreated by `fully_shard`), RNG state (training has no dropout; sampling lives in vLLM),
dataloader position (lives in the trainer process).

### Why one `all_to_all_single` per tensor is enough

FSDP2 shards every parameter along dim 0 with `torch.chunk` semantics (`Shard._local_shard_size_and_offset`),
so each rank's local shard is a contiguous row interval and the intervals are ordered by mesh rank. With
`old_ranks` and `new_ranks` both sorted:

* the rows an old rank must send to successive new ranks are consecutive sub-intervals of its own shard, in
  order -- `FSDPResizePlanner.send_row_splits` partitions the old shard;
* the rows a new rank receives from successive old ranks are consecutive sub-intervals of its new shard, in
  order -- `recv_row_splits` partitions the new shard.

`all_to_all_single(recv, send, output_split_sizes, input_split_sizes, group=old ∪ new)` therefore uses
`old_param.to_local().view(-1)` and `new_param.to_local().view(-1)` directly (`to_local()` returns FSDP2's
padded sharded storage minus the padding, which is the tensor's permanent home). Send and receive sizes
differ in general; the only constraints are `send.numel() == sum(input_split_sizes)` and
`recv.numel() == sum(output_split_sizes)`, both asserted. Ranks on one side only pass a zero-length buffer
and an all-zero split vector. Rows a rank keeps are NCCL's self chunk, an on-device copy
(`resize/gpu_direct_self_copy_bytes`); every other row crosses the wire exactly once.

### Worker protocol (`TrainingWorker.resize_runtime`)

Every physical rank executes the same sequence; non-members "accompany" the collectives that need WORLD.

1. `barrier`; `new_group(new_ranks)` and `new_group(old ∪ new)` in the same order on all ranks.
2. New ranks run the rank-local preconditions (`check_gpu_direct_resize_supported`, a shut-down or empty
   engine shell is fine); the result is exchanged with `raise_if_any_failed` over `old ∪ new` *before* the
   first collective on the new group. NCCL communicator creation has no watchdog, so a rank that raised
   there would leave the others hanging forever.
3. New ranks: `engine.prepare_resize_target(new_pg)` -- save the live runtime attributes, switch
   `process_group`, set `_build_without_weights=True`, `_init_device_mesh()` (`DeviceMesh.from_group`),
   `_build_model_optimizer()` (`from_config` on the meta device -> `fully_shard` -> `module.to_empty(device)`,
   optimizer and scheduler with empty state), park the result in `_resize_target`, restore the live runtime.
   Old and new runtimes now coexist in the process.
4. Second verdict exchange: every new rank's target exists, or everyone raises.
5. `old ∪ new` ranks: `engine.transfer_resize_state(old_ranks, new_ranks, group)` ->
   `transfer_fsdp2_runtime`: `reshard()` both modules, manifest, structure check, parameters, optimizer
   state, buffers, optional checksums, `synchronize` + `barrier`. Status per rank: `member` (in the new
   group), `sender` (old only), `non_member`.
6. Record `peak_alloc_bytes` (old and new shards coexist), `engine.commit_resize()` (shut the live runtime
   down, install the target, rebuild `FSDPCheckpointManager`), destroy the old training group and the
   transfer group, `clear_dtensor_dispatch_caches()` on every rank, refresh dispatch info, `barrier`.

Any exception -> `engine.abort_resize()` drops the half-built target and re-raises. There is no fallback
path: a failed switch fails the run rather than silently training on wrong state.

### Startup: the initial split

`ActorRolloutRefWorker.init_model(actor_ranks=...)` and `TrainingWorker.reset_on(ranks)` create the
initial actor subgroup first and build the actor, ref and critic runtimes only on its members
(`FSDPEngine.set_process_group` drops the stale mesh so `initialize()` builds it on the new group).
Ranks that never joined a training subgroup look like shut-down engines: `module`, `optimizer`,
`lr_scheduler` and `checkpoint_manager` are `None`. Without `actor_ranks` the behaviour is upstream's,
so non-resize runs are unaffected. Before this, the actor was built on the whole pool and moved to the
initial ranks with one extra transfer.

### Low-memory pipeline (`handoff.gpu_direct_low_memory=true`)

The default path holds old and new shards at the same time (peak ~ `12 B/param * (1/old + 1/new)` for
fp32 parameters plus two Adam moments). With the flag, `prepare_resize_target` leaves the parameters on
the meta device after `fully_shard` (only buffers are materialised) and `_Transfer` works one FSDP2
**unit** at a time -- a decoder layer or the root group (`fsdp2_units`): `materialize_unit` allocates the
unit's new shards (`swap_tensors` + `FSDPParam.reset_sharded_param`), its parameters and optimizer state
move, `release_unit` frees the old shards (`FSDPParam._sharded_param_data` and optimizer state). Units
are ordered smallest-first when expanding and largest-first when shrinking, so the peak is
`max(old, new) + one unit`. Communication is unchanged; checksums accumulate the old side per unit before
release. Off by default because it relies on FSDP2 internals; the default path keeps the old runtime
intact until the transfer is verified.

### Preconditions (`check_gpu_direct_resize_supported`, checked at startup)

* `actor.strategy=fsdp2` with `fsdp_size=-1` (one-dimensional mesh, `Shard(0)` only);
* parameters and optimizer state on GPU (no `param_offload` / `optimizer_offload` / FSDP2 `CPUOffloadPolicy`);
* no LoRA, `ulysses_sequence_parallel_size=1`, CUDA;
* the critic, if any, is an `AutoModelForTokenClassification` model (trl value-head critics are rejected);
  a separate reference worker group is rejected, while a reference model co-located in
  `ActorRolloutRefWorker` follows the actor through `resize_actor_runtime` as long as it keeps its
  parameters on GPU;
* every actor and rollout size in the schedule is a multiple of the replica size (`_validate_hybrid_split`).

## Rollout side: vLLM replicas across resizes

A replica is an independent vLLM engine whose TP communicators live inside its own GPUs, and in the
TP-aligned prefix layout a replica either stays wholly on rollout GPUs or wholly becomes training GPUs.
Its `replica_rank` (hence its server actor name and port) depends only on its physical position, and the
hybrid workers' `ServerAdapter` resolves the server by that name. So a replica that stays needs no rename
and no reconnect, and `rebuild_hybrid_runtime` is a diff rather than a rebuild:

* `LLMServerManager.retire_hybrid_replicas_below(physical_rank, dormant)` -- called before the training side allocates on those GPUs; removes the affected servers from the load balancer and either sleeps them (`_sleep_replicas`: abort -> `replica.sleep(force=True)` -> park in `_dormant_replicas`) or shuts them down (`_shutdown_replicas`).
* `plan_hybrid_replica_reuse(live, new_workers, start_rank, replica_world_size, dormant)` (pure, CPU-tested) -- for every new GPU pair: **keep** if a live replica with the same `replica_rank`, the same worker handles and live servers sits there; **wake** if a dormant replica with the same `replica_rank` and workers sits there; otherwise **launch**. Live replicas outside the new range are retired (normally already done in step 1); dormant ones outside the range stay asleep.
* `rebuild_hybrid_runtime` executes the plan: kept replicas only update `worker_group_rank`, woken ones run `wake_up()` (weights + KV cache re-allocated, prefix cache reset) concurrently with the `init_hybrid` of launched ones; the load balancer gets `add_servers`; the agent-loop manager is recreated. The old kill-everything path remains as `_rebuild_hybrid_runtime_from_scratch` for trtllm.

**Dormant replicas.** `handoff.dormant_rollout_replicas` (default `true`) requires vLLM with
`enable_sleep_mode=true` (`supports_dormant_replicas()`); otherwise the manager logs a warning and shuts
replicas down instead. The one-step-off recipe runs with `rollout.free_cache_engine=False`, which makes
`vLLMHttpServer.sleep()` a no-op for the replicas that must stay online; `sleep(force=True)` on
`vLLMHttpServer`, `vLLMReplica` and the `RolloutReplica` base (which raises for other backends) bypasses
that only for the replicas that make room. Sleep level 2 keeps the server actor, the engine core, the TP
worker processes and their NCCL communicators; about 0.86 GB of CUDA context per GPU remains on the
training GPUs. Pairs that have never been rollout are still launched cold -- there is no pre-launched pool.

What a switch keeps and what it rebuilds:

| Object | On a switch |
|---|---|
| hybrid worker processes, CUDA contexts, WORLD process group | kept |
| training subgroups and their NCCL communicators, DeviceMesh / DTensor caches | destroyed and recreated / cleared |
| actor / critic parameters, optimizer state, scheduler, buffers | moved shard to shard |
| vLLM server actors and their engine / TP worker processes | kept if their GPUs stay rollout; slept if their GPUs become training; woken if they return; launched only for GPU pairs that were never rollout |
| vLLM KV cache | kept on surviving replicas; freed on sleep, re-allocated on wake |
| load balancer | kept, servers added / removed in place |
| weight-sync NCCL group (checkpoint engine) | recreated with a fresh group name; `rebuild_group=True` from then on |
| agent-loop workers | recreated |

## Process groups and DTensor caches across resizes

Every hard switch destroys the training subgroup and the weight-sync engine and creates new ones. Two
things break on the *second* switch back to a topology that existed before:

* `DeviceMesh.__eq__` / `__hash__` compare content (rank list, shape, dim names) and ignore the process
  group, which the mesh references by name. DTensor's sharding propagator and redistribute planner cache
  results keyed on specs that contain the mesh. After a subgroup is destroyed and a later switch recreates
  one with identical ranks, the cache hits and returns a spec whose mesh resolves to the dead group:
  `Could not resolve the process group registered under the name N` at the first DTensor collective
  (gradient clipping is the first). `clear_dtensor_dispatch_caches()` (`verl/utils/distributed.py`) clears
  `propagate_op_sharding` (a `LocalLRUCache` in torch >= 2.9, hence `.cache`), `_propagate_tensor_meta` and
  `_gen_transform_infos`; `TrainingWorker.destroy_process_group` and `resize_runtime` call it on **every** rank, members or not.
* `NCCLCheckpointEngine.finalize()` runs twice with `rebuild_group=True` (after each `update_weights`, and
  again from `reconfigure_checkpoint_engine` when a resize retires the engine) and compared `None >= 0`.
  It now treats `rank is None` as "never joined".

Rules the hybrid runtime imposes: all `new_group` calls happen on every rank in the same order, every
worker method that touches groups is `ONE_TO_ALL` with barriers, and non-member ranks still make the calls.

## Verification

Always on:

* **structure** -- manifest from `old_ranks[0]`; every rank checks names, shapes, dtypes and placements of its
  old and new runtime (and optimizer state) against it; verdicts are exchanged so all ranks raise together.
* **sizes** -- the receive buffer is the real FSDP2 shard, so `numel == sum(splits)` checks that the planner
  and FSDP2's chunking agree; the trainer checks that every member reports the same tensor counts.

Opt-in, `VERL_RESIZE_VERIFY=1` (metric `resize/gpu_direct_verified`, `resize/gpu_direct_critic_verified`):

* `_shard_digest` computes, per tensor, two int64 sums over the raw 32-bit words of the local shard:
  `sum(v_i)` and `sum(v_i * ((global_index_i mod 2^20) + 1))`. Because the weight is the *global* flat index,
  shards of one tensor combine by plain addition however the tensor is cut, so each side reduces its digests
  inside its own mesh with one `all_reduce` -- no all-gather, no full tensors. The old side computes before
  the transfer, the new side after; the old digest is broadcast and compared element-wise. The second sum
  catches rows that landed in the wrong place; integer arithmetic on raw bits makes the comparison exact.
  Any mismatch names the tensors and raises on every rank. Cost: about 0.3 s for Qwen3-0.6B.

Offline:

* `tests/utils/test_fsdp_resize_on_cpu.py` -- planner against DTensor's own chunking, expand / shrink /
  non-prefix rank sets, uneven and empty shards, row conservation, invalid input.
* `tests/special_distributed/run_fsdp2_resize_transfer.py` (`torchrun --nproc-per-node=6`) -- 4 -> 6 -> 2 -> 6
  with AdamW state: bitwise equality against the old runtime, tied weight and non-persistent buffer
  restored, and training *continues* on the new mesh against a single-device golden model. The last check
  matters because `full_tensor()` reads `_local_tensor` through DTensor's own all-gather and would pass even
  if FSDP2's `_sharded_param_data` had not been written; training exercises FSDP2's all-gather path.
* `tests/workers/rollout/test_hybrid_replica_reuse_on_cpu.py` -- `plan_hybrid_replica_reuse` and the
  manager's retire / rebuild / sleep / wake flow with fake replicas.
* `tests/experimental/one_step_off_policy/test_dynamic_resize_control_on_cpu.py` -- trainer-level flow and
  metrics with mocked worker groups.

Why not just watch the loss: in RL the loss depends on sampled rollouts (which change with replica count,
batch composition and RNG stream), it is a single scalar, and optimizer state is invisible to it. It
catches only gross corruption.

## Configuration, environment, metrics

| Setting | Default | Effect |
|---|---|---|
| `trainer.dynamic_resize.handoff.backend` | `gpu_direct` (example yaml) | `gpu_direct`: this document; `disk_fallback`: checkpoint round trip (optimizer state not restored across world sizes) |
| `trainer.dynamic_resize.handoff.gpu_direct_low_memory` | `false` | one FSDP2 unit at a time, peak `max(old, new) + one unit` |
| `trainer.dynamic_resize.handoff.dormant_rollout_replicas` | `true` | sleep / wake vLLM replicas instead of kill / relaunch; `false` falls back to the kill path |
| `VERL_RESIZE_VERIFY` | unset | `1`: per-tensor checksum comparison after every transfer |
| `VERL_LOGGING_LEVEL` | `WARN` | `INFO` shows the `[one-step-off][resize] ...` lines from workers and the server manager |

Metrics reported on a resize step (`step:N - ... - resize/...` line):

| Metric | Meaning |
|---|---|
| `resize/gpu_direct_status` | `moved` on success |
| `resize/gpu_direct_tensor_count`, `resize/gpu_direct_optimizer_tensor_count` | parameters / optimizer tensors moved (Qwen3-0.6B: 310 / 620) |
| `resize/gpu_direct_net_recv_bytes`, `resize/gpu_direct_self_copy_bytes` | bytes received over NCCL / copied on-device; network bytes equal the lower bound of the shard layout (4 -> 6: 3/4 of the state, 6 -> 2 or 2 -> 6: 5/6) |
| `resize/gpu_direct_transfer_seconds`, `resize/gpu_direct_peak_alloc_bytes` | transfer wall time; peak device memory while old and new shards coexist |
| `resize/gpu_direct_verified` | 1.0 when `VERL_RESIZE_VERIFY=1` and all checksums matched |
| `resize/gpu_direct_critic_*` | the same for the critic |
| `resize/rollout_replicas_kept`, `_slept`, `_woken`, `_retired`, `_launched` | replica plan of the switch; `kept + woken + launched == rollout_size / TP` |
| `resize/rollout_rebuild_seconds`, `resize/hard_switch_seconds` | rollout re-plan time (milliseconds without a launch, 35-50 s with one); whole switch wall time |
| `resize/active_actor_size`, `resize/active_rollout_size` | topology after the switch |

`CKPT_DIR` stays empty with `gpu_direct`; a `global_step_*` directory means the run used `disk_fallback`.

## Measured (8x RTX 5090, PCIe, single node)

Qwen3-0.6B GRPO, `VERL_RESIZE_VERIFY=1`, [4,4] -> [6,2] -> [2,6] -> [6,2] -> [2,6] (replica 1 = GPUs 2,3,
replica 2 = GPUs 4,5, replica 3 = GPUs 6,7):

| Step | Switch | Training-state move (time, network bytes) | kept / slept / woken / launched | Rollout re-plan | Whole switch |
|---|---|---|---|---|---|
| 2 | [4,4] -> [6,2] | 1.22 s, 5.4 GB | 1 / 1 / 0 / 0 | 0.4 ms | 6.6 s |
| 4 | [6,2] -> [2,6] | 1.05 s, 6.0 GB | 1 / 0 / 1 / 1 (GPUs 2,3 first time) | 37.2 s | 43.4 s |
| 6 | [2,6] -> [6,2] | 1.15 s, 6.0 GB | 1 / 2 / 0 / 0 | 0.2 ms | 5.8 s |
| 8 | [6,2] -> [2,6] | 1.03 s, 6.0 GB | 1 / 0 / 2 / 0 | 35 ms | 6.1 s |

Before replica reuse every switch was about 45 s (34 s of vLLM start-up). A sleeping replica leaves about
0.86 GB per GPU (TP worker CUDA context) versus about 10 GB for an active one; generation speed after a
wake matches a fresh replica in the same topology. Qwen2.5-3B, 4 -> 6 -> 4: default path 1.70 / 1.53 s,
low-memory pipeline 1.70 / 1.77 s, peak device memory for 6 -> 4 16.6 GB -> 10.4 GB with the pipeline.

## Limitations and open items

* Validated on one node only. The transfer has no single-node assumption (`old ∪ new` is an arbitrary
  subgroup of WORLD), but multi-node transport and overlap with compute have not been exercised.
* A GPU pair that has never been rollout still pays a cold vLLM launch the first time (~35-50 s). An
  asynchronous join (generate on survivors while new replicas start) and a pre-launched replica pool are
  not implemented.
* Sleeping replicas leave about 0.86 GB per training GPU; use `dormant_rollout_replicas=false` when a
  large model is at the memory edge.
* `prepare_resize_target` relies on FSDP2's `FSDPModule._apply` -> `reset_sharded_param()` making the
  `to_empty`'d storage the sharded parameter's home (the same mechanism as official meta-device init); the low-memory pipeline additionally uses `FSDPParam._sharded_param_data`, `reset_sharded_param` and `swap_tensors` directly.
* `clear_dtensor_dispatch_caches` touches private DTensor caches whose names may change between torch
  versions (torch 2.9 verified).
* `disk_fallback` is kept as the alternative backend; it does not restore optimizer state across a world-size
  change.
