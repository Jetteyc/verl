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

## Environment requirements and test commands

Run everything from the repository root of a clean checkout, with `export PYTHONPATH=$PWD`. What each
check proves is described under *Verification*.

| Check | Needs | Reference environment |
|---|---|---|
| CPU tests | Python 3.12 and the packages listed under *CPU tests*; no accelerator | Python 3.12.3, torch 2.8.0+cpu, ray 2.49.2, transformers 4.57.1, tensordict 0.10.0, numpy 2.5.4, pytest 9.2.0 |
| Transfer test | one node with 6 CUDA GPUs; a CUDA build of torch with FSDP2 and NCCL | 6 of the 8 GPUs below, same software |
| End to end | one node with 8 CUDA GPUs (the launcher's default pool); vLLM and Ray; `flash_attn` when a critic is trained; a causal LM, a GSM8K parquet pair and a writable `CKPT_DIR` | 8x RTX 5090 (32 GB), NVIDIA driver 580.82.09; Python 3.12.12, torch 2.9.0+cu129 (CUDA 12.9, NCCL 2.27.5), vLLM 0.12.0, Ray 2.53.0, transformers 4.57.5, tensordict 0.10.0, hydra-core 1.3.2, numpy 1.26.4, flash_attn 2.8.1; Qwen3-0.6B, GSM8K in verl format |

### CPU tests

In a Python 3.12 virtual environment. The first command and the packages up to `psutil` are what the
`v1 integration` workflow installs for the three files it runs; the packages after `psutil` are imported
by the other files. Keep the second command in one piece: installed in a separate step, `datasets`
upgrades `huggingface-hub` beyond what `transformers==4.57.1` accepts.

```bash
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install pytest pytest-asyncio hydra-core numpy packaging tensordict==0.10.0 transformers==4.57.1 \
  ray==2.49.2 psutil codetiming peft datasets torchdata cachetools pillow uvicorn fastapi
```

```bash
python -m pytest -q -rs tests/experimental/one_step_off_policy \
  tests/workers/test_training_worker_resize_snapshot_on_cpu.py \
  tests/utils/test_fsdp_resize_on_cpu.py tests/utils/test_fsdp_load_scope_on_cpu.py \
  tests/workers/rollout/test_replica_on_cpu.py tests/workers/rollout/test_hybrid_replica_reuse_on_cpu.py
```

No test may fail; one is skipped because it needs an accelerator.

### Transfer test (6 GPUs)

```bash
torchrun --standalone --nproc-per-node=6 tests/special_distributed/run_fsdp2_resize_transfer.py
RESIZE_PIPELINE=1 torchrun --standalone --nproc-per-node=6 tests/special_distributed/run_fsdp2_resize_transfer.py
```

Rank 0 prints `FSDP2 GPU-direct resize transfer 4->6->2->6 passed (pipeline=False)`, and
`(pipeline=True)` for the low-memory pipeline that `RESIZE_PIPELINE=1` selects.

### End to end (8 GPUs)

```bash
export MODEL_PATH=<model directory> TRAIN_FILE=<gsm8k train.parquet> VAL_FILE=<gsm8k test.parquet>
export PROFILER_ENABLE=false TRAIN_BATCH_SIZE=96 PPO_MINI_BATCH_SIZE=48
export TORCH_CUDA_ARCH_LIST=12.0   # RTX 5090 (sm_120) only, as in the reference environment
L=verl/experimental/one_step_off_policy/shell/run_dynamic_resize.sh
G="algorithm.adv_estimator=grpo actor_rollout_ref.rollout.n=5 actor_rollout_ref.actor.entropy_coeff=0"
```

`$G` selects GRPO, which trains no critic; without it the launcher runs GAE with the token-classification
critic. The two batch sizes keep a run short. `$RUN` is a new, empty directory for each run.

```bash
# resize: [4,4] (actor, rollout) -> [6,2] at step 2 -> [2,6] at step 4, six steps
VERL_RESIZE_VERIFY=1 CKPT_DIR=$RUN/ckpt bash $L $G
# the same with GAE and the critic
VERL_RESIZE_VERIFY=1 CKPT_DIR=$RUN/ckpt bash $L
# the same through the low-memory pipeline
VERL_RESIZE_VERIFY=1 CKPT_DIR=$RUN/ckpt bash $L $G trainer.dynamic_resize.handoff.gpu_direct_low_memory=true
# back to an earlier topology: [4,4] -> [6,2] -> [4,4]
VERL_RESIZE_VERIFY=1 SECOND_SPLIT_PLAN="[4,4]" CKPT_DIR=$RUN/ckpt bash $L $G
# the two default switches through the checkpoint round trip
HOST_STAGING_BACKEND=disk_fallback CKPT_DIR=$RUN/ckpt bash $L $G
# dynamic resize off: separate actor and rollout workers on 4 + 4 GPUs
CKPT_DIR=$RUN/ckpt bash $L $G actor_rollout_ref.hybrid_engine=false trainer.dynamic_resize.enable=false
```

A run passes when the launcher exits with code 0 after its last step line (`step:6`) and every switch
line (`step:2` and `step:4`) reports `resize/hard_switch_success:1.0`; with resize off every step line
reports `resize/enabled:0.0` instead. In addition:

* `VERL_RESIZE_VERIFY=1`: every switch line reports `resize/gpu_direct_verified:1.0`, with the critic
  also `resize/gpu_direct_critic_verified:1.0`.
* `gpu_direct` leaves no `global_step_*` directory in `CKPT_DIR`; `disk_fallback` leaves one per switch.
* Replicas: in the default schedule the replica on GPUs 6,7 is kept by both switches
  (`resize/rollout_replicas_kept:1.0`) and the second switch launches two
  (`resize/rollout_replicas_launched:2.0`).

## The hard switch, step by step

`OneStepOffRayTrainer._apply_dynamic_resize_hard_switch` with `handoff.backend=gpu_direct`, for a switch
to `(actor_target, rollout_target)`. Nothing is written to or read from disk, and nothing is staged on the
host; the training state moves GPU to GPU.

1. **Rollout makes room** (`_destroy_dynamic_resize_runtime(actor_target)`): abort in-flight requests on
   every replica, kill the agent-loop workers, then `LLMServerManager.retire_hybrid_replicas_below(actor_target)`
   shuts down only the replicas whose GPUs become training ranks and removes them from the load balancer.
   Replicas that stay on rollout GPUs are untouched.
2. **Move the training state** (`_rebuild_hybrid_training_runtime` -> `resize_actor_runtime` on every
   physical rank): build the new FSDP2 runtime on the new ranks next to the live one, fill it shard to
   shard with one `all_to_all_single` per tensor, swap, destroy the old groups. The critic goes through the
   same `resize_runtime` call right after the actor. Details under *Training state: shard-to-shard
   transfer*.
3. **Re-point the control plane**: `actor_wg` / `critic_wg` become `RayWorkerGroup.subgroup(new actor ranks)`
   views (no Ray actors are created or destroyed), the rollout view becomes the suffix ranks, and the
   hybrid workers drop their cached server handles (`reset_rollout_server_handle`).
4. **Re-plan rollout** (`LLMServerManager.rebuild_hybrid_runtime(start_rank=actor_target // TP)`):
   `plan_hybrid_replica_reuse` diffs the live replicas against the new GPU pairs -- survivors are kept
   with engine, weights and KV cache intact, pairs new to rollout get a freshly launched replica -- and
   the load balancer is updated in place (`add_servers` / `remove_servers`). Details under *Rollout side*.
5. **Weight sync**: `reconfigure_checkpoint_engine(group_name, rebuild_group=True)` finalizes the old NCCL
   weight-sync engine and creates a new one on every rank; `CheckpointEngineManager.update_weights`
   pushes the current actor weights to the replicas and resumes generation.
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

* `LLMServerManager.retire_hybrid_replicas_below(physical_rank)` -- called before the training side allocates on those GPUs; removes the affected servers from the load balancer and shuts them down (`shutdown_runtime(kill_workers=False)`).
* `plan_hybrid_replica_reuse(live, new_workers, start_rank, replica_world_size)` (pure, CPU-tested) -- for every new GPU pair: **keep** if a live replica with the same `replica_rank`, the same worker handles and live servers sits there; otherwise **launch**. Live replicas outside the new range are retired (normally already done in step 1).
* `rebuild_hybrid_runtime` executes the plan: kept replicas only update `worker_group_rank`, launched ones run `init_hybrid`; the load balancer gets `add_servers`; the agent-loop manager is recreated. The old kill-everything path remains as `_rebuild_hybrid_runtime_from_scratch` for trtllm.

What a switch keeps and what it rebuilds:

| Object | On a switch |
|---|---|
| hybrid worker processes, CUDA contexts, WORLD process group | kept |
| training subgroups and their NCCL communicators, DeviceMesh / DTensor caches | destroyed and recreated / cleared |
| actor / critic parameters, optimizer state, scheduler, buffers | moved shard to shard |
| vLLM server actors and their engine / TP worker processes | kept if their GPUs stay rollout; shut down if their GPUs become training; launched for GPU pairs that join rollout |
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
  manager's retire / rebuild flow with fake replicas.
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
| `resize/rollout_replicas_kept`, `_retired`, `_launched` | replica plan of the switch; `kept + launched == rollout_size / TP` |
| `resize/rollout_rebuild_seconds`, `resize/hard_switch_seconds` | rollout re-plan time (milliseconds without a launch, 35-50 s with one); whole switch wall time |
| `resize/active_actor_size`, `resize/active_rollout_size` | topology after the switch |

`CKPT_DIR` stays empty with `gpu_direct`; a `global_step_*` directory means the run used `disk_fallback`.

## Measured (8x RTX 5090, PCIe, single node)

* Qwen3-0.6B GRPO, [4,4] -> [6,2] -> [2,6] -> [6,2]: the training-state move takes 1.0-1.4 s per switch
  and verifies. A switch that shrinks rollout (no replica to launch) takes 6.8 s end to end, down from
  about 45 s; a switch that expands rollout takes 56 s because the new replicas' start-up is on the
  critical path. The replica on GPUs 6,7 kept the same server process through all three switches.
* Qwen2.5-3B, 4 -> 6 -> 4: default path 1.70 / 1.53 s, low-memory pipeline 1.70 / 1.77 s; peak device
  memory for 6 -> 4 drops from 16.6 GB to 10.4 GB with the pipeline.

## Limitations and open items

* Validated on one node only. The transfer has no single-node assumption (`old ∪ new` is an arbitrary
  subgroup of WORLD), but multi-node transport and overlap with compute have not been exercised.
* Switches that give GPUs back to rollout pay a cold vLLM launch for the new replicas (~35-50 s). Hiding
  it needs either dormant replicas or an asynchronous join of new replicas after generation resumes.
* `prepare_resize_target` relies on FSDP2's `FSDPModule._apply` -> `reset_sharded_param()` making the
  `to_empty`'d storage the sharded parameter's home (the same mechanism as official meta-device init); the low-memory pipeline additionally uses `FSDPParam._sharded_param_data`, `reset_sharded_param` and `swap_tensors` directly.
* `clear_dtensor_dispatch_caches` touches private DTensor caches whose names may change between torch
  versions (torch 2.9 verified).
* `disk_fallback` is kept as the alternative backend; it does not restore optimizer state across a world-size
  change.
