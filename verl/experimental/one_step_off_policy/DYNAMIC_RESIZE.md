# Dynamic resize in the hybrid one-step-off trainer

This document describes how `verl.experimental.one_step_off_policy` moves GPUs between rollout and
training while a run is in progress ("dynamic resize"): what a resize is, how to run the single-node
example, what happens during a hard switch, and how the result is verified. It is the reader's guide to
the code; the commit messages carry the per-change rationale.

Related files:

* `README.md` -- the one-step-off recipe itself (pipeline, parameter sync, configuration).
* `DYNAMIC_RESIZE_DIRECT_RESHARD_PLAN.md` -- the design plan for direct (checkpoint-free) resharding.
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

`OneStepOffRayTrainer._apply_dynamic_resize_hard_switch` currently does, for a switch to
`(actor_target, rollout_target)`:

1. Save a full checkpoint of actor (and critic) and, if `handoff.stage_optimizer`, page each rank's
   optimizer state to the host staging directory.
2. Quiesce rollout: abort in-flight requests on every replica, kill the agent-loop workers.
3. `handoff.backend=gpu_direct` and an *expansion*: `FSDPEngine.stage_model_for_resize` sends each old
   rank's parameter shards over a temporary NCCL group into a staging buffer on the new ranks
   (`verl/utils/fsdp_resize.py`). Shrinks, and any staging failure, fall back to the checkpoint.
4. Rebuild the training runtime: destroy the training subgroup, create the new one, rebuild the FSDP2
   engine on the new ranks from the pretrained weights, then either copy the staged shards in or
   `load_checkpoint`. Optimizer files are per world size, so across a world-size change the optimizer
   restarts from a fresh state (the plan document calls this "effectively weights_only").
5. Kill and relaunch every vLLM server actor on the new rollout ranks, rebuild the load balancer and the
   agent-loop manager.
6. `reconfigure_checkpoint_engine` with a fresh group name, then `update_weights` pushes the actor
   weights to the new replicas and resumes generation.

Steps 4 and 6 destroy and recreate process groups on every switch; the caveats this creates are
described under *Process groups and DTensor caches across resizes* below.

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
  `_gen_transform_infos`; `TrainingWorker.destroy_process_group` calls it on **every** rank, members or not.
* `NCCLCheckpointEngine.finalize()` runs twice with `rebuild_group=True` (after each `update_weights`, and
  again from `reconfigure_checkpoint_engine` when a resize retires the engine) and compared `None >= 0`.
  It now treats `rank is None` as "never joined".

Rules the hybrid runtime imposes: all `new_group` calls happen on every rank in the same order, every
worker method that touches groups is `ONE_TO_ALL` with barriers, and non-member ranks still make the calls.
