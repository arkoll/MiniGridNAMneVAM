# Privileged PPO training and active runs

Last updated: 2026-07-30 10:29 Europe/Moscow.

This is the hand-off document for AI agents working on RL trajectory
collection. The authoritative server checkout is:

```text
/home/User9/xland-minigrid-five-object
```

The conda environment is `xland-five-object`. The training entry point is:

```text
/home/User9/xland-minigrid-five-object/training/train_privileged_crafting_ppo.py
```

## Why this policy should learn quickly

The algorithm is PPO with a deliberately privileged Markov observation:

- complete fixed-orientation 8×8 symbolic map;
- agent direction and visible carried-object tile/color;
- active ground-truth goal and rule encodings;
- one-hot ground-truth crafting phase;
- ground-truth valid-action mask;
- potential-based dense credit for `ingredients → hexagon → blue key → open
  door`;
- native task success with a larger reward scale.

The actor never receives the correct next action. It must still learn
navigation, pickup/drop ordering, rule composition, door opening, and reaching
the star. The extra inputs are allowed because these policies are data
collectors, not the world/action model being evaluated.

The network is an MLP over embedded symbolic tokens. Do not change it back to a
cuDNN convolution without first fixing the server's CUDA/cuDNN compatibility:
V100 convolution autotuning currently fails with cuDNN status 5003.

## Success-rate semantics

`train_success_rate` is in-distribution SR over the training task bank.
`val_success_rate` is held-out compositional/color-binding SR:

- ExactCraft Easy: six train trees, three held-out validation trees;
- ShapeCraft Easy: 324 train shape/color tasks, 324 complementary validation
  tasks;
- Stride2-Adaptive Easy: the same six/three ExactCraft split under different
  locomotion.

SR counts an episode as successful only if the native
`AgentNearGoal(yellow star)` terminates it. Dense shaping is used for PPO
optimization but is not counted as success.

`train_success_rate` and `val_success_rate` use deterministic `argmax`
actions. `train/rollout_success_rate` uses the stochastic categorical policy
that generates PPO experience. Both use the same native success flag, but they
answer different questions. Threshold checkpoint names (`train_sr_20`, etc.)
refer to the stricter deterministic train SR.

## Active runs

All runs started on 2026-07-30 and have a four-hour wall-clock budget after
JAX compilation, capped at 200M environment transitions.

| tmux session | Physical GPU | Environment | Seed | Run directory |
|---|---:|---|---:|---|
| `xland_rl_exact_easy` | 0 | `XLand-MiniGrid-ExactCraft-Easy-8x8-v1` | 41 | `runs/privileged_rl_easy_20260730/exact_easy` |
| `xland_rl_shape_easy` | 1 | `XLand-MiniGrid-ShapeCraft-Easy-8x8-v1` | 42 | `runs/privileged_rl_easy_20260730/shape_easy` |
| `xland_rl_stride2_easy` | 2 | `XLand-MiniGrid-EmbodiedExactCraft-Stride2-Adaptive-Easy-8x8-v1` | 43 | `runs/privileged_rl_easy_20260730/stride2_easy` |

The active process log in each directory is `run_active.log`. Earlier
`run.log` and `run_retry.log` files are preserved diagnostics from the
discarded cuDNN attempts; they contain no training progress or checkpoints.

GPU 3 is intentionally unused. It reports two volatile uncorrectable ECC
errors and fails JAX with `CUDA_ERROR_ECC_UNCORRECTABLE`. This requires a
sysadmin GPU reset or hardware remediation, not a training-code change.

### Verified live metrics

Snapshot at 2026-07-30 10:29 Europe/Moscow:

| Run | Steps | Stochastic rollout SR | Deterministic train SR | Deterministic val SR | Throughput |
|---|---:|---:|---:|---:|---:|
| ExactCraft Easy | 55.57M | 99.6% | 92.2% | 71.4% | 149k steps/s |
| ShapeCraft Easy | 54.53M | 99.4% | 87.0% | 68.8% | 150k steps/s |
| Stride2-Adaptive Easy | 53.48M | 99.4% | 92.7% | 74.5% | 150k steps/s |

All three tmux sessions were alive and physical GPUs 0/1/2 were at 100%
utilization at this snapshot. All three runs had saved deterministic
`train_sr_05`, `train_sr_20`, `train_sr_50`, and `train_sr_80` checkpoints;
ExactCraft had also saved `train_sr_95`. Training was still active.

## TensorBoard

TensorBoard runs in tmux session `xland_rl_tb`:

```text
http://SERVER_HOST:6007/
```

Log root:

```text
/home/User9/xland-minigrid-five-object/runs/privileged_rl_easy_20260730
```

If port 6007 is not externally exposed, use an SSH tunnel:

```bash
ssh -L 6007:localhost:6007 User9@SERVER_HOST
```

Then open `http://localhost:6007/`.

## Checkpoints and metrics

Each run contains:

```text
config.json
tensorboard/events.out.tfevents.*
checkpoints/latest/
checkpoints/best_val/
checkpoints/train_sr_05/
checkpoints/train_sr_20/
checkpoints/train_sr_50/
checkpoints/train_sr_80/
checkpoints/train_sr_95/
checkpoints/final/
finished.json
```

Threshold directories appear when their train SR is first crossed. Every
checkpoint directory contains `train_state.msgpack` and `metadata.json`.
`train_state.msgpack` includes policy parameters, value parameters, optimizer
state, and optimizer step, so it can be used both for inference and exact
training continuation.

Important TensorBoard scalars:

- `eval/train_success_rate`;
- `eval/val_success_rate`;
- `train/rollout_success_rate`;
- `train/loss`, `train/actor_loss`, `train/value_loss`,
  `train/entropy`;
- `system/steps_per_second`, `system/elapsed_hours`.

Evaluation occurs every four training blocks, approximately every 1.05M
transitions with the current configuration.

## Monitoring

```bash
tmux ls
tail -f runs/privileged_rl_easy_20260730/exact_easy/run_active.log
tail -f runs/privileged_rl_easy_20260730/shape_easy/run_active.log
tail -f runs/privileged_rl_easy_20260730/stride2_easy/run_active.log
nvidia-smi
```

Do not treat the device string `cuda:0` inside every log as a mapping error.
Each process has a different `CUDA_VISIBLE_DEVICES`, so its assigned physical
GPU is remapped to local device zero.

## Resume

Use the same environment, run directory, and GPU assignment, adding:

```text
--resume RUN_DIR/checkpoints/latest/train_state.msgpack
```

For a new experiment, use a new run directory so existing TensorBoard events
and checkpoints are not overwritten. A SIGTERM/SIGINT requests a graceful
final checkpoint at the next completed training block.
