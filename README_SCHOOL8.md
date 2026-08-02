# XLand ExactCraft joint NanoWM bundle (school8)

This directory contains the code required to train the RGB-only joint
video/action model for:

`XLand-MiniGrid-ExactCraft-Easy-8x8-v1`

The dataset is intentionally **not** included.

## Layout

- `repos/nano-world-model` — NanoWM plus commit `fd91e0d`, adding XLand
  loading, future-video prediction, and a six-class discrete action head.
- `repos/xland-minigrid-five-object` — the XLand environment and new crafting
  tasks. PPO runs/checkpoints are excluded.
- `src` — collection, resize, closed-loop evaluation, and action-server tools.
- `bin` — portable school8 setup and launch scripts.
- `data`, `runs`, `logs` — empty runtime directories.

## Model contract

- Input: one `128x128` RGB frame.
- Outputs: three future video frames and three discrete actions.
- Actions: forward, turn clockwise, turn counterclockwise, pick up, put down,
  toggle.
- `rule_encoding` and `goal_encoding` are not exposed to the model.
- The single goal is visually specified by the yellow star.
- Offline validation runs every 10,000 optimization steps.
- Checkpoints are written every 5,000 steps.

## Dataset location

Put or mount the external 128px dataset at:

```text
/home/User8/xland_joint_wam/data/exactcraft_easy_10k_128/train/
```

The directory may contain shard subdirectories. Each episode must be an
`episode_train_*.npz` file with:

- `rgb`: `[T+1, 128, 128, 3] uint8`
- `actions`: `[T] int64`, values in `[0, 5]`

The loader performs an internal 90/10 split of these train trajectories. No
held-out task-bank trajectories are required.

## Setup and launch

```bash
cd /home/User8/xland_joint_wam
bash bin/prepare_envs.sh
GPU=0 bash bin/run_train_128.sh
PORT=6008 bash bin/run_tensorboard.sh
```

The default run directory is:

```text
/home/User8/xland_joint_wam/runs/xland_joint_128
```

Environment names and paths can be overridden:

```bash
NANOWM_ENV=nanowm-xland \
DATASET_ROOT=/path/to/data \
RESULTS_ROOT=/path/to/results \
RUN_NAME=my_run \
GPU=2 \
bash bin/run_train_128.sh
```

## Important configs

- `repos/nano-world-model/src/configs/dataset/xland/exactcraft_easy.yaml`
- `repos/nano-world-model/src/configs/model/nanowm_xland_s4.yaml`
- `repos/nano-world-model/src/configs/experiment/xland_joint.yaml`

## Read-only collaborators

`user8_1`, `user8_2`, `user8_3`, `user8_4`, and `user8_5` have ACL read and
directory traversal access to this bundle. They can copy it into their own
homes, but cannot modify the shared original.
