"""Fast privileged PPO for the Easy crafting environments.

The collector policy is deliberately privileged.  It receives the fixed
full-map symbolic observation, carried tile, active goal/rules, exact crafting
progress, and a ground-truth valid-action mask.  Native task completion remains
the success criterion.  Potential-based progress shaping only makes sparse
exploration practical on a short project schedule.

This script trains one policy across every train task in a family and evaluates
the same checkpoint on both the train and held-out validation task banks.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import distrax
import flax.linen as nn
from flax import serialization, struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from tensorboardX import SummaryWriter

import xminigrid
from xminigrid.benchmarks import Benchmark
from xminigrid.core.constants import DIRECTIONS, NUM_COLORS, NUM_TILES, Colors, Tiles
from xminigrid.core.grid import check_can_put, check_pickable, check_walkable
from xminigrid.types import RuleSet
from xminigrid.wrappers import (
    GymAutoResetWrapper,
    RulesAndGoalsObservationWrapper,
    Wrapper,
)


jax.config.update("jax_threefry_partitionable", True)


SUPPORTED_ENVS = (
    "XLand-MiniGrid-ExactCraft-Easy-8x8-v1",
    "XLand-MiniGrid-ShapeCraft-Easy-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Stride2-Adaptive-Easy-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Omni8-Easy-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Crab-Easy-8x8-v1",
)

OMNI_DIRECTIONS = jnp.asarray(
    (
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ),
    dtype=jnp.int32,
)


@dataclass
class Config:
    env_id: str
    run_dir: str
    hours: float = 4.0
    total_timesteps: int = 200_000_000
    num_envs: int = 512
    rollout_steps: int = 64
    updates_per_block: int = 8
    update_epochs: int = 2
    num_minibatches: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_every_blocks: int = 4
    eval_episodes: int = 192
    seed: int = 42
    resume: str | None = None

    @property
    def transitions_per_update(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def transitions_per_block(self) -> int:
        return self.transitions_per_update * self.updates_per_block


def _progress(state) -> jax.Array:
    """0=ingredients, 1=D/hex, 2=key, 3=open door."""

    grid_types = state.grid[..., 0]
    pocket_type = state.agent.pocket[0]
    has_intermediate = jnp.any(grid_types == Tiles.HEX) | (pocket_type == Tiles.HEX)
    has_key = jnp.any(grid_types == Tiles.KEY) | (pocket_type == Tiles.KEY)
    door_open = jnp.any(grid_types == Tiles.DOOR_OPEN)
    return jnp.where(door_open, 3, jnp.where(has_key, 2, jnp.where(has_intermediate, 1, 0)))


def _action_mask(state, movement: str) -> jax.Array:
    """Ground-truth mask for the six standard crafting actions."""

    if movement == "omni8":
        delta = jax.lax.dynamic_index_in_dim(
            OMNI_DIRECTIONS,
            state.agent.direction % 8,
            keepdims=False,
        )
    else:
        delta = jax.lax.dynamic_index_in_dim(
            DIRECTIONS,
            state.agent.direction % 4,
            keepdims=False,
        )
    target = jnp.clip(
        state.agent.position + delta,
        min=jnp.asarray((0, 0)),
        max=jnp.asarray((state.grid.shape[0] - 1, state.grid.shape[1] - 1)),
    )
    target_tile = state.grid[target[0], target[1]]
    pocket = state.agent.pocket
    pocket_empty = pocket[0] == Tiles.EMPTY
    if movement == "crab":
        move_delta = jax.lax.dynamic_index_in_dim(
            DIRECTIONS,
            (state.agent.direction + 1) % 4,
            keepdims=False,
        )
        move_target = jnp.clip(
            state.agent.position + move_delta,
            min=jnp.asarray((0, 0)),
            max=jnp.asarray((state.grid.shape[0] - 1, state.grid.shape[1] - 1)),
        )
        forward_ok = check_walkable(state.grid, move_target)
    elif movement == "omni8":
        orthogonal_y = jnp.clip(
            state.agent.position + jnp.asarray((delta[0], 0)),
            min=jnp.asarray((0, 0)),
            max=jnp.asarray((state.grid.shape[0] - 1, state.grid.shape[1] - 1)),
        )
        orthogonal_x = jnp.clip(
            state.agent.position + jnp.asarray((0, delta[1])),
            min=jnp.asarray((0, 0)),
            max=jnp.asarray((state.grid.shape[0] - 1, state.grid.shape[1] - 1)),
        )
        diagonal = (delta[0] != 0) & (delta[1] != 0)
        corner_clear = check_walkable(state.grid, orthogonal_y) & check_walkable(
            state.grid,
            orthogonal_x,
        )
        forward_ok = check_walkable(state.grid, target) & jnp.where(
            diagonal,
            corner_clear,
            True,
        )
    else:
        # Standard and Stride2-Adaptive can move iff the first microstep is
        # walkable. Stride2 then optionally executes its second microstep.
        forward_ok = check_walkable(state.grid, target)
    pickup_ok = check_pickable(state.grid, target) & pocket_empty
    putdown_ok = check_can_put(state.grid, target) & (~pocket_empty)
    locked_ok = (
        (target_tile[0] == Tiles.DOOR_LOCKED)
        & (pocket[0] == Tiles.KEY)
        & (pocket[1] == target_tile[1])
    )
    toggle_ok = locked_ok | (target_tile[0] == Tiles.DOOR_CLOSED) | (target_tile[0] == Tiles.DOOR_OPEN)
    return jnp.stack(
        (
            forward_ok,
            jnp.asarray(True),
            jnp.asarray(True),
            pickup_ok,
            putdown_ok,
            toggle_ok,
        )
    )


class PrivilegedCraftingWrapper(Wrapper):
    """Adds privileged features, an action mask, and potential shaping."""

    success_scale: float = 10.0
    stage_scale: float = 0.30
    pocket_scale: float = 0.03

    def __init__(self, env, movement: str):
        super().__init__(env)
        self.movement = movement

    def observation_shape(self, params):
        shape = dict(self._env.observation_shape(params))
        shape.update({"progress": (4,), "action_mask": (6,)})
        return shape

    def _potential(self, state):
        pocket_nonempty = state.agent.pocket[0] != Tiles.EMPTY
        return self.stage_scale * _progress(state) + self.pocket_scale * pocket_nonempty

    def _extend(self, timestep):
        observation = dict(timestep.observation)
        observation["progress"] = jax.nn.one_hot(_progress(timestep.state), 4)
        observation["action_mask"] = _action_mask(
            timestep.state,
            self.movement,
        )
        return timestep.replace(observation=observation)

    def reset(self, params, key):
        return self._extend(self._env.reset(params, key))

    def step(self, params, timestep, action):
        old_potential = self._potential(timestep.state)
        next_timestep = self._env.step(params, timestep, action)
        new_potential = self._potential(next_timestep.state)
        shaped_reward = (
            self.success_scale * next_timestep.reward
            + self.gamma * new_potential
            - old_potential
        )
        return self._extend(next_timestep.replace(reward=shaped_reward))

    @property
    def gamma(self):
        return 0.99


class PrivilegedActorCritic(nn.Module):
    num_actions: int = 6
    embed_dim: int = 16
    hidden_dim: int = 256
    agent_direction_classes: int = 5

    @nn.compact
    def __call__(self, observation):
        img = observation["img"].astype(jnp.int32)
        tile_emb = nn.Embed(NUM_TILES, self.embed_dim)(img[..., 0])
        color_emb = nn.Embed(NUM_COLORS, self.embed_dim)(img[..., 1])
        agent_emb = jax.nn.one_hot(
            img[..., 2],
            self.agent_direction_classes,
        )
        grid_features = jnp.concatenate((tile_emb, color_emb, agent_emb), axis=-1)
        grid_features = grid_features.reshape((*grid_features.shape[:-3], -1))
        # A flat MLP is intentional.  The map is only 8x8 and absolute world
        # coordinates are task-relevant.  It also avoids a cuDNN dependency,
        # making runs robust on older V100 driver/runtime combinations.
        grid_features = nn.relu(nn.Dense(512)(grid_features))
        grid_features = nn.relu(nn.Dense(self.hidden_dim)(grid_features))

        pocket = observation["pocket"].astype(jnp.int32)
        pocket_features = jnp.concatenate(
            (
                nn.Embed(NUM_TILES, self.embed_dim)(pocket[..., 0]),
                nn.Embed(NUM_COLORS, self.embed_dim)(pocket[..., 1]),
            ),
            axis=-1,
        )

        # Encodings are short integer programs.  Token one-hot keeps their
        # identities exact instead of imposing an arbitrary ordinal geometry.
        rule_tokens = jax.nn.one_hot(observation["rule_encoding"].astype(jnp.int32), 32)
        goal_tokens = jax.nn.one_hot(observation["goal_encoding"].astype(jnp.int32), 32)
        program_features = jnp.concatenate(
            (
                rule_tokens.reshape((*rule_tokens.shape[:-3], -1)),
                goal_tokens.reshape((*goal_tokens.shape[:-2], -1)),
            ),
            axis=-1,
        )
        program_features = nn.relu(nn.Dense(128)(program_features))

        features = jnp.concatenate(
            (
                grid_features,
                pocket_features,
                program_features,
                observation["direction"],
                observation["progress"],
            ),
            axis=-1,
        )
        features = nn.tanh(nn.Dense(self.hidden_dim)(features))
        features = nn.tanh(nn.Dense(self.hidden_dim)(features))

        logits = nn.Dense(
            self.num_actions,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(features)
        logits = jnp.where(observation["action_mask"], logits, -1e9)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(features)
        return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)


class Transition(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    success: jax.Array
    log_prob: jax.Array
    observation: Any


def _make_benchmark(tasks) -> Benchmark:
    return Benchmark(
        goals=jnp.stack(tuple(task.ruleset.goal for task in tasks)),
        rules=jnp.stack(tuple(task.ruleset.rules for task in tasks)),
        init_tiles=jnp.stack(tuple(task.ruleset.init_tiles for task in tasks)),
        num_rules=jnp.full((len(tasks),), tasks[0].ruleset.rules.shape[0], dtype=jnp.int32),
    )


def task_banks(env_id: str) -> tuple[Benchmark, Benchmark]:
    if env_id == "XLand-MiniGrid-ShapeCraft-Easy-8x8-v1":
        from xminigrid.envs.shape_crafting_easy import TRAIN_TASKS, VAL_TASKS
    else:
        from xminigrid.envs.easy_crafting import TRAIN_TASKS, VAL_TASKS
    return _make_benchmark(TRAIN_TASKS), _make_benchmark(VAL_TASKS)


class FlexibleDirectionObservationWrapper(Wrapper):
    """Direction observation for both four- and eight-heading embodiments."""

    def __init__(self, env, direction_count: int):
        super().__init__(env)
        self.direction_count = direction_count

    def observation_shape(self, params):
        base_shape = self._env.observation_shape(params)
        return {**base_shape, "direction": self.direction_count}

    def _extend(self, timestep):
        observation = dict(timestep.observation)
        observation["direction"] = jax.nn.one_hot(
            timestep.state.agent.direction,
            self.direction_count,
        )
        return timestep.replace(observation=observation)

    def reset(self, params, key):
        return self._extend(self._env.reset(params, key))

    def step(self, params, timestep, action):
        return self._extend(self._env.step(params, timestep, action))


def embodiment_spec(env_id: str) -> tuple[str, int]:
    if "Omni8" in env_id:
        return "omni8", 8
    if "Crab" in env_id:
        return "crab", 4
    if "Stride2" in env_id:
        return "stride2", 4
    return "standard", 4


def make_env(env_id: str, autoreset: bool):
    env, params = xminigrid.make(env_id)
    movement, direction_count = embodiment_spec(env_id)
    env = FlexibleDirectionObservationWrapper(env, direction_count)
    env = RulesAndGoalsObservationWrapper(env)
    env = PrivilegedCraftingWrapper(env, movement)
    if autoreset:
        env = GymAutoResetWrapper(env)
    return env, params


def _sample_params(params, benchmark: Benchmark, keys):
    rulesets = jax.vmap(benchmark.sample_ruleset)(keys)
    return params.replace(ruleset=rulesets)


def calculate_gae(transitions, last_value, gamma, gae_lambda):
    def step(carry, transition):
        gae, next_value = carry
        not_done = 1.0 - transition.done
        delta = transition.reward + gamma * next_value * not_done - transition.value
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        step,
        (jnp.zeros_like(last_value), last_value),
        transitions,
        reverse=True,
    )
    return advantages, advantages + transitions.value


def make_train_block(env, base_params, benchmark, config: Config, network):
    batch_size = config.transitions_per_update
    if batch_size % config.num_minibatches:
        raise ValueError("num_envs * rollout_steps must be divisible by num_minibatches")

    def ppo_update(train_state, transitions, advantages, targets, rng):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        flat = jtu.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), transitions)
        flat_advantages = advantages.reshape((batch_size,))
        flat_targets = targets.reshape((batch_size,))

        def epoch(carry, _):
            train_state, rng = carry
            rng, permutation_key = jax.random.split(rng)
            permutation = jax.random.permutation(permutation_key, batch_size)
            shuffled = jtu.tree_map(lambda x: x[permutation], (flat, flat_advantages, flat_targets))
            minibatches = jtu.tree_map(
                lambda x: x.reshape((config.num_minibatches, -1) + x.shape[1:]),
                shuffled,
            )

            def minibatch_step(train_state, minibatch):
                mb_transition, mb_advantage, mb_target = minibatch

                def loss_fn(params):
                    dist, value = network.apply(params, mb_transition.observation)
                    log_prob = dist.log_prob(mb_transition.action)
                    ratio = jnp.exp(log_prob - mb_transition.log_prob)
                    actor_loss = -jnp.minimum(
                        mb_advantage * ratio,
                        mb_advantage * jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps),
                    ).mean()
                    value_clipped = mb_transition.value + jnp.clip(
                        value - mb_transition.value,
                        -config.clip_eps,
                        config.clip_eps,
                    )
                    value_loss = 0.5 * jnp.maximum(
                        jnp.square(value - mb_target),
                        jnp.square(value_clipped - mb_target),
                    ).mean()
                    entropy = dist.entropy().mean()
                    total = actor_loss + config.vf_coef * value_loss - config.ent_coef * entropy
                    return total, (actor_loss, value_loss, entropy)

                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                actor_loss, value_loss, entropy = aux
                return train_state, jnp.stack((loss, actor_loss, value_loss, entropy))

            train_state, losses = jax.lax.scan(minibatch_step, train_state, minibatches)
            return (train_state, rng), losses.mean(axis=0)

        (train_state, rng), losses = jax.lax.scan(
            epoch,
            (train_state, rng),
            None,
            config.update_epochs,
        )
        return train_state, rng, losses.mean(axis=0)

    def train_block(train_state, rng):
        rng, task_key, reset_key = jax.random.split(rng, 3)
        task_keys = jax.random.split(task_key, config.num_envs)
        reset_keys = jax.random.split(reset_key, config.num_envs)
        params = _sample_params(base_params, benchmark, task_keys)
        timestep = jax.vmap(env.reset, in_axes=(0, 0))(params, reset_keys)

        def update_step(carry, _):
            train_state, rng, timestep = carry

            def env_step(carry, _):
                rng, timestep = carry
                rng, action_key = jax.random.split(rng)
                dist, value = network.apply(train_state.params, timestep.observation)
                action, log_prob = dist.sample_and_log_prob(seed=action_key)
                next_timestep = jax.vmap(env.step, in_axes=(0, 0, 0))(params, timestep, action)
                transition = Transition(
                    done=next_timestep.last().astype(jnp.float32),
                    action=action,
                    value=value,
                    reward=next_timestep.reward,
                    success=(next_timestep.reward >= 0.9) & next_timestep.last(),
                    log_prob=log_prob,
                    observation=timestep.observation,
                )
                return (rng, next_timestep), transition

            (rng, timestep), transitions = jax.lax.scan(
                env_step,
                (rng, timestep),
                None,
                config.rollout_steps,
            )
            _, last_value = network.apply(train_state.params, timestep.observation)
            advantages, targets = calculate_gae(
                transitions,
                last_value,
                config.gamma,
                config.gae_lambda,
            )
            train_state, rng, losses = ppo_update(
                train_state,
                transitions,
                advantages,
                targets,
                rng,
            )
            episodes = transitions.done.sum()
            successes = transitions.success.sum()
            metrics = jnp.concatenate(
                (
                    losses,
                    jnp.asarray(
                        (
                            transitions.reward.mean(),
                            episodes,
                            successes,
                            successes / jnp.maximum(episodes, 1),
                        )
                    ),
                )
            )
            return (train_state, rng, timestep), metrics

        (train_state, rng, _), metrics = jax.lax.scan(
            update_step,
            (train_state, rng, timestep),
            None,
            config.updates_per_block,
        )
        return train_state, rng, metrics.mean(axis=0)

    return jax.jit(train_block)


def make_eval_fn(env, base_params, benchmark, config: Config, network):
    eval_count = config.eval_episodes
    # Cover the whole bank approximately uniformly.  Small banks are repeated
    # over many independent reset seeds so SR is not a six-episode estimate.
    indices = (
        np.arange(eval_count, dtype=np.int32)
        * benchmark.num_rulesets()
        // eval_count
    )
    rulesets = RuleSet(
        goal=benchmark.goals[indices],
        rules=benchmark.rules[indices],
        init_tiles=benchmark.init_tiles[indices],
    )
    params = base_params.replace(ruleset=rulesets)

    def eval_policy(policy_params, key):
        keys = jax.random.split(key, eval_count)
        timestep = jax.vmap(env.reset, in_axes=(0, 0))(params, keys)
        done = jnp.zeros((eval_count,), dtype=jnp.bool_)
        success = jnp.zeros((eval_count,), dtype=jnp.bool_)
        lengths = jnp.zeros((eval_count,), dtype=jnp.int32)

        def one_step(carry, _):
            timestep, done, success, lengths = carry
            dist, _ = network.apply(policy_params, timestep.observation)
            action = dist.mode()

            def step_if_active(single_params, single_timestep, single_action, single_done):
                return jax.lax.cond(
                    single_done,
                    lambda: single_timestep,
                    lambda: env.step(single_params, single_timestep, single_action),
                )

            next_timestep = jax.vmap(step_if_active)(params, timestep, action, done)
            new_success = (~done) & next_timestep.last() & (next_timestep.reward >= 0.9)
            success = success | new_success
            lengths = lengths + (~done)
            done = done | next_timestep.last()
            return (next_timestep, done, success, lengths), None

        (_, done, success, lengths), _ = jax.lax.scan(
            one_step,
            (timestep, done, success, lengths),
            None,
            base_params.max_steps,
        )
        return success.mean(), lengths.mean(), done.mean()

    return jax.jit(eval_policy)


def save_checkpoint(path: Path, train_state, metadata: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "train_state.msgpack").write_bytes(serialization.to_bytes(jax.device_get(train_state)))
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def train(config: Config):
    if config.env_id not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported env_id {config.env_id!r}; choose one of {SUPPORTED_ENVS}")

    run_dir = Path(config.run_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True))
    writer = SummaryWriter(str(tensorboard_dir))

    train_benchmark, val_benchmark = task_banks(config.env_id)
    train_env, base_params = make_env(config.env_id, autoreset=True)
    eval_env, eval_base_params = make_env(config.env_id, autoreset=False)
    _, direction_count = embodiment_spec(config.env_id)
    network = PrivilegedActorCritic(
        num_actions=train_env.num_actions(base_params),
        agent_direction_classes=direction_count + 1,
    )

    rng = jax.random.key(config.seed)
    rng, init_key, reset_key = jax.random.split(rng, 3)
    init_params = base_params.replace(ruleset=train_benchmark.get_ruleset(0))
    init_timestep = train_env.reset(init_params, reset_key)
    network_params = network.init(init_key, jtu.tree_map(lambda x: x[None], init_timestep.observation))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate * 0.1,
        peak_value=config.learning_rate,
        warmup_steps=100,
        decay_steps=max(
            101,
            config.total_timesteps
            // config.transitions_per_update
            * config.update_epochs
            * config.num_minibatches,
        ),
        end_value=config.learning_rate * 0.05,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(schedule, eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=optimizer)
    if config.resume:
        train_state = serialization.from_bytes(train_state, Path(config.resume).read_bytes())

    train_block = make_train_block(train_env, base_params, train_benchmark, config, network)
    eval_train = make_eval_fn(eval_env, eval_base_params, train_benchmark, config, network)
    eval_val = make_eval_fn(eval_env, eval_base_params, val_benchmark, config, network)

    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    print(json.dumps(
        {
            "event": "start",
            "env_id": config.env_id,
            "device": str(jax.devices()[0]),
            "train_tasks": train_benchmark.num_rulesets(),
            "val_tasks": val_benchmark.num_rulesets(),
            "run_dir": str(run_dir),
        },
        sort_keys=True,
    ), flush=True)
    compile_start = time.time()
    train_state, rng, _ = jax.block_until_ready(train_block(train_state, rng))
    compile_seconds = time.time() - compile_start
    total_steps = config.transitions_per_block
    block = 1
    start_time = time.time()
    threshold_saved: set[int] = set()
    best_val = -1.0

    while total_steps < config.total_timesteps and not stop_requested:
        train_start = time.time()
        train_state, rng, metrics = jax.block_until_ready(train_block(train_state, rng))
        block_seconds = time.time() - train_start
        total_steps += config.transitions_per_block
        block += 1
        loss, actor_loss, value_loss, entropy, reward, episodes, successes, rollout_sr = map(float, metrics)
        elapsed = time.time() - start_time
        sps = config.transitions_per_block / max(block_seconds, 1e-6)

        scalar_metrics = {
            "train/loss": loss,
            "train/actor_loss": actor_loss,
            "train/value_loss": value_loss,
            "train/entropy": entropy,
            "train/mean_shaped_reward": reward,
            "train/episodes_per_update": episodes,
            "train/successes_per_update": successes,
            "train/rollout_success_rate": rollout_sr,
            "system/steps_per_second": sps,
            "system/elapsed_hours": elapsed / 3600,
            "system/compile_seconds": compile_seconds,
        }
        for name, value in scalar_metrics.items():
            writer.add_scalar(name, value, total_steps)

        should_eval = block == 2 or block % config.eval_every_blocks == 0
        if should_eval:
            rng, train_eval_key, val_eval_key = jax.random.split(rng, 3)
            train_sr, train_len, train_done = map(
                float,
                jax.block_until_ready(eval_train(train_state.params, train_eval_key)),
            )
            val_sr, val_len, val_done = map(
                float,
                jax.block_until_ready(eval_val(train_state.params, val_eval_key)),
            )
            eval_metrics = {
                "eval/train_success_rate": train_sr,
                "eval/val_success_rate": val_sr,
                "eval/train_mean_length": train_len,
                "eval/val_mean_length": val_len,
                "eval/train_completed_fraction": train_done,
                "eval/val_completed_fraction": val_done,
            }
            for name, value in eval_metrics.items():
                writer.add_scalar(name, value, total_steps)

            metadata = {
                "env_id": config.env_id,
                "steps": total_steps,
                "block": block,
                "elapsed_seconds": elapsed,
                "train_success_rate": train_sr,
                "val_success_rate": val_sr,
                "rollout_success_rate": rollout_sr,
                "steps_per_second": sps,
            }
            save_checkpoint(checkpoint_dir / "latest", train_state, metadata)
            if val_sr > best_val:
                best_val = val_sr
                save_checkpoint(checkpoint_dir / "best_val", train_state, metadata)
            for threshold in (5, 20, 50, 80, 95):
                if threshold not in threshold_saved and train_sr >= threshold / 100:
                    save_checkpoint(checkpoint_dir / f"train_sr_{threshold:02d}", train_state, metadata)
                    threshold_saved.add(threshold)
            print(json.dumps({"event": "eval", **metadata}, sort_keys=True), flush=True)
        else:
            print(json.dumps(
                {
                    "event": "train",
                    "steps": total_steps,
                    "block": block,
                    "elapsed_seconds": elapsed,
                    "rollout_success_rate": rollout_sr,
                    "steps_per_second": sps,
                },
                sort_keys=True,
            ), flush=True)
        writer.flush()
        if elapsed >= config.hours * 3600:
            break

    final_metadata = {
        "env_id": config.env_id,
        "steps": total_steps,
        "block": block,
        "elapsed_seconds": time.time() - start_time,
        "stop_requested": stop_requested,
        "best_val_success_rate": best_val,
    }
    save_checkpoint(checkpoint_dir / "final", train_state, final_metadata)
    (run_dir / "finished.json").write_text(json.dumps(final_metadata, indent=2, sort_keys=True))
    writer.close()
    print(json.dumps({"event": "finished", **final_metadata}, sort_keys=True), flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True, choices=SUPPORTED_ENVS)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--total-timesteps", type=int, default=200_000_000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--updates-per-block", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every-blocks", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    args = parser.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    train(parse_args())
