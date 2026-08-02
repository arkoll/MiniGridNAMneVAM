"""PPO-сборщик траекторий для эксперимента «форма против цвета».

Политика — только источник данных для world-модели, поэтому она обучается на ВСЕХ цветах
(и обучающих, и отложенных): нужно, чтобы качество траекторий на трейне и на валидации
не различалось, иначе к разнице по цвету примешается разница по компетентности.

Вход политики — картинка ВСЕГО поля 128x128 (вид от агента не используется нигде).
Сеть без памяти: поле видно целиком, состояние цепочки читается по форме объекта,
то есть задача марковская.

Устройство одно, схема как в PureJaxRL: весь цикл под jit, среды векторизованы через vmap.

Важное свойство нашей постановки: все наборы задач имеют ОДИНАКОВЫЕ правила и цель,
различаются только выкладываемыми объектами (`init_tiles`). Правила и цель копируются
в состояние при сбросе, поэтому пересэмплировать наборы можно в любой момент — на
текущий эпизод это не влияет.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from functools import partial

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from flax.training.train_state import TrainState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xland_common as xc


class ActorCritic(nn.Module):
    """Кодировщик кадра без свёрток: кадр режется на плитки, каждая вкладывается линейно.

    Почему не свёртки: на этом сервере стоит cuDNN 9.24, который не поддерживает Volta
    (V100, sm_70) — любая свёртка падает с CUDNN_STATUS 5003, даже на батче 256.
    Матричные умножения при этом работают штатно (14.8 TFLOPS fp32).

    Заодно это честнее по существу: кадр здесь — мозаика из плиток известного размера,
    границы плиток известны точно, и разбиение по ним ничего не размывает.
    """

    num_actions: int
    patch: int = xc.POLICY_TILE_SIZE
    emb_dim: int = 64
    hidden: int = 256

    @nn.compact
    def __call__(self, obs_uint8):
        x = obs_uint8.astype(jnp.float32) / 255.0
        b, h, w, c = x.shape
        gh, gw = h // self.patch, w // self.patch
        x = x.reshape(b, gh, self.patch, gw, self.patch, c)
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape(b, gh * gw, self.patch * self.patch * c)
        x = nn.relu(nn.Dense(self.emb_dim)(x))
        x = x.reshape(b, -1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        logits = nn.Dense(self.num_actions)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        return logits, value


def sample_rulesets(benchmark, key, n):
    ids = jax.random.randint(key, shape=(n,), minval=0, maxval=benchmark.num_rulesets())
    return jax.vmap(benchmark.get_ruleset)(ids)


def make_train(env, base_params, benchmark, cfg):
    num_updates = cfg.total_timesteps // (cfg.num_envs * cfg.num_steps)
    net = ActorCritic(num_actions=len(xc.ACTIONS))

    def linear_schedule(count):
        frac = 1.0 - count / (num_updates * cfg.update_epochs * cfg.num_minibatches)
        return cfg.lr * frac

    def init_state(key):
        key, k_net = jax.random.split(key)
        h, w = base_params.height * xc.POLICY_TILE_SIZE, base_params.width * xc.POLICY_TILE_SIZE
        dummy = jnp.zeros((1, h, w, 3), dtype=jnp.uint8)
        params = net.init(k_net, dummy)
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        return TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    def rollout_and_update(carry, _):
        rng, train_state, params, timestep = carry

        def env_step(carry, _):
            rng, train_state, params, timestep = carry
            rng, k_act = jax.random.split(rng)
            logits, value = net.apply(train_state.params, timestep.observation)
            dist = distrax.Categorical(logits=logits)
            action, log_prob = dist.sample_and_log_prob(seed=k_act)
            new_timestep = jax.vmap(env.step)(params, timestep, action)
            transition = dict(
                obs=timestep.observation,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=new_timestep.reward,
                # эпизод обрывается и по цели, и по лимиту шагов: и то и другое — конец эпизода
                done=new_timestep.last(),
            )
            return (rng, train_state, params, new_timestep), transition

        (rng, train_state, params, timestep), traj = jax.lax.scan(
            env_step, (rng, train_state, params, timestep), None, cfg.num_steps
        )

        _, last_value = net.apply(train_state.params, timestep.observation)

        def gae_step(carry, x):
            gae, next_value = carry
            reward, value, done = x["reward"], x["value"], x["done"]
            delta = reward + cfg.gamma * next_value * (1 - done) - value
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            gae_step, (jnp.zeros_like(last_value), last_value), traj, reverse=True
        )
        targets = advantages + traj["value"]

        def epoch(carry, _):
            rng, train_state = carry
            rng, k_perm = jax.random.split(rng)
            batch = dict(traj, advantages=advantages, targets=targets)
            batch = jtu.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), batch)
            perm = jax.random.permutation(k_perm, cfg.num_envs * cfg.num_steps)
            batch = jtu.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
            batch = jtu.tree_map(lambda x: x.reshape((cfg.num_minibatches, -1) + x.shape[1:]), batch)

            def minibatch(train_state, mb):
                def loss_fn(p):
                    logits, value = net.apply(p, mb["obs"])
                    dist = distrax.Categorical(logits=logits)
                    log_prob = dist.log_prob(mb["action"])
                    adv = (mb["advantages"] - mb["advantages"].mean()) / (mb["advantages"].std() + 1e-8)
                    ratio = jnp.exp(log_prob - mb["log_prob"])
                    a_loss = -jnp.minimum(
                        adv * ratio, adv * jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                    ).mean()
                    v_clipped = mb["value"] + (value - mb["value"]).clip(-cfg.clip_eps, cfg.clip_eps)
                    v_loss = 0.5 * jnp.maximum(
                        jnp.square(value - mb["targets"]), jnp.square(v_clipped - mb["targets"])
                    ).mean()
                    entropy = dist.entropy().mean()
                    return a_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy, (a_loss, v_loss, entropy)

                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
                return train_state.apply_gradients(grads=grads), (loss,) + aux

            train_state, info = jax.lax.scan(minibatch, train_state, batch)
            return (rng, train_state), info

        (rng, train_state), info = jax.lax.scan(epoch, (rng, train_state), None, cfg.update_epochs)

        # новые наборы задач на следующий отрезок: правила и цель везде одинаковые,
        # поэтому подмена по ходу эпизода безопасна и влияет только на будущие сбросы
        rng, k_rs = jax.random.split(rng)
        params = params.replace(ruleset=sample_rulesets(benchmark, k_rs, cfg.num_envs))

        stats = dict(
            loss=info[0].mean(),
            actor_loss=info[1].mean(),
            value_loss=info[2].mean(),
            entropy=info[3].mean(),
            # доля шагов с положительной наградой — прокси успеха внутри отрезка
            reward_rate=(traj["reward"] > 0).mean(),
            mean_reward_on_success=jnp.where(
                (traj["reward"] > 0).sum() > 0, traj["reward"].sum() / jnp.maximum((traj["reward"] > 0).sum(), 1), 0.0
            ),
            episodes=traj["done"].sum(),
            success_rate=jnp.where(
                traj["done"].sum() > 0, (traj["reward"] > 0).sum() / jnp.maximum(traj["done"].sum(), 1), 0.0
            ),
        )
        return (rng, train_state, params, timestep), stats

    def train(key):
        key, k_init, k_rs, k_reset = jax.random.split(key, 4)
        train_state = init_state(k_init)
        params = base_params.replace(ruleset=sample_rulesets(benchmark, k_rs, cfg.num_envs))
        reset_keys = jax.random.split(k_reset, cfg.num_envs)
        timestep = jax.vmap(env.reset)(params, reset_keys)
        carry = (key, train_state, params, timestep)
        carry, stats = jax.lax.scan(rollout_and_update, carry, None, num_updates)
        return carry[1], stats

    return train, num_updates


def evaluate(env, base_params, benchmark, net_params, key, num_envs, max_steps, greedy=True, temperature=1.0):
    """Прогон без обучения: доля побед, средняя награда, средняя длина эпизода."""
    net = ActorCritic(num_actions=len(xc.ACTIONS))
    key, k_rs, k_reset = jax.random.split(key, 3)
    params = base_params.replace(ruleset=sample_rulesets(benchmark, k_rs, num_envs))
    timestep = jax.vmap(env.reset)(params, jax.random.split(k_reset, num_envs))

    def body(carry, _):
        rng, timestep, done, reward, length = carry
        rng, k = jax.random.split(rng)
        logits, _ = net.apply(net_params, timestep.observation)
        action = jnp.argmax(logits, -1) if greedy else distrax.Categorical(logits=logits / temperature).sample(seed=k)
        new_timestep = jax.vmap(env.step)(params, timestep, action)
        just_done = new_timestep.last() & (~done)
        reward = reward + jnp.where(just_done, new_timestep.reward, 0.0)
        length = length + jnp.where(done, 0, 1)
        done = done | new_timestep.last()
        return (rng, new_timestep, done, reward, length), None

    init = (key, timestep, jnp.zeros(num_envs, bool), jnp.zeros(num_envs), jnp.zeros(num_envs, jnp.int32))
    (_, _, done, reward, length), _ = jax.lax.scan(body, init, None, max_steps)
    return dict(
        success_rate=float((reward > 0).mean()),
        mean_reward=float(reward.mean()),
        mean_length=float(length.mean()),
        finished=float(done.mean()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=30_000_000)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--num-minibatches", type=int, default=16)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="/home/user8_2/AIRI_WAM/runs/xland-ppo")
    p.add_argument("--smoke", action="store_true")
    cfg = p.parse_args()

    if cfg.smoke:
        cfg.total_timesteps = cfg.num_envs * cfg.num_steps * 3

    os.makedirs(cfg.out, exist_ok=True)
    env, base_params = xc.make_env(tile_size=xc.POLICY_TILE_SIZE, auto_reset=True)
    benchmarks = xc.all_benchmarks()

    print(f"поле {base_params.height}x{base_params.width}, кадр политики {env.observation_shape(base_params)}")
    print(f"действий: {len(xc.ACTIONS)} {xc.ACTION_NAMES}, max_steps={base_params.max_steps}")
    print(f"наборов для обучения политики: {benchmarks['ppo'].num_rulesets()}, правил {int(benchmarks['ppo'].num_rules[0])}")

    train_fn, num_updates = make_train(env, base_params, benchmarks["ppo"], cfg)
    print(f"апдейтов: {num_updates}, батч {cfg.num_envs * cfg.num_steps}")

    t0 = time.time()
    print("компиляция...", flush=True)
    compiled = jax.jit(train_fn).lower(jax.random.key(cfg.seed)).compile()
    print(f"скомпилировано за {time.time() - t0:.1f} с", flush=True)

    t0 = time.time()
    train_state, stats = jax.block_until_ready(compiled(jax.random.key(cfg.seed)))
    dt = time.time() - t0
    sps = cfg.total_timesteps / dt
    print(f"обучение: {dt:.1f} с, {sps / 1e3:.1f} тыс. шагов/с", flush=True)

    stats = jtu.tree_map(lambda x: np.asarray(x).tolist(), stats)
    with open(os.path.join(cfg.out, "train_curve.jsonl"), "w") as f:
        for i in range(num_updates):
            row = {k: v[i] for k, v in stats.items()}
            row["update"] = i
            row["transitions"] = (i + 1) * cfg.num_envs * cfg.num_steps
            f.write(json.dumps(row) + "\n")

    with open(os.path.join(cfg.out, "policy.pkl"), "wb") as f:
        pickle.dump(jax.device_get(train_state.params), f)

    print("\nоценка политики на каждом наборе (жадно и с температурой):")
    report = {"steps_per_second": sps, "train_seconds": dt, "config": vars(cfg)}
    for name in ("ppo", "train", "val_seen", "val_ood_chain", "val_ood_all"):
        g = evaluate(env, base_params, benchmarks[name], train_state.params, jax.random.key(7), 512, base_params.max_steps)
        t = evaluate(
            env, base_params, benchmarks[name], train_state.params, jax.random.key(7), 512,
            base_params.max_steps, greedy=False, temperature=1.0,
        )
        report[name] = {"greedy": g, "sampled": t}
        print(
            f"  {name:>14}: жадно success={g['success_rate']:.3f} len={g['mean_length']:.1f} | "
            f"с сэмплированием success={t['success_rate']:.3f} len={t['mean_length']:.1f}"
        )

    with open(os.path.join(cfg.out, "ppo_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nPPO_STATUS=OK")


if __name__ == "__main__":
    main()
