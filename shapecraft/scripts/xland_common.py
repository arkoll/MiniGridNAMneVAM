"""Общий модуль эксперимента «правило зависит от формы, а не от цвета».

Цепочка правил одна и та же на всех уровнях:

    квадрат(c) --шаг вперёд рядом--> круг(c) --> пирамида(c) --> зелёная цель

Цвет c переносится по цепочке, а последнее звено схлопывает его в фиксированную
зелёную цель, чтобы цель эпизода можно было записать одной плиткой.

Правило в этой среде сравнивает плитку целиком, вместе с цветом, поэтому
«квадрат любого цвета» набирается перечислением: по одному правилу на каждый цвет.
Правила заводятся сразу на ВСЕ цвета, включая отложенные, — среда обязана вести себя
одинаково. Новизна создаётся только составом `init_tiles`: какие цвета реально
выкладываются на поле.

ВАЖНО про порядок правил. `check_rule` прогоняет весь список по очереди внутри
одного шага, передавая обновлённую сетку дальше. Если положить правила в порядке
цепочки, квадрат за один шаг станет целью. Поэтому правила последнего звена лежат
в массиве первыми.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from xminigrid.benchmarks import Benchmark
from xminigrid.core.constants import NUM_COLORS, NUM_TILES, TILES_REGISTRY, Colors, Tiles
from xminigrid.core.goals import AgentOnTileGoal
from xminigrid.core.rules import AgentNearRule
from xminigrid.registration import make, register, registered_environments
from xminigrid.rendering.rgb_render import render_tile
from xminigrid.wrappers import Wrapper

# ------------------------------------------------------------------ палитра ---

# GREY занят стенами, BLACK — полом, GREEN зарезервирован под финальную цель.
# RED занят АГЕНТОМ: он рисуется красным треугольником, а пирамида — тоже треугольник,
# то есть красная пирамида и агент, смотрящий вверх, в кадре почти неразличимы
# (увидели это на первой же валидационной картинке). Убрав красный из палитры объектов,
# делаем агента единственным красным пятном в кадре.
TRAIN_COLORS = (Colors.WHITE, Colors.PURPLE, Colors.YELLOW, Colors.BROWN, Colors.PINK)
HELD_OUT_COLORS = (Colors.BLUE, Colors.ORANGE)
ALL_COLORS = TRAIN_COLORS + HELD_OUT_COLORS

COLOR_NAMES = {
    Colors.RED: "red",
    Colors.GREEN: "green",
    Colors.BLUE: "blue",
    Colors.PURPLE: "purple",
    Colors.YELLOW: "yellow",
    Colors.GREY: "grey",
    Colors.BLACK: "black",
    Colors.ORANGE: "orange",
    Colors.WHITE: "white",
    Colors.BROWN: "brown",
    Colors.PINK: "pink",
}
TILE_NAMES = {
    Tiles.FLOOR: "floor",
    Tiles.WALL: "wall",
    Tiles.BALL: "ball",
    Tiles.SQUARE: "square",
    Tiles.PYRAMID: "pyramid",
    Tiles.GOAL: "goal",
    Tiles.KEY: "key",
    Tiles.HEX: "hex",
    Tiles.STAR: "star",
}

# цепочка форм: с чего начинаем и во что превращается на каждом звене
CHAIN = ((Tiles.SQUARE, Tiles.BALL), (Tiles.BALL, Tiles.PYRAMID))
TERMINAL_FROM = Tiles.PYRAMID
TERMINAL_TILE = (Tiles.GOAL, Colors.GREEN)
CHAIN_DEPTH = 3  # квадрат->круг, круг->пирамида, пирамида->цель

# формы, которых нет ни в одном правиле: модель должна выучить, что они не меняются
INERT_SHAPES = (Tiles.HEX, Tiles.STAR)

# среда: одна комната 8x8, кадр ровно 8*32 = 256 пикселей
ENV_ID = "XLand-MiniGrid-R1-8x8"
GRID_SIZE = 8
MAX_STEPS = 96
DATASET_TILE_SIZE = 32  # 8*32 = 256, кадры датасета WAM
POLICY_TILE_SIZE = 16  # 8*16 = 128, вход политики PPO

# pick_up и put_down не используются: без них мертвы все правила семейства TileNear,
# а нам нужны только AgentNear. Номера действий обязаны остаться исходными.
ACTIONS = (0, 1, 2)  # вперёд, поворот по часовой, против часовой
ACTION_NAMES = ("forward", "turn_cw", "turn_ccw")

# непересекающиеся диапазоны сидов раскладки (METHODS.md §7)
SEED_RANGES = {
    "ppo": 0,
    "train": 10_000_000,
    "val_seen": 20_000_000,
    "val_ood_chain": 30_000_000,
    "val_ood_all": 40_000_000,
}


# ------------------------------------------------------------- наборы правил ---


def build_rules(colors=ALL_COLORS) -> jnp.ndarray:
    """Правила цепочки для всех цветов, в порядке от последнего звена к первому."""
    rules = []
    # последнее звено: пирамида любого цвета -> зелёная цель
    for c in colors:
        rules.append(
            AgentNearRule(
                tile=TILES_REGISTRY[TERMINAL_FROM, c],
                prod_tile=TILES_REGISTRY[TERMINAL_TILE[0], TERMINAL_TILE[1]],
            )
        )
    # средние звенья, в обратном порядке
    for src, dst in reversed(CHAIN):
        for c in colors:
            rules.append(AgentNearRule(tile=TILES_REGISTRY[src, c], prod_tile=TILES_REGISTRY[dst, c]))
    return jnp.stack([r.encode() for r in rules])


def build_goal() -> jnp.ndarray:
    """Цель: встать на зелёную клетку-цель, полученную в конце цепочки."""
    return AgentOnTileGoal(tile=TILES_REGISTRY[TERMINAL_TILE[0], TERMINAL_TILE[1]]).encode()


def build_benchmark(chain_colors, inert_colors) -> Benchmark:
    """Набор задач: правила и цель одинаковые, различаются только выкладываемые объекты.

    chain_colors — цвета квадрата, с которого начинается цепочка;
    inert_colors — цвета двух инертных объектов.
    """
    rules = build_rules()
    goal = build_goal()

    init_tiles = []
    inert_variants = list(itertools.product(INERT_SHAPES, inert_colors))
    for cc in chain_colors:
        for (s1, c1), (s2, c2) in itertools.product(inert_variants, inert_variants):
            init_tiles.append(
                jnp.stack(
                    [
                        TILES_REGISTRY[Tiles.SQUARE, cc],
                        TILES_REGISTRY[s1, c1],
                        TILES_REGISTRY[s2, c2],
                    ]
                )
            )
    n = len(init_tiles)
    return Benchmark(
        goals=jnp.stack([goal] * n),
        rules=jnp.stack([rules] * n),
        init_tiles=jnp.stack(init_tiles),
        num_rules=jnp.full((n,), rules.shape[0], dtype=jnp.int32),
    )


def all_benchmarks() -> dict[str, Benchmark]:
    """Пять наборов: один для обучения политики и четыре для сбора данных."""
    return {
        # политика учится на всех цветах сразу — она только сборщик данных,
        # и должна быть одинаково компетентна на трейне и на валидации
        "ppo": build_benchmark(ALL_COLORS, ALL_COLORS),
        # обучающий пул WAM: только цвета трейна
        "train": build_benchmark(TRAIN_COLORS, TRAIN_COLORS),
        # контроль: те же цвета, новые раскладки — обычное обобщение
        "val_seen": build_benchmark(TRAIN_COLORS, TRAIN_COLORS),
        # главный тест: новый цвет у реагирующего объекта, инертные — знакомые
        "val_ood_chain": build_benchmark(HELD_OUT_COLORS, TRAIN_COLORS),
        # тест пожёстче: знакомых цветов на поле нет вообще
        "val_ood_all": build_benchmark(HELD_OUT_COLORS, HELD_OUT_COLORS),
    }


# ------------------------------------------------------------------ рендер ---


def _build_tile_caches(tile_size: int):
    """Таблица отрисованных плиток: пустая клетка и та же клетка с агентом всех четырёх направлений.

    Это тот же самый `render_tile` из библиотеки, что и в `env.render`, но без
    подсветки области видимости — она нам не нужна и меняла бы пиксели по причине,
    не связанной с динамикой.
    """
    n = NUM_TILES * NUM_COLORS
    base = np.zeros((n, tile_size, tile_size, 3), dtype=np.uint8)
    with_agent = np.zeros((4, n, tile_size, tile_size, 3), dtype=np.uint8)
    for t in range(NUM_TILES):
        for c in range(NUM_COLORS):
            idx = t * NUM_COLORS + c
            base[idx] = render_tile(tile=(t, c), agent_direction=None, highlight=False, tile_size=tile_size)
            for d in range(4):
                with_agent[d, idx] = render_tile(tile=(t, c), agent_direction=d, highlight=False, tile_size=tile_size)
    return base, with_agent


_CACHES: dict[int, tuple] = {}


def get_caches(tile_size: int):
    if tile_size not in _CACHES:
        base, with_agent = _build_tile_caches(tile_size)
        _CACHES[tile_size] = (jnp.asarray(base), jnp.asarray(with_agent))
    return _CACHES[tile_size]


def make_render_fn(tile_size: int):
    """jit-совместимый рендер ВСЕГО поля сверху. Вид от агента не используется нигде."""
    base, with_agent = get_caches(tile_size)

    def render(grid, agent):
        idx = grid[:, :, 0].astype(jnp.int32) * NUM_COLORS + grid[:, :, 1].astype(jnp.int32)
        img = base[idx]  # (H, W, ts, ts, 3)
        y, x = agent.position[0], agent.position[1]
        img = img.at[y, x].set(with_agent[agent.direction, idx[y, x]])
        h, w = grid.shape[0], grid.shape[1]
        return img.transpose(0, 2, 1, 3, 4).reshape(h * tile_size, w * tile_size, 3)

    return render


def render_reference(grid, agent, tile_size: int) -> np.ndarray:
    """Эталонный рендер на numpy: собирается прямой сборкой плиток, без наших таблиц.

    Нужен только как ворота: наш быстрый рендер обязан совпадать с ним побайтово.
    """
    grid = np.asarray(grid)
    h, w = grid.shape[0], grid.shape[1]
    img = np.zeros((h * tile_size, w * tile_size, 3), dtype=np.uint8)
    pos = np.asarray(agent.position)
    direction = int(agent.direction)
    for y in range(h):
        for x in range(w):
            d = direction if (y == pos[0] and x == pos[1]) else None
            tile_img = render_tile(
                tile=(int(grid[y, x, 0]), int(grid[y, x, 1])),
                agent_direction=d,
                highlight=False,
                tile_size=tile_size,
            )
            img[y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size] = tile_img
    return img


class FullFieldRGBWrapper(Wrapper):
    """Наблюдение — картинка всего поля сверху вместо штатной вырезки вокруг агента."""

    def __init__(self, env, tile_size: int):
        super().__init__(env)
        self._tile_size = tile_size
        self._render = make_render_fn(tile_size)

    def observation_shape(self, params):
        return (params.height * self._tile_size, params.width * self._tile_size, 3)

    def _swap(self, timestep):
        return timestep.replace(observation=self._render(timestep.state.grid, timestep.state.agent))

    def reset(self, params, key):
        return self._swap(self._env.reset(params, key))

    def step(self, params, timestep, action):
        return self._swap(self._env.step(params, timestep, action))


# --------------------------------------------------------------------- среда ---

_REGISTERED = False


def make_env(tile_size: int, auto_reset: bool = False, rgb_obs: bool = True):
    """Среда проекта: одна комната 8x8, кадр — всё поле, действий три.

    rgb_obs=False нужен сбору данных: он рендерит кадры сам и только для тех шагов,
    которые реально попадут в датасет, поэтому штатный рендер наблюдения был бы
    чистой потерей времени.
    """
    global _REGISTERED
    if not _REGISTERED and ENV_ID not in registered_environments():
        register(
            id=ENV_ID,
            entry_point="xminigrid.envs.xland:XLandMiniGrid",
            grid_type="R1",
            height=GRID_SIZE,
            width=GRID_SIZE,
        )
        _REGISTERED = True

    env, env_params = make(ENV_ID)
    env_params = env_params.replace(max_steps=MAX_STEPS)
    if auto_reset:
        from xminigrid.wrappers import GymAutoResetWrapper

        env = GymAutoResetWrapper(env)
    if rgb_obs:
        env = FullFieldRGBWrapper(env, tile_size=tile_size)
    return env, env_params


def episode_seed(kind: str, index) -> jax.Array:
    """Ключ раскладки из зарезервированного диапазона: сиды наборов не пересекаются."""
    return jax.random.key(SEED_RANGES[kind] + index)
