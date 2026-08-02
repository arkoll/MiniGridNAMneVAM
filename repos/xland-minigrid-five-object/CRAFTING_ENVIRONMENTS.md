# Easy and Medium crafting environments for XLand-MiniGrid

This bundle contains difficulty-qualified ExactCraft, ShapeCraft, and
four-embodiment ExactCraft families:

| Family | Easy canonical ID | Medium canonical ID |
|---|---|---|
| ExactCraft | `XLand-MiniGrid-ExactCraft-Easy-8x8-v1` | `XLand-MiniGrid-ExactCraft-Medium-8x8-v1` |
| ShapeCraft | `XLand-MiniGrid-ShapeCraft-Easy-8x8-v1` | `XLand-MiniGrid-ShapeCraft-Medium-8x8-v1` |
| Embodied | `XLand-MiniGrid-EmbodiedExactCraft-{body}-Easy-8x8-v1` | `XLand-MiniGrid-EmbodiedExactCraft-{body}-Medium-8x8-v1` |

Here `{body}` is `Standard`, `Stride2`, `Omni8`, or `Crab`. All former
difficulty-unqualified IDs remain Medium aliases, including
`XLand-MiniGrid-FiveObjectCrafting-8x8`, so existing scripts and checkpoints
continue to work.

The explicit aliases
`XLand-MiniGrid-EmbodiedExactCraft-Stride2-Adaptive-{Easy,Medium}-8x8-v1`
refer to the same progressive fallback locomotion as the shorter `Stride2`
IDs. Prefer the explicit name in new experiment manifests.

## Difficulty definition

Easy and Medium intentionally share the same 8×8 layout, observation,
interaction mechanics, door/key/star objective, object identities, and
primitive-rule semantics. Difficulty changes only the size and depth of the
crafting dynamics:

| Property | Easy | Medium |
|---|---:|---:|
| Primitive rules | 6 | 9 |
| Active rules per episode | 2 | 3 |
| Crafting depth | 2 | 3 |
| Ordinary object roles | 4 | 5 |
| Initial ingredient instances | 3 | 4 |
| Abstract trees | 9 | 27 |
| Recommended horizon | 192 | 256 |

ExactCraft Easy has six train and three validation trees. Every primitive rule
appears in both splits; validation holds out only unseen two-rule
compositions. ShapeCraft Easy keeps all nine abstract trees in both splits and
uses the same strict complementary color-binding split as Medium: 36 palettes
per split and 324 tasks per split.

## ExactCraft Medium

This patch adds a deliberately small world-model benchmark:

- ordinary objects: `A`, `B`, `C`, `D`, `E`;
- special objects: blue key, blue locked door, yellow star;
- nine non-directional `TileNear` rules;
- 27 three-stage crafting trees;
- split: 18 train / 9 validation;
- every primitive rule `R1`–`R9` occurs in both splits;
- validation holds out only complete three-rule combinations.

The ordinary-object palette uses five deliberately contrasting colors:
`A=red`, `B=green`, `C=white`, `D=purple`, and `E=orange`. Blue and yellow are
reserved for the key/door mechanism and the star goal.

## Environment

The agent and four ingredient instances spawn in one of two rooms. The yellow
star always spawns in the other room. A blue locked door separates the rooms,
and there is no initial key. The active task's three rules must be composed to
create the blue key. Success is `AgentNearGoal(yellow star)`.

The map is 8×8. Layout mirroring is random, but the divider moves with the
mirror: the agent and four ingredients always receive the larger three-column
room, while the star receives the smaller two-column room. Observations use
fixed world coordinates and expose the entire map at every step—there is no
egocentric crop or camera rotation. The symbolic `img` observation is 8×8×3:
tile ID, color ID, and an agent layer. Rendering also shows the complete map
without field-of-view shading.

Picked-up objects remain visible in RGB frames. The carried-object sprite is
drawn at the triangle tip inside the agent cell, follows movement, and rotates
with agent direction. This avoids hidden visual state without changing XLand's
safe interaction rules: walls and objects block movement, `put_down` rejects
occupied targets, and an object can only be released onto a free adjacent floor
cell. Pocket tile/color IDs are also included explicitly in the symbolic
observation.

Only the selected three rules are active in an episode. They are encoded in
reverse topological order so that XLand's sequential rule scan cannot execute
multiple crafting stages after a single `put_down`.

## Files

- `src/xminigrid/envs/five_object_crafting.py`: task family and environment.
- `src/xminigrid/envs/easy_crafting.py`: depth-2 ExactCraft Easy task bank.
- `src/xminigrid/envs/shape_crafting.py`: color-invariant ShapeCraft family.
- `src/xminigrid/envs/shape_crafting_easy.py`: depth-2 ShapeCraft Easy family.
- `src/xminigrid/envs/embodied_crafting.py`: four embodiment action kernels,
  sprites, rendering, and factory.
- `src/xminigrid/core/rules.py`: adds `ShapeTileNearRule` (encoding ID 12).
- `src/xminigrid/core/constants.py` and rendering files: add the fifth unique
  ordinary shape, `DIAMOND`.
- `register_five_object_env.patch`: adds the environment to the XLand registry.
- `scripts/check_five_object_crafting.py`: split, reset, JIT, crafting-chain,
  and locked-door checks.
- `scripts/check_shape_crafting.py`: strict split, shape-only dynamics,
  registry, reset, rendering, and JIT checks.
- `scripts/check_embodied_crafting.py`: identical-layout, transition,
  connectivity, collision, interaction, registry, rendering, and JIT checks.
- `scripts/check_easy_levels.py`: Easy splits, chains, color OOD, embodiments,
  canonical IDs, JIT, and Medium compatibility checks.
- `scripts/render_five_object_samples.py`: labelled 4×3 sample grid.
- `scripts/render_carry_mechanics.py`: action-by-action pickup/carry/drop grid.
- `scripts/render_scripted_episode.py`: complete successful T02 storyboard and
  machine-readable action manifest.

## Usage

Copy the module and scripts into a checkout of XLand-MiniGrid at commit
`0d306d1`, then apply `register_five_object_env.patch`.

```python
import jax

from xminigrid.envs.five_object_crafting import make_env_and_params

env, params, task = make_env_and_params(task_id=2)
timestep = env.reset(params, jax.random.PRNGKey(0))
rgb = env.render(params, timestep)
```

Easy ExactCraft:

```python
from xminigrid.envs.easy_crafting import make_easy_env_and_params

env, params, task = make_easy_env_and_params(task_id=2)
```

## EmbodiedExactCraft v1

All four Medium variants use the same 27 ExactCraft tasks and 18/9
train-validation split. All four Easy variants use the same nine ExactCraft
Easy tasks and 6/3 split. Both difficulties share the 8×8 two-room generator,
object positions for a given seed, blue-door mechanism, full-map observation,
visible carried object, and six discrete actions. Only the agent sprite and
locomotion dynamics change between bodies.

| Embodiment | Locomotion action (`action=0`) | Turns |
|---|---|---|
| Standard | one cardinal cell forward | 90 degrees |
| Stride2-Adaptive | up to two forward microsteps | 90 degrees |
| Omni8 | one cardinal or diagonal cell | 45 degrees |
| Crab | one cell to the right of the current heading | 90 degrees |

`Stride2-Adaptive` executes its two microsteps progressively. If both are
walkable it moves two cells; if only the first is walkable it moves one; if the
first is blocked it does not move. It cannot jump through a locked door or
wall. This fallback also removes the fixed-parity reachability problem.

`Omni8` has eight headings and forbids diagonal corner cutting. `Crab` changes
position sideways while keeping its heading unchanged. All variants retain the
same forward-facing `pickup`, `putdown`, and `toggle` semantics; Omni8 can
interact in eight directions, while cardinal orientation remains available for
ordinary TileNear crafting.

```python
import jax

from xminigrid.envs.embodied_crafting import make_embodied_env_and_params

env, params, task = make_embodied_env_and_params(
    "stride2",
    task_id=2,
    difficulty="easy",
)
timestep = env.reset(params, jax.random.PRNGKey(19))
timestep = env.step(params, timestep, 0)
```

## ShapeCraft Medium

ShapeCraft keeps the same two-room layout, visible carried object, fixed
full-map observation, blue locked door, and yellow-star goal. Its five ordinary
roles have unique shapes:

| role | shape |
|---|---|
| A | ball |
| B | square |
| C | pyramid |
| D | hexagon |
| E | diamond |

The shared abstract dynamics are:

```text
R1 ball + square  -> hexagon
R2 ball + pyramid -> hexagon
R3 square + pyramid -> hexagon
R4 hexagon + ball    -> diamond
R5 hexagon + square  -> diamond
R6 hexagon + pyramid -> diamond
R7 diamond + ball    -> blue key
R8 diamond + square  -> blue key
R9 diamond + pyramid -> blue key
```

All 27 choices of one rule from each stage are present in both train and
validation. Colors come from the same six-color vocabulary in both splits:
red, green, orange, purple, pink, and white. For each shape, train receives
three colors and validation receives the complementary three. Therefore:

- there are no validation-only colors;
- all abstract rules and all trees occur in both splits;
- no concrete `(shape, color)` binding—and thus no
  `(rule, shape, color)` binding—occurs in both splits;
- each scene uses five distinct colors;
- each split has 27 trees × 36 balanced palettes = 972 tasks.

```python
import jax

from xminigrid.envs.shape_crafting import make_shape_env_and_params

env, params, task = make_shape_env_and_params("train", task_index=1)
timestep = env.reset(params, jax.random.PRNGKey(0))
rgb = env.render(params, timestep)
```

Easy ShapeCraft:

```python
from xminigrid.envs.shape_crafting_easy import (
    make_shape_easy_env_and_params,
)

env, params, task = make_shape_easy_env_and_params(
    "train",
    task_index=1,
)
```

## Privileged RL trajectory-collector policy

`training/train_privileged_crafting_ppo.py` implements fast PPO for the three
Easy environments currently used for trajectory collection. The policy is
intentionally privileged: it receives the full symbolic map, carried object,
active rule and goal encodings, crafting progress, and a ground-truth mask of
valid actions. Task completion is still the native `AgentNearGoal(yellow
star)` signal. Potential-based shaping provides intermediate credit for
creating the hexagon, creating the blue key, and opening the door.

One policy is trained across every train task in its family and evaluated on
both train and held-out validation banks. TensorBoard logs include throughput,
PPO losses, rollout SR, in-distribution train SR, and held-out validation SR.
Checkpoints include `latest`, `best_val`, `final`, and the first policy crossing
train SR 5%, 20%, 50%, 80%, and 95%.

See `RL_TRAINING.md` for the active server runs, GPU assignment, monitoring,
checkpoint layout, resume command, and the exact definition of reported SR.
