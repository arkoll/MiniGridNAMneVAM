# Five-object XLand-MiniGrid crafting environment

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
- `register_five_object_env.patch`: adds the environment to the XLand registry.
- `scripts/check_five_object_crafting.py`: split, reset, JIT, crafting-chain,
  and locked-door checks.
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
