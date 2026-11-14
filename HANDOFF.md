# Rog-O-Matic RL Handoff

## Workspace Layout
- `/Users/markv/rogue/rogue5.4/` — Cloned from `RoguelikeRestorationProject/rogue5.4` and compiled on macOS (`arm64`) with:
  ```sh
  ./configure
  make CFLAGS="-g -O2 -std=gnu89 -DNCURSES_OPAQUE=0"
  ```
  - Binary: `/Users/markv/rogue/rogue5.4/rogue`
- `/Users/markv/rogue/rogomatic-rl/` — Python RL environment, recurrent PPO trainer, and tests.
  - Virtualenv: `/Users/markv/rogue/rogomatic-rl/.venv` (`uv`, Python 3.12.13, `torch 2.15.0`, `gymnasium 1.4.0`, `pyte 0.8.2`, `numpy 2.5.4`, `pytest 9.2.0`).

## Implemented Components (`rogomatic-rl/`)
- [pty_client.py](file:///Users/markv/rogue/rogomatic-rl/rogomatic_rl/pty_client.py) — `RoguePTYClient`: spawns `rogue5.4/rogue` in a 24×80 PTY (`TERM=vt100`, `ROGUEOPTS=terse,name=rogomatic,jump`), feeds `pyte`, auto-dismisses `--More--` prompts, and uses `step_timeout=0.02`, `idle_timeout=0.001`.
- [parser.py](file:///Users/markv/rogue/rogomatic-rl/rogomatic_rl/parser.py) — parses row 23 status bar (`RogueStatus`), vectorized `screen_to_array` `(24, 80)`, `extract_local_crop` `(11, 11)` centered on `@`, `find_player`, and `is_death_or_end_screen`.
- [env.py](file:///Users/markv/rogue/rogomatic-rl/rogomatic_rl/env.py) — `RogueEnv` (registered as `Rogue54-v0` in [__init__.py](file:///Users/markv/rogue/rogomatic-rl/rogomatic_rl/__init__.py)):
  - **Observations:** `screen` `(24, 80) uint8`, `local` `(11, 11) uint8`, `stats` `(10,) float32`, `action_mask` `(12,) uint8`.
  - **Actions (12):** `h, j, k, l, y, u, b, n, ., s, >, ,`.
  - **Action masking:** masks out moves into `' '`/`-`/`|` walls or out-of-bounds, diagonal moves blocked by flanking walls/doorways (`diag_ok`), `>` unless `@` is on `%`, and `,` unless an uncollected item (`!?:)]/=,`) remains beneath `@`.
  - **Rewards:** `-0.002`/step, `-0.05` wasted turn (wall bump, `"no way down"`, `"nothing here"`), `+0.02` per newly visited `(r, c)` cell, `+0.02` per newly revealed tile, `+0.05`/gold, `+0.1`/exp, `+50.0`/level, `-10.0` death.
- [model.py](file:///Users/markv/rogue/rogomatic-rl/rogomatic_rl/model.py) — `RogueActorCritic`: global `24×80` CNN (`AdaptiveAvgPool2d((3, 10))`) + local `11×11` `stride=1` CNN + stats MLP (with registered `stats_scale` buffer) + `nn.GRU(256, 256)` + masked actor & critic heads.
- [ppo.py](file:///Users/markv/rogue/rogomatic-rl/rogomatic_rl/ppo.py) — `PPOConfig` & `train_ppo`: recurrent PPO over `AsyncVectorEnv` (`AutoresetMode.SAME_STEP`, value bootstrapping on time-limit truncations) with trajectory-level minibatching, checkpoint saving, and `--resume` / `--resume-from` support.
- Scripts: [run_random.py](file:///Users/markv/rogue/rogomatic-rl/scripts/run_random.py), [train_ppo.py](file:///Users/markv/rogue/rogomatic-rl/scripts/train_ppo.py), [eval_ppo.py](file:///Users/markv/rogue/rogomatic-rl/scripts/eval_ppo.py).
- Tests: [test_env.py](file:///Users/markv/rogue/rogomatic-rl/tests/test_env.py), [test_model.py](file:///Users/markv/rogue/rogomatic-rl/tests/test_model.py).

## Current Status & Verification
- **Tests:** `.venv/bin/pytest -v` passes (**6/6** in `2.95s`).
- **Training throughput:** ~`180–200 sps` with `--num-envs 8` on Apple Silicon.
- **Latest checkpoint:** saved during training at `checkpoints/rogomatic_ppo_latest.pt` (includes `model_state_dict`, `optimizer_state_dict`, `global_step`, `update`, `best_level`, `best_gold`).

## Useful Commands
```sh
cd /Users/markv/rogue/rogomatic-rl

# Run test suite
.venv/bin/pytest -v

# Train PPO (500k steps, 8 parallel Rogue PTYs)
.venv/bin/python scripts/train_ppo.py --total-timesteps 500000 --num-envs 8

# Resume PPO training from latest checkpoint
.venv/bin/python scripts/train_ppo.py --resume --total-timesteps 1000000 --num-envs 8

# Watch trained agent play
.venv/bin/python scripts/eval_ppo.py --render --episodes 1 --max-steps 300
```

## Recommended Next Steps
1. **Add `--resume` support** to [ppo.py](file:///Users/markv/rogue/rogomatic-rl/rogomatic_rl/ppo.py) and [train_ppo.py](file:///Users/markv/rogue/rogomatic-rl/scripts/train_ppo.py) (save `optimizer_state_dict`, `update`, and `best_level`/`best_gold` in checkpoints).
2. **Inspect behavior with `eval_ppo.py --render`:** if the agent stalls on `.` (rest) or `s` (search), add a small penalty when `.` or `s` is used with no adjacent monster/door.
3. **Expand action space later:** add food/potion/scroll handling (`e`, `q`, `r` + item-letter prompt handling) once the agent reliably explores levels 1–3.
