# Rog-O-Matic RL Handoff

## 1. Environment & Build Status (Linux `x86_64` + CUDA)
- **Rogue 5.4 Binary:** `/home/markv/rogomatic-rl/rogue5.4/rogue` — Rebuilt from source for `x86_64-unknown-linux-gnu` (`./configure && make CFLAGS="-g -O2 -std=gnu89 -DNCURSES_OPAQUE=0"`).
- **Python RL Package:** `/home/markv/rogomatic-rl/rogomatic-rl/`
  - Virtualenv: `/home/markv/rogomatic-rl/rogomatic-rl/.venv` (`uv`, Python 3.13.5, `torch 2.15.0+cu132`, `gymnasium 1.4.0`, `pyte 0.8.2`, `numpy 2.5.4`, `pytest 9.2.0`).
  - Test suite: `.venv/bin/pytest -v` passes **32/32** tests ([test_env.py](file:///home/markv/rogomatic-rl/rogomatic-rl/tests/test_env.py), [test_model.py](file:///home/markv/rogomatic-rl/rogomatic-rl/tests/test_model.py)).

## 2. Key Improvements Made in This Session
1. **Tombstone & Death Screen Detection Fix:**
   - [parser.py](file:///home/markv/rogomatic-rl/rogomatic-rl/rogomatic_rl/parser.py#L141-L154): Fixed `is_death_or_end_screen()` to match Rogue 5.4's actual uppercase `REST` / `IN` / `PEACE` tombstone, `[Press return to continue]` prompt, `Killed by ... gold` (`notombstone`), and `Top Ten` scoreboard screens (compared against [rip.c](file:///home/markv/rogomatic-rl/rogue5.4/rip.c)).
   - [env.py](file:///home/markv/rogomatic-rl/rogomatic-rl/rogomatic_rl/env.py#L198-L204): Forces `self._status.hp = 0` when `terminated` triggers on a cleared tombstone screen.
2. **Breadcrumb Encoding & Anti-Oscillation Action Masking:**
   - [env.py](file:///home/markv/rogomatic-rl/rogomatic-rl/rogomatic_rl/env.py#L406-L423): Encodes visited room floor (`.`) as `~` (`126`) and visited corridor (`#`) as `` ` `` (`96`) in `obs["screen"]` and `obs["local"]` while preserving doorways (`+`), stairs (`%`), items, and monsters.
   - [env.py](file:///home/markv/rogomatic-rl/rogomatic-rl/rogomatic_rl/env.py#L313-L387): Masks out `.` (rest) at full HP when no monster is adjacent; restricts `s` (search) to tiles adjacent to walls/corridor dead ends with a 6-search cap per tile; prevents immediate U-turns (`self._prev_pos`) in corridors unless at a dead end; steers movement toward least-visited neighbor tiles (`self._visit_counts`) when not in combat; applies a small revisit penalty (`-0.005 * min(6, revisit_count - 2)`) for `revisit_count >= 3`.
3. **Extended Step Budget & Automatic Ration Eating on `"Hungry"`:**
   - [pty_client.py](file:///home/markv/rogomatic-rl/rogomatic-rl/rogomatic_rl/pty_client.py#L66): Sets `ESCDELAY=0` so `ESC` (`\x1b`) cancels prompts with zero latency.
   - [env.py](file:///home/markv/rogomatic-rl/rogomatic-rl/rogomatic_rl/env.py#L182-L205) & `_try_auto_eat` ([env.py](file:///home/markv/rogomatic-rl/rogomatic-rl/rogomatic_rl/env.py#L278-L311)): Automatically queries the food inventory (`e` -> `*`) and eats a food ration or slime-mold (`+2.0` reward) whenever `hungry_state >= 1`.
   - Raised default `max_episode_steps` / `--max-steps` to `2000` in [ppo.py](file:///home/markv/rogomatic-rl/rogomatic-rl/rogomatic_rl/ppo.py#L27), [train_ppo.py](file:///home/markv/rogomatic-rl/rogomatic-rl/scripts/train_ppo.py#L15), and [eval_ppo.py](file:///home/markv/rogomatic-rl/rogomatic-rl/scripts/eval_ppo.py#L23), plus a dynamic per-floor bonus of `+500 * (level - 1)` steps in [env.py](file:///home/markv/rogomatic-rl/rogomatic-rl/rogomatic_rl/env.py#L204-L205).
4. **Interactive HTML Training Dashboard:**
   - Generated at [training_report.html](file:///home/markv/rogomatic-rl/rogomatic-rl/training_report.html) via [generate_report.py](file:///home/markv/.gemini/antigravity-cli/brain/7fa71b42-841a-4eed-8af8-8d254ba306ff/scratch/generate_report.py) with 5 auto-scaled SVG charts (including dual-axis `best_gold`/`best_lvl` and `sps`/`pg`), phase breakdown table, and in-browser live reload from [train_8h.log](file:///home/markv/rogomatic-rl/rogomatic-rl/train_8h.log).

## 3. Training Progress & Active Run Status
- **Completed 12M-step run (`update=5859/5859`):**
  - Reached **`best_lvl=7`**, **`best_gold=535`**, peak 20-episode `mean_ret=+86.61`, and Q4 average `mean_ret=+34.91` at `636 sps`.
- **Ongoing 24M-step run (`update=6159/11718`, `step=12,613,632+`):**
  - Currently running in the background (`--resume --device cuda --num-envs 16 --total-timesteps 24000000`), logging to [train_8h.log](file:///home/markv/rogomatic-rl/rogomatic-rl/train_8h.log) and saving checkpoints to [rogomatic_ppo_latest.pt](file:///home/markv/rogomatic-rl/rogomatic-rl/checkpoints/rogomatic_ppo_latest.pt).
  - Already reached **Dungeon Level 8 (`best_lvl=8`)** and **`586` gold (`best_gold=586`)**.

## 3a. Measuring Play Quality (issues #11, #12, #16)
- **Training log.** Each `update=` line now ends with
  `ent_max` (mean of ln(legal actions), the ceiling the action mask
  puts on entropy), `ent_ratio` (`ent / ent_max`; near 1 means the
  policy is uniform over the legal actions), `legal` (mean legal
  action count) and `forced` (share of states with one legal
  action). A second line, `episodes update=N ...`, gives rolling
  statistics over the last 100 finished games: final level
  (mean, median, max), length, gold, experience and the share
  ending in death versus the step limit.
- **Per-game records.** `train_ppo.py` appends one row per finished
  game to `logs/episodes.jsonl`, including `killed_by` from the
  tombstone (`--no-episode-log` to skip).
- **Fixed evaluation.** Every `--eval-interval` updates (default
  250, 0 = never) training plays `--eval-episodes` (default 50)
  numbered dungeons, starting at `--eval-seed` (default 1000),
  prints an `eval update=N ...` line and appends a row to
  `logs/evals.jsonl`. The dungeons are the same every time, so rows
  compare across checkpoints and reward changes. Play is sampled
  unless `--eval-deterministic` is given. Evaluation games are never
  posted to the high score list (Rogue does not score seeded games).
- `RogueEnv(fixed_dungeons=True)` makes `reset(seed=n)` play Rogue's
  dungeon number `n`. Without it dungeons are random, as before.

## 4. Useful Commands
```bash
cd /home/markv/rogomatic-rl/rogomatic-rl

# Monitor active training log
tail -f train_8h.log

# Regenerate HTML report from latest train_8h.log
python3 /home/markv/.gemini/antigravity-cli/brain/7fa71b42-841a-4eed-8af8-8d254ba306ff/scratch/generate_report.py

# Watch latest checkpoint play (safe to run alongside training)
.venv/bin/python scripts/eval_ppo.py --render --episodes 1 --max-steps 2000 --delay 0.05

# Fixed evaluation of the latest checkpoint on dungeons 1000-1049
# (prints each game and a summary, appends a row to logs/evals.jsonl)
.venv/bin/python scripts/eval_ppo.py --device cuda --num-envs 16

# Watch one particular dungeon, e.g. one the evaluation died early on
.venv/bin/python scripts/eval_ppo.py --render --episodes 1 --seed 1002

# Run test suite
.venv/bin/pytest -v

# Show the agent's Rogue high score list (written by train_ppo.py and
# eval_ppo.py to scores/rogue.scr; pass --no-score-file to skip)
(cd scores && ../../rogue5.4/rogue -s)
```

## 5. Suggested Skills for Next Session
- `diagnose` (`/home/markv/.gemini/antigravity-cli/skills/diagnose/SKILL.md`): Use if diagnosing agent behavior bottlenecks or PTY/curses edge cases on deeper dungeon levels.
- `tdd` (`/home/markv/.gemini/antigravity-cli/skills/tdd/SKILL.md`): Use when adding new inventory/equipment automation (e.g., auto-equipping better armor/weapons or quaffing healing potions) with unit tests in [test_env.py](file:///home/markv/rogomatic-rl/rogomatic-rl/tests/test_env.py).
