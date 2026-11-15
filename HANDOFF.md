# Rog-O-Matic RL Handoff

Written 2026-11-14 16:55 PST, updated 17:24 PST. The previous
handoff is in git history (`git log -- HANDOFF.md`).

## 1. What this project is

A PPO agent (CNN + GRU) learning to play Rogue 5.4 through a
pseudo-terminal.

- Rogue binary: `rogue5.4/rogue`, built from source in this repo
  (`./configure && make CFLAGS="-g -O2 -std=gnu89 -DNCURSES_OPAQUE=0"`).
- Python package: `rogomatic-rl/`. Run commands from there, with
  `.venv/bin/python`.
- Issues: GitHub `brainwagon/rogomatic-rl`, via `gh`. See
  `AGENTS.md` and `docs/agents/`.
- Tests: `.venv/bin/pytest -q` gives 54 passed. One run in twelve
  showed a single failure that could not be reproduced or
  identified; treat a one-off failure as possibly flaky, then look.

## 2. State right now

### A training run is in progress

- Started 16:19 with the new rewards ("rewards v2"), from the
  mask-only weights with the value head reset:

        .venv/bin/python scripts/train_ppo.py \
          --init-from checkpoints/mask_only_final.pt --reset-critic \
          --device cuda --num-envs 16 --total-timesteps 12000000 \
          --episode-log logs/reward-v2/episodes.jsonl \
          --eval-log logs/reward-v2/evals.jsonl

- Text log: `rogomatic-rl/train_reward_v2.log`. At 17:24 it was at
  update 780 of 5859, about 468 steps/s, so it should finish
  around 23:30.
- It writes `checkpoints/rogomatic_ppo_latest.pt`.
- To stop it: list the process with
  `pgrep -af "train_ppo.py --init-from"`, then send that PID an
  interrupt (`kill -INT`) in a separate command. That saves a
  checkpoint. `--resume` with the same other flags continues it.
- The running process loaded the code as it was at 16:19. Edits
  to `env.py` or `ppo.py` do not affect it.

### The question this run answers

Do the new rewards improve play beyond the fight-back mask alone?

| Run | Mean final level (fixed evaluation) |
|---|---|
| Before the mask, old rewards | 1.6-2.1 |
| Mask only, old rewards (4 evaluations) | 2.79 (2.64-2.90) |
| Rewards v2, update 250 | 2.92 |
| Rewards v2, update 500 | 3.34 |
| Rewards v2, update 750 | 2.44 |

The three rewards-v2 evaluations average 2.90, level with the
baseline: the 3.34 at update 500 was not a trend.

- Each evaluation is 50 sampled games on dungeons 1000-1049.
  Differences of about 0.3 are noise.
- Read the `eval update=N` lines in `train_reward_v2.log`, or
  `logs/reward-v2/evals.jsonl`. Only a sustained level clearly
  above 2.9 means the rewards helped.
- `mean_ret` is on a new scale and is not comparable with older
  runs.

### Git state

Everything up to the rewards and replays is on `main` and pushed
(`origin/main` at `0a5f941`). The working tree is on a new branch,
`feat/auto-equip-and-quaff`, made from `main` for #8 and #9. It has
no commits or code changes yet; only this file differs.

Left out of the commits on purpose:

- `checkpoints/rogomatic_ppo_latest.pt` is tracked, but the running
  trainer rewrites it every 10 updates, so it always shows as
  modified. It is an unproven mid-run snapshot.
- Untracked: `rogomatic-rl/rogue.scr`, `train_8h.log`,
  `train_reward_v2.log`, `training_report.html` and
  `checkpoints/mask_only_final.pt`.

`git push` over HTTPS has no stored credentials here. This works
for one command:

    git -c credential.helper= \
      -c credential.helper='!gh auth git-credential' push

Do not `git stash` while training runs: the tracked checkpoint
gets stashed and the trainer then rewrites it, so the pop fails.

### A draft blog post

- `~/brainwagon-okf/content/2026-11-14-teaching-a-neural-network-to-play-rogue-badly.md`,
  `draft: true`, with a replay embedded. Not committed or
  published; publishing is the user's call (`bw publish`).
- A preview server was left running on port 8377. Stop it with
  `./bw preview --stop` from `~/brainwagon-okf`.
- The Rog-O-Matic sentence in the post was written from memory and
  has not been checked against a source.
- The recording and snippet used in the post are kept in
  `rogomatic-rl/logs/replays/dungeon-1018.json` and `.html`.
- The Archive's build escapes any line that starts with `#`, even
  inside raw HTML. The replay template avoids such lines, and
  `render_html` asserts it.

## 3. What was learned

- **The policy was not fighting.** With a monster adjacent it
  attacked on 37% of steps and moved away on 62%. 95% of games
  ended in death, 55% on level 1, mostly to hobgoblins. Details
  and numbers are in issue #17.
- **A mask fixed most of it without retraining.** Making attacks
  the only legal action next to a monster took mean final level
  from about 1.9 to about 2.8, on two separate sets of 50
  dungeons.
- **Training under the mask with the old rewards added nothing**
  over about 900 updates.
- **A crude rest rule did not help** (rest below 60% hp with no
  monster in view). Not conclusive at 50 games.
- **Entropy was never the problem.** `ent_ratio` sits near 0.65
  with about 3.3 legal actions per state.
- **The old reward scale caused the value-loss spikes** (90th
  percentile 30-36, peaks of 80). With rewards v2 and scaling it
  is about 0.01-0.3.
- **Longer games exposed new problems:** 14-19% of games now hit
  the step limit, mostly on levels 1-2, and starvation and
  hypothermia appear among the killers.
- **Best ever:** level 13, 979 gold. The amulet is on level 26,
  and the agent has no `<` action to climb back up.

## 4. How to measure play

- Each `update=` line ends with `ent_max`, `ent_ratio`, `legal`,
  `forced` and `rscale`.
- Each `episodes update=N` line gives rolling statistics over the
  last 100 training games.
- The episode log (`--episode-log`) has one row per training game,
  including `killed_by`.
- A fixed evaluation runs every 250 updates on numbered dungeons.
  `RogueEnv(fixed_dungeons=True)` makes `reset(seed=n)` pick the
  dungeon; seeded games are never posted to the high score list.

## 5. Rewards v2

Constants at the top of `rogomatic_rl/env.py`:

| Term | Value | Before |
|---|---|---|
| Descend a level | +10 | +50 |
| Death | -10 | -10 |
| Health, full to zero | 5, as a difference in hp fraction | none |
| Experience point | +0.2, capped at 5 per step | +0.1 |
| Experience level | +3 | none |

Exploration, gold, step and revisit terms are unchanged. The
trainer also divides rewards by the running spread of the
discounted return (`RewardScaler` in `ppo.py`, saved in the
checkpoint; `--no-normalize-reward` turns it off).

## 6. Issues

| Issue | Status |
|---|---|
| #8 auto-equip armor and weapons | In progress; research done, no code yet (section 7) |
| #9 quaff potions | In progress; research done, no code yet (section 7) |
| #13 why games end shallow | Mostly answered (combat); step-limit games on levels 1-2 remain |
| #14, #15 rewards | Merged with `Refs`; open until the run is judged, then close by hand |
| #10, #11, #12, #16, #17 | Closed |

No issue exists for an ascend (`<`) action or for a rest rule.

## 7. In progress: #8 and #9

The user asked to implement both. Work was interrupted after reading
the Rogue source, before any design was settled or code written.
Both are meant to be scripted in the env, like `_try_auto_eat`, and
to share one inventory tracker.

### Facts from the Rogue 5.4 source (`rogue5.4/`)

Inventory and prompts:

- `i` lists the whole pack and takes no game turn. With one item
  Rogue prints it on the message line. With several it draws an
  overlay at the top right, one `x) name` line per item, ending in
  `--Press space to continue--` (`add_line` and `end_line`).
  `_overlay_slot` and `_dismiss_overlay` in `env.py` already deal
  with this for food.
- Items in use carry ` (being worn)` or ` (weapon in hand)`
  (`inv_name` in `things.c`; `inv_describe` is on by default).
- A pickup prints `<name> (x)` in terse mode (`add_pack` in
  `pack.c`), which can trigger an inventory refresh.
- Item prompts read `wear what? (* for list): ` in terse mode (the
  same for `wield`, `quaff`). ESC cancels without using a turn.

Item names (`inv_name`, `nameit`, `num`):

- Armor, unidentified: `Scale mail`. Identified: `+1 ring mail [4]`
  (no `protection` in terse mode). The bracket is 10 minus the
  armor value, the same scale as `Arm:` on the status line.
- Weapon, unidentified: `A long sword`. Identified:
  `A +1,+1 mace`. Stacks: `25 arrows`.
- Potion, unidentified: `A pink potion` or `2 pink potions`.
  Identified: `A potion of healing(pink)`. Named by the player:
  `A potion called X(pink)`.

Equipment:

- Base armor class, lower is better (`a_class`, `extern.c`):
  leather 8, ring mail 7, studded leather 7, scale 6, chain 5,
  splint 4, banded 4, plate 3. The player starts in +1 ring mail
  (`Arm: 4`).
- Wielded damage (`init_dam`, `weapons.c`): mace 2x4, long sword
  3x4, two handed sword 4x4, spear 2x3, dagger 1x6, shuriken 1x2,
  bow, arrow and dart 1x1. The player starts with a +1,+1 mace.
- `W` refuses while armor is worn (`you are already wearing some`),
  so take it off with `T` first. Wearing takes extra turns
  (`waste_time`) and identifies the armor.
- `T` and `w` fail on a cursed item in use:
  `you can't.  It appears to be cursed`. A cursed item is stuck.
- Messages: `wearing <name>`, `was wearing x) <name>`,
  `wielding <name> (x)`.

Potions (`potions.c`, frequencies out of 100 in `pot_info`):

- Good: healing 13 (heals level d4, cures blindness, +1 max hp if
  it overflows), extra healing 5 (level d8, also cures
  hallucination), gain strength 13, restore strength 13, raise
  level 2, haste self 5, see invisible 3.
- Neutral-ish: monster detection 6, magic detection 6, levitation 6
  (cannot take stairs while floating).
- Bad: confusion 7 (20 turns), hallucination 8 (850 turns, changes
  monster and item letters), poison 8 (lose 1-4 strength until
  restored), blindness 5 (850 turns).
- After drinking a potion that is still unidentified, Rogue asks
  `call it: ` (`call_it`). The env must answer with ESC, or the
  game sits at that prompt.
- Healing is weak early: 1d4 at experience level 1.

Known hazards for the env:

- Blindness: unseen squares are blank, which the action mask reads
  as walls, so the agent may be stuck resting for up to 850 turns
  (the issue's own warning).
- Hallucination: monsters still show as capital letters, so the
  fight-back mask keeps working, but `killed_by` and the replay
  colouring see random letters.

### Open design decisions

- Whether to drink unknown potions to identify them when safe
  (full hp, no monster in view), as the original Rog-O-Matic did,
  or only drink potions already known. Blindness is the main risk
  of the first.
- When to drink healing: an hp threshold, and whether to drink an
  unknown potion in an emergency.
- What counts as better armor or weapon before identification:
  compare base types only, or accept the cursed-item risk.
- Whether successful equips or quaffs earn a shaping reward.

### Safety notes for the code change

- The running trainer uses forked workers that already imported the
  old code, and fixed evaluations fork from it too, so editing
  `env.py` does not affect that run.
- Test with `_FakeClient` scripted screens in `tests/test_env.py`,
  then on the real binary (`RogueEnv(fixed_dungeons=True)` gives a
  repeatable dungeon).

## 8. Suggested next steps

1. When the run finishes, compare its `eval` lines with the 2.79
   baseline and decide whether rewards v2 stays; then close #14 and
   #15.
2. Settle the open decisions in section 7 with the user, then build
   the inventory tracker, #8 and #9 test-first.
3. Measure each with a fixed evaluation of the same checkpoint with
   the feature on and off, as was done for the fight-back mask.

## 9. Useful commands

Run from `rogomatic-rl/`.

    # Follow the current run
    tail -f train_reward_v2.log
    grep "^eval " train_reward_v2.log

    # Fixed evaluation of a checkpoint on dungeons 1000-1049
    # (appends a row to logs/evals.jsonl; --no-out to skip)
    .venv/bin/python scripts/eval_ppo.py --device cuda --num-envs 16

    # Watch one dungeon on screen
    .venv/bin/python scripts/eval_ppo.py --render --episodes 1 --seed 1002

    # Record the best of 20 games and make an embeddable replay
    # (--id must be unique on the page it is pasted into)
    .venv/bin/python scripts/eval_ppo.py --record game.json --record-best 20
    .venv/bin/python scripts/make_replay.py game.json -o replay.html --id rogue-1

    # The agent's Rogue high score list
    (cd scores && ../../rogue5.4/rogue -s)

    # Tests
    .venv/bin/pytest -q

Evaluation and recording are safe to run while training.

## 10. Suggested skills

- `tdd`: for #8 and #9, test-first in `tests/test_env.py` with the
  `_FakeClient` scripted screens.
- `diagnosing-bugs`: if a run stalls or a test fails
  intermittently.
- `brainwagon-blog-post`: for any change to the draft post.
- `pr`: when opening the pull request.
