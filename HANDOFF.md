# Rog-O-Matic RL Handoff

Written 2026-11-14 16:55 PST. Replaces the earlier handoff; the
previous text is in git history (`git log -- HANDOFF.md`).

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

- Text log: `rogomatic-rl/train_reward_v2.log`. It was at update
  440 of 5859, about 470 steps/s, so it should finish around
  23:30.
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

- Each evaluation is 50 sampled games on dungeons 1000-1049.
  Differences of about 0.3 are noise.
- Read the `eval update=N` lines in `train_reward_v2.log`, or
  `logs/reward-v2/evals.jsonl`. Only a sustained level clearly
  above 2.9 means the rewards helped.
- `mean_ret` is on a new scale and is not comparable with older
  runs.

### Git state

All the work is committed and merged into local `main`, as a
fast-forward from `feat/episode-stats-and-fixed-eval` (which is on
`origin`). `main` was not pushed when this was written; check with
`git status -sb`. There is no pull request.

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
| #8 auto-equip armor and weapons | Not started; next in line |
| #9 quaff potions | Not started; after #8 |
| #13 why games end shallow | Mostly answered (combat); step-limit games on levels 1-2 remain |
| #11, #12, #16, #17 | Done and merged to local `main`; the commit messages say `Closes`, so they close when `main` is pushed |
| #10 high score list | Done and merged; the commit only references it, so close by hand |
| #14, #15 rewards | Done and merged, but the commit says `Refs`, so they stay open until the run's evaluations confirm the gain; close by hand |

No issue exists for an ascend (`<`) action or for a rest rule.

## 7. Suggested next steps

1. When the run finishes, or after several evaluations, compare
   its `eval` lines with the 2.79 baseline and decide whether
   rewards v2 stays.
2. Push `main` when the user asks, which closes #11, #12, #16 and
   #17.
3. Close #10, #14 and #15 by hand once the run has been judged.
4. Start #8 (auto-equip). It needs inventory tracking, which #9
   reuses. The multi-line inventory overlay needs care; see how
   `_try_auto_eat` handles it.

## 8. Useful commands

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

## 9. Suggested skills

- `tdd`: for #8 and #9, test-first in `tests/test_env.py` with the
  `_FakeClient` scripted screens.
- `diagnosing-bugs`: if a run stalls or a test fails
  intermittently.
- `brainwagon-blog-post`: for any change to the draft post.
- `pr`: when opening the pull request.
