# LEARN-TODO: giving learning a real role

Ideas for making the PPO agent learn something the scripted rules
do not already do.  Nothing here is implemented yet.

## Where things stand

- In #28 a uniform-random choice among the legal actions reached
  mean depth 5.48 against 5.24 for the trained checkpoint.  The
  policy adds nothing measurable.
- Every gain since #21 came from scripted rules in
  `rogomatic-rl/rogomatic_rl/env.py`.  Main now reaches about
  5.2-5.4 mean final depth on 1,200 fixed dungeons.
- Why the policy has nothing to learn:
  - `_compute_action_mask` runs a chain of steering rules (items,
    stairs, rest, dead ends, frontier, sleepers, wall search,
    fallback).  Most of them leave one legal move, or a few equally
    good ones.
  - `_run_auto_actions` makes the item and combat choices (heal
    quaff, emergency scroll, zap, last-chance quaff, throw, bow,
    eating) without the policy seeing them.
  - `_compute_reward` pays for new tiles and penalises revisits.
    What we measure is final depth and deaths, so even a real
    choice gets only a rough signal.

## Ideas, most promising first

### 1. Options: let the policy choose when, scripts do how

Turn the scripted behaviours into options, each running for many
turns, and make the policy choose between them:

- explore (frontier)
- go to the stairs / descend
- search here
- rest
- fight the adjacent monster / retreat to a corridor
- shoot or throw
- quaff, read, zap, eat
- walk to a sleeper

Pathfinding, aiming and inventory menus stay scripted.  The mask
forbids only what is illegal or plainly suicidal, not what the rules
think is suboptimal.  One decision covers many game turns, which
makes credit assignment far easier than per-keypress actions.

These are the decisions hand rules have handled worst:

- when to go down (the #26 rule made things worse)
- fight or flee, by monster, HP, XL and pack
- whether to try an unknown potion or scroll
- when to stop searching and accept a stalled level

### 2. Start from imitation, then fine-tune with RL

- Record the scripted agent's option choices over many games and
  train the network to copy them (behaviour cloning).  It then
  starts at the scripted agent's level instead of below random.
- Fine-tune with PPO, with a KL penalty toward the scripted policy
  so early updates do not throw away what it copied.

### 3. Reward what we measure

- Reward depth reached, with a death penalty and perhaps gold.
- Keep only potential-based shaping, which cannot change which
  policy is best.  The current reward can rank "explore every tile
  and die on level 3" above "descend safely".

### 4. Search and self-play substitutes

Rogue is single-player, so AlphaZero-style self-play has no
opponent.  The nearest equivalents:

- Expert iteration with search.  Fixed seeds make games replayable
  (seed plus action list reproduces a game), so a planner can try
  lookaheads such as fight vs. flee from a decision point, and the
  network learns to predict the planner.  Highest ceiling, but each
  lookahead replays the game through the PTY from the start, so it
  is slow.
- Population / evolutionary training (#34): evolve rule thresholds
  or several policies, selecting on depth over fixed dungeons.  The
  cheapest form of learning and the best fit for the current code.
- Curriculum: train more on the seeds where the agent stalls or
  dies.

### 5. Give the policy memory

The model already has a GRU.  A learned policy also needs state the
screen shows badly: identified items, potions tried with no effect,
turns since the last meal, searches per wall.  The env already
tracks much of this for the rules and could feed it to the network
as observations.

## Likely results (estimates, not measurements)

- Pure end-to-end RL from the screen, with a small mask and sparse
  reward, at this project's compute: almost certainly worse than
  the scripted agent.  In the NetHack Challenge (NeurIPS 2021) the
  symbolic bots beat the best neural agents by a wide margin, and
  Rog-O-Matic itself was an expert system.
- Options plus imitation plus RL fine-tuning: should match the
  current agent, then perhaps gain half a level to a level of mean
  depth, mostly from descend timing and fight-or-flee.  A guess
  until a pilot shows otherwise.
- Search-based expert iteration: highest ceiling, but probably
  days of compute per improvement step at this simulator speed.

## Measuring

- 1,200 fixed dungeons give about ±0.1 standard error on mean
  depth, and two runs of the same code have differed by 0.07.  A
  difference needs to be about 0.3 to be clearly real.
- One 1,200-game run takes about half an hour on 13 workers, so
  evaluation is a large share of the cost of any training loop.

## First step: a pilot

Make one decision a policy choice and keep everything else
scripted: descend now vs. keep exploring, when the stairs are known.

1. Imitate the current rule to initialise the policy.
2. Fine-tune with PPO on depth-based reward.
3. Compare against main on 1,200 fixed dungeons.

If PPO beats the rule, widen the option set.  If it cannot beat
even this one rule, the hybrid approach needs rethinking before
more work goes into it.
