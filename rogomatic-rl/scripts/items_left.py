"""Count the items a checkpoint leaves behind when it goes downstairs.

Plays numbered dungeons (as the fixed evaluation does) and, at every
``>`` keypress, counts the items and gold on screen and whether a path
over seen tiles reaches them.  Prints a summary and, with ``--out``,
saves every descent as JSON.  Written for issue #23.

    .venv/bin/python scripts/items_left.py checkpoints/rogomatic_ppo_latest.pt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from multiprocessing import Pool
from typing import Any

import numpy as np
import torch

from rogomatic_rl import ACTIONS, RogueEnv
from rogomatic_rl.env import _COLLECT_CHARS
from rogomatic_rl.evaluate import DEFAULT_EVAL_SEED
from rogomatic_rl.model import RogueActorCritic


def _play(job: tuple[str, int, int]) -> dict[str, Any]:
    checkpoint, seed, max_steps = job
    torch.set_num_threads(1)
    agent = RogueActorCritic(num_actions=len(ACTIONS))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    env = RogueEnv(max_steps=max_steps, fixed_dungeons=True)
    descents: list[dict[str, Any]] = []
    try:
        obs, info = env.reset(seed=seed)
        hx = torch.zeros((1, 1, agent.hidden_dim))
        done = False
        while not done:
            def t(key: str, dtype: torch.dtype) -> torch.Tensor:
                return torch.as_tensor(obs[key], dtype=dtype).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _, hx = agent.get_action_and_value(
                    t("screen", torch.uint8),
                    t("local", torch.uint8),
                    t("stats", torch.float32),
                    hx,
                    action_mask=t("action_mask", torch.uint8),
                )
            a = int(action.item())
            if ACTIONS[a] == ">":
                descents.append(_survey(env, int(obs["stats"][0])))
            obs, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated
    finally:
        env.close()
    return {
        "seed": seed,
        "level": info.get("level"),
        "steps": env._steps,
        "end": info.get("end"),
        "killed_by": info.get("killed_by"),
        "descents": descents,
    }


def _survey(env: RogueEnv, level: int) -> dict[str, Any]:
    """Items on screen now, and path lengths to the reachable ones."""
    screen = env._last_screen
    found = np.argwhere(screen == ord("@"))
    reach = (
        env._path_distances(screen, [tuple(int(v) for v in found[0])])
        if len(found)
        else {}
    )
    visible: Counter[str] = Counter()
    reachable: Counter[str] = Counter()
    dists: list[int] = []
    for r in range(1, 23):
        for c in range(80):
            ch = chr(int(screen[r, c]))
            if ch in _COLLECT_CHARS:
                visible[ch] += 1
                if (r, c) in reach:
                    reachable[ch] += 1
                    dists.append(reach[(r, c)])
    return {
        "level": level,
        "step": env._steps,
        "visible": dict(visible),
        "reachable": dict(reachable),
        "dists": dists,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", help="save every game and descent as JSON")
    args = parser.parse_args()

    jobs = [
        (args.checkpoint, args.seed + k, args.max_steps)
        for k in range(args.episodes)
    ]
    with Pool(args.workers) as pool:
        games = pool.map(_play, jobs)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(games, f)

    descents = [d for g in games for d in g["descents"]]
    levels = [g["level"] or 0 for g in games]
    print(f"games {len(games)}  descents {len(descents)}")
    print(f"final level mean {np.mean(levels):.2f}  max {max(levels)}")
    leaving = sum(1 for d in descents if d["reachable"])
    print(
        f"descents leaving a reachable item {leaving} "
        f"({leaving / max(1, len(descents)):.0%})"
    )
    reachable: Counter[str] = Counter()
    for d in descents:
        reachable.update(d["reachable"])
    print(f"reachable items left {dict(reachable.most_common())}")
    dists = [x for d in descents for x in d["dists"]]
    if dists:
        print(
            f"path length to them: median {np.median(dists):.0f}  "
            f"90th percentile {np.percentile(dists, 90):.0f}"
        )
    ends = Counter(g["killed_by"] or g["end"] for g in games)
    print(f"ends {dict(ends.most_common())}")


if __name__ == "__main__":
    main()
