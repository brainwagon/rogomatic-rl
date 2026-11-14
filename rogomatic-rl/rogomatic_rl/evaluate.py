"""Fixed evaluation: play a set of numbered dungeons with a trained policy.

Every evaluation with the same ``seed_base`` and ``episodes`` plays the
same dungeons, so results are comparable between checkpoints and between
runs whatever the reward shaping was.  Nothing is learned and nothing is
posted to the high score list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch

from .env import RogueEnv
from .episodes import EpisodeRecord
from .model import RogueActorCritic
from .pty_client import DEFAULT_ROGUE_BIN

# Dungeon numbers used by default: episode k plays dungeon DEFAULT_EVAL_SEED + k.
DEFAULT_EVAL_SEED = 1000


def _make_eval_env(max_steps: int, rogue_bin: str | Path) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        return RogueEnv(
            rogue_bin=rogue_bin,
            max_steps=max_steps,
            score_file=None,
            fixed_dungeons=True,
        )

    return _thunk


def evaluate_policy(
    agent: RogueActorCritic,
    device: torch.device | str = "cpu",
    episodes: int = 50,
    num_envs: int = 8,
    max_steps: int = 2000,
    seed_base: int = DEFAULT_EVAL_SEED,
    deterministic: bool = False,
    rogue_bin: str | Path = DEFAULT_ROGUE_BIN,
) -> list[EpisodeRecord]:
    """Play dungeons ``seed_base .. seed_base + episodes - 1``, one game each.

    Returns one record per dungeon, in dungeon order.  With
    ``deterministic`` the policy plays its most probable action; otherwise
    it samples, as it does in training.  Sampled play of a fixed dungeon
    still varies from one evaluation to the next.
    """
    if episodes <= 0:
        return []
    device = torch.device(device)
    num_envs = max(1, min(num_envs, episodes))

    envs = gym.vector.AsyncVectorEnv(
        [_make_eval_env(max_steps, rogue_bin) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    try:
        # Each slot plays one dungeon at a time; a slot whose game ends
        # takes the next unplayed dungeon, or goes idle (None) when none
        # are left.  An idle slot is still stepped, but its results are
        # ignored.
        pending = list(range(seed_base, seed_base + episodes))
        slot_seed: list[Optional[int]] = [pending.pop(0) for _ in range(num_envs)]
        results: dict[int, EpisodeRecord] = {}
        returns = np.zeros(num_envs, dtype=np.float64)

        obs, _ = envs.reset(seed=list(slot_seed))
        hx = torch.zeros((1, num_envs, agent.hidden_dim), dtype=torch.float32, device=device)

        while len(results) < episodes:
            with torch.no_grad():
                action, _, _, _, hx = agent.get_action_and_value(
                    torch.as_tensor(obs["screen"], dtype=torch.uint8, device=device),
                    torch.as_tensor(obs["local"], dtype=torch.uint8, device=device),
                    torch.as_tensor(obs["stats"], dtype=torch.float32, device=device),
                    hx,
                    action_mask=torch.as_tensor(
                        obs["action_mask"], dtype=torch.uint8, device=device
                    ),
                    deterministic=deterministic,
                )
            obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            returns += reward

            reset_mask = np.zeros(num_envs, dtype=np.bool_)
            for i in range(num_envs):
                seed = slot_seed[i]
                if seed is None or not (terminated[i] or truncated[i]):
                    continue
                results[seed] = EpisodeRecord.from_info(
                    {key: values[i] for key, values in infos.items()},
                    ret=float(returns[i]),
                    seed=seed,
                )
                slot_seed[i] = pending.pop(0) if pending else None
                if slot_seed[i] is not None:
                    reset_mask[i] = True
                    returns[i] = 0.0
                    hx[:, i] = 0.0

            if reset_mask.any():
                obs, _ = envs.reset(
                    seed=[s if m else None for s, m in zip(slot_seed, reset_mask)],
                    options={"reset_mask": reset_mask},
                )
    finally:
        envs.close()

    return [results[seed] for seed in sorted(results)]
