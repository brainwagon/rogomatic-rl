"""Smoke tests for RogueEnv and the Rogue 5.4 screen parser."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from rogomatic_rl import ACTIONS, RogueEnv
from rogomatic_rl.parser import parse_status, screen_to_array


def test_parse_status() -> None:
    line = "Level: 1  Gold: 0      Hp: 12(12)  Str: 16(16)  Arm: 4   Exp: 1/0"
    status = parse_status(line)
    assert status is not None
    assert status.level == 1
    assert status.gold == 0
    assert status.hp == 12
    assert status.max_hp == 12
    assert status.strength == 16
    assert status.max_strength == 16
    assert status.armor == 4
    assert status.exp_level == 1
    assert status.exp == 0
    assert status.hungry_state == 0

    hungry_line = "Level: 3  Gold: 142    Hp: 8(15)   Str: 16(16)  Arm: 5   Exp: 2/18  Hungry"
    hungry_status = parse_status(hungry_line)
    assert hungry_status is not None
    assert hungry_status.level == 3
    assert hungry_status.gold == 142
    assert hungry_status.hp == 8
    assert hungry_status.hungry_state == 1


def test_env_reset_and_step() -> None:
    env = RogueEnv(max_steps=50)
    try:
        obs, info = env.reset()
        assert obs["screen"].shape == (24, 80)
        assert obs["screen"].dtype == np.uint8
        assert obs["local"].shape == (11, 11)
        assert obs["local"].dtype == np.uint8
        assert obs["stats"].shape == (10,)
        assert obs["stats"].dtype == np.float32
        assert info["player_pos"] is not None
        assert info["hp"] > 0

        for i in range(20):
            action = i % len(ACTIONS)
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs["screen"].shape == (24, 80)
            assert isinstance(reward, float)
            if terminated or truncated:
                break
    finally:
        env.close()


def test_gymnasium_check_env() -> None:
    env = gym.make("Rogue54-v0", max_steps=30)
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_action_mask_corridors_and_diagonals() -> None:
    env = RogueEnv(max_steps=10)
    try:
        lines = [" " * 80 for _ in range(24)]
        # Player in doorway '+' at (5, 5) with vertical walls '|' at (4, 5) and (6, 5),
        # room floor '.' to the left, and corridor '#' to the right.
        row4 = list(lines[4])
        row5 = list(lines[5])
        row6 = list(lines[6])
        row4[4], row4[5] = ".", "|"
        row5[4], row5[5], row5[6] = ".", "+", "#"
        row6[4], row6[5] = ".", "|"
        lines[4] = "".join(row4)
        lines[5] = "".join(row5)
        lines[6] = "".join(row6)

        screen_arr = screen_to_array(lines)
        mask = env._compute_action_mask(screen_arr, (5, 5))

        # h (left to '.') and l (right to '#') are valid
        assert mask[0] == 1  # h
        assert mask[3] == 1  # l
        # j (down into '|') and k (up into '|') are masked
        assert mask[1] == 0  # j
        assert mask[2] == 0  # k
        # All 4 diagonals (y, u, b, n) are masked by flanking '|' walls
        assert mask[4] == 0  # y
        assert mask[5] == 0  # u
        assert mask[6] == 0  # b
        assert mask[7] == 0  # n
    finally:
        env.close()
