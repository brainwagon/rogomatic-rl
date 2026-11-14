"""Gymnasium environment wrapping Rogue 5.4 via a 24x80 PTY."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .parser import (
    RogueStatus,
    extract_local_crop,
    find_player,
    is_death_or_end_screen,
    parse_status,
    screen_to_array,
)
from .pty_client import DEFAULT_ROGUE_BIN, RoguePTYClient


ACTIONS: tuple[str, ...] = (
    "h",  # 0: move left
    "j",  # 1: move down
    "k",  # 2: move up
    "l",  # 3: move right
    "y",  # 4: move up-left
    "u",  # 5: move up-right
    "b",  # 6: move down-left
    "n",  # 7: move down-right
    ".",  # 8: rest one turn
    "s",  # 9: search adjacent squares
    ">",  # 10: descend stairs
    ",",  # 11: pick up item
)

_MOVE_DELTAS: tuple[tuple[int, int], ...] = (
    (0, -1),   # h
    (1, 0),    # j
    (-1, 0),   # k
    (0, 1),    # l
    (-1, -1),  # y
    (-1, 1),   # u
    (1, -1),   # b
    (1, 1),    # n
)

_WALL_CHARS = frozenset({" ", "-", "|"})
_ITEM_CHARS = frozenset({"!", "?", ":", ")", "]", "/", "=", ","})


class RogueEnv(gym.Env[dict[str, np.ndarray], int]):
    """Gymnasium environment for Rogue 5.4."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 15}

    def __init__(
        self,
        rogue_bin: str | Path = DEFAULT_ROGUE_BIN,
        max_steps: int = 2000,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.rogue_bin = Path(rogue_bin)
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(
                    low=0, high=127, shape=(24, 80), dtype=np.uint8
                ),
                "local": spaces.Box(
                    low=0, high=127, shape=(11, 11), dtype=np.uint8
                ),
                "stats": spaces.Box(
                    low=-1e6, high=1e6, shape=(10,), dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(len(ACTIONS),), dtype=np.uint8
                ),
            }
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

        self.client = RoguePTYClient(rogue_bin=self.rogue_bin)
        self._steps: int = 0
        self._status: RogueStatus = RogueStatus()
        self._last_lines: list[str] = [" " * 80 for _ in range(24)]
        self._last_screen: np.ndarray = np.full((24, 80), ord(" "), dtype=np.uint8)
        self._last_pos: Optional[tuple[int, int]] = None
        self._tile_under_player: str = "."
        self._seen_mask: np.ndarray = np.zeros((22, 80), dtype=bool)
        self._visited_mask: np.ndarray = np.zeros((22, 80), dtype=bool)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._steps = 0
        self._seen_mask.fill(False)
        self._visited_mask.fill(False)
        self._tile_under_player = "."

        messages = self.client.start()
        lines = self.client.get_lines()
        self._last_lines = lines

        parsed = parse_status(lines[23])
        self._status = parsed if parsed is not None else RogueStatus()
        self._last_pos = find_player(lines)

        screen_arr = screen_to_array(lines)
        self._last_screen = screen_arr
        self._seen_mask |= screen_arr[1:23] != 32
        if self._last_pos is not None:
            self._visited_mask[self._last_pos[0] - 1, self._last_pos[1]] = True

        obs = self._build_obs(screen_arr, self._last_pos, self._status)
        info = self._build_info(self._last_pos, messages)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}")

        self._steps += 1
        action_idx = int(action)
        messages = self.client.send_key(ACTIONS[action_idx])
        lines = self.client.get_lines()
        self._last_lines = lines

        prev_status = self._status
        prev_pos = self._last_pos
        parsed = parse_status(lines[23])
        if parsed is not None:
            self._status = parsed

        curr_pos = find_player(lines)
        screen_arr = screen_to_array(lines)
        msg_text = " ".join(messages).lower()

        if self._status.level != prev_status.level:
            self._seen_mask.fill(False)
            self._visited_mask.fill(False)
            self._tile_under_player = "."
        elif curr_pos is not None and curr_pos != prev_pos:
            stepped_tile = chr(int(self._last_screen[curr_pos[0], curr_pos[1]]))
            if stepped_tile == "%":
                self._tile_under_player = "%"
            elif stepped_tile in _ITEM_CHARS and (
                "moved onto" in msg_text or "no room" in msg_text
            ):
                self._tile_under_player = stepped_tile
            else:
                self._tile_under_player = "."
        elif action_idx == 11 and "no room" not in msg_text:
            self._tile_under_player = "."

        self._last_pos = curr_pos
        self._last_screen = screen_arr

        terminated = (
            not self.client.alive
            or is_death_or_end_screen(lines)
            or self._status.hp <= 0
        )
        truncated = (not terminated) and (self._steps >= self.max_steps)

        curr_mask = screen_arr[1:23] != 32
        new_tiles = int(np.count_nonzero(curr_mask & ~self._seen_mask))
        self._seen_mask |= curr_mask

        new_visit = False
        if curr_pos is not None and 1 <= curr_pos[0] <= 22:
            if not self._visited_mask[curr_pos[0] - 1, curr_pos[1]]:
                self._visited_mask[curr_pos[0] - 1, curr_pos[1]] = True
                new_visit = True

        wasted_turn = False
        if action_idx < len(_MOVE_DELTAS):
            wasted_turn = (
                prev_pos is not None
                and curr_pos == prev_pos
                and self._status.exp == prev_status.exp
                and self._status.hp == prev_status.hp
                and not terminated
            )
        elif action_idx == 10:  # '>'
            wasted_turn = (self._status.level == prev_status.level) or (
                "no way down" in msg_text
            )
        elif action_idx == 11:  # ','
            wasted_turn = "nothing here" in msg_text

        reward = self._compute_reward(
            prev_status,
            self._status,
            new_tiles,
            new_visit,
            wasted_turn,
            terminated,
        )
        obs = self._build_obs(screen_arr, curr_pos, self._status)
        info = self._build_info(curr_pos, messages)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        output = "\n".join(self._last_lines)
        if self.render_mode == "human":
            print("\033[H\033[J" + output, flush=True)
            return None
        if self.render_mode == "ansi":
            return output
        return None

    def close(self) -> None:
        self.client.close()

    def _compute_action_mask(
        self, screen_arr: np.ndarray, player_pos: Optional[tuple[int, int]]
    ) -> np.ndarray:
        mask = np.ones(len(ACTIONS), dtype=np.uint8)
        if player_pos is not None:
            pr, pc = player_pos
            for idx, (dr, dc) in enumerate(_MOVE_DELTAS):
                nr, nc = pr + dr, pc + dc
                if not (1 <= nr <= 22 and 0 <= nc < 80):
                    mask[idx] = 0
                else:
                    ch = chr(int(screen_arr[nr, nc]))
                    if ch in _WALL_CHARS:
                        mask[idx] = 0
                    elif dr != 0 and dc != 0:
                        orth_r = chr(int(screen_arr[nr, pc]))
                        orth_c = chr(int(screen_arr[pr, nc]))
                        if orth_r in _WALL_CHARS or orth_c in _WALL_CHARS:
                            mask[idx] = 0

        if self._tile_under_player != "%":
            mask[10] = 0  # '>' only valid on stairs
        if self._tile_under_player not in _ITEM_CHARS:
            mask[11] = 0  # ',' only valid on items

        return mask

    @staticmethod
    def _compute_reward(
        prev: RogueStatus,
        curr: RogueStatus,
        new_tiles: int,
        new_visit: bool,
        wasted_turn: bool,
        terminated: bool,
    ) -> float:
        reward = -0.002
        if wasted_turn:
            reward -= 0.05
        if new_visit:
            reward += 0.02
        reward += 0.05 * max(0, curr.gold - prev.gold)
        reward += 0.1 * max(0, curr.exp - prev.exp)
        reward += 50.0 * max(0, curr.level - prev.level)
        reward += 0.02 * new_tiles
        if terminated:
            reward -= 10.0
        return float(reward)

    def _build_obs(
        self,
        screen_arr: np.ndarray,
        player_pos: Optional[tuple[int, int]],
        status: RogueStatus,
    ) -> dict[str, np.ndarray]:
        return {
            "screen": screen_arr,
            "local": extract_local_crop(screen_arr, player_pos, radius=5),
            "stats": status.to_array(),
            "action_mask": self._compute_action_mask(screen_arr, player_pos),
        }

    def _build_info(
        self, player_pos: Optional[tuple[int, int]], messages: list[str]
    ) -> dict[str, Any]:
        return {
            "steps": self._steps,
            "level": self._status.level,
            "gold": self._status.gold,
            "hp": self._status.hp,
            "max_hp": self._status.max_hp,
            "exp": self._status.exp,
            "player_pos": player_pos,
            "messages": messages,
        }
