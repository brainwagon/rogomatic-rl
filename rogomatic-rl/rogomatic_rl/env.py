"""Gymnasium environment wrapping Rogue 5.4 via a 24x80 PTY."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .parser import (
    RogueStatus,
    extract_local_crop,
    find_player,
    is_death_or_end_screen,
    parse_killer,
    parse_status,
    screen_to_array,
)
from .pty_client import DEFAULT_ROGUE_BIN, DEFAULT_SCORE_FILE, RoguePTYClient


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

# Prompt at the bottom of Rogue's multi-line inventory overlay (things.c).
_OVERLAY_PROMPT = "--Press space to continue--"
# Pack-add message for food, e.g. "2 rations of food (a)", "a slime-mold (c)".
_FOOD_PICKUP_RE = re.compile(r"(?:food|slime-mold)\s*\([a-z]\)")
# Messages printed by eat() in misc.c when food is actually consumed.
_ATE_RE = re.compile(r"tasted good|tastes awful|that was a yummy")
_SLOT_RE = re.compile(r"^\s*([a-z])\) ")


class RogueEnv(gym.Env[dict[str, np.ndarray], int]):
    """Gymnasium environment for Rogue 5.4."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 15}

    def __init__(
        self,
        rogue_bin: str | Path = DEFAULT_ROGUE_BIN,
        max_steps: int = 2000,
        render_mode: Optional[str] = None,
        score_file: Optional[str | Path] = None,
        fixed_dungeons: bool = False,
    ) -> None:
        super().__init__()
        self.rogue_bin = Path(rogue_bin)
        self.max_steps = max_steps
        self.render_mode = render_mode
        # When set, reset(seed=n) plays Rogue's dungeon number n, so the
        # same seed gives the same dungeon (used by fixed evaluation).
        # Off by default: training dungeons stay random.
        self.fixed_dungeons = fixed_dungeons

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

        # With a score_file, finished games are posted to that shared Rogue
        # high score list; without one, scores are discarded.
        self.client = RoguePTYClient(rogue_bin=self.rogue_bin, score_file=score_file)
        self._steps: int = 0
        self._status: RogueStatus = RogueStatus()
        self._last_lines: list[str] = [" " * 80 for _ in range(24)]
        self._last_screen: np.ndarray = np.full((24, 80), ord(" "), dtype=np.uint8)
        self._last_pos: Optional[tuple[int, int]] = None
        self._prev_pos: Optional[tuple[int, int]] = None
        self._tile_under_player: str = "."
        self._seen_mask: np.ndarray = np.zeros((22, 80), dtype=bool)
        self._visited_mask: np.ndarray = np.zeros((22, 80), dtype=bool)
        self._visit_counts: np.ndarray = np.zeros((22, 80), dtype=np.int32)
        self._search_counts: np.ndarray = np.zeros((22, 80), dtype=np.int32)
        self._has_food: bool = True
        # How the current episode ended: None while it is running.
        self._end: Optional[str] = None
        self._killed_by: Optional[str] = None

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
        self._visit_counts.fill(0)
        self._search_counts.fill(0)
        self._prev_pos = None
        self._tile_under_player = "."
        self._has_food = True
        self._end = None
        self._killed_by = None

        messages = self.client.start(
            seed=seed if (self.fixed_dungeons and seed is not None) else None
        )
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
            self._visit_counts[self._last_pos[0] - 1, self._last_pos[1]] = 1

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
        prev_screen = self._last_screen
        parsed = parse_status(lines[23])
        if parsed is not None:
            self._status = parsed

        curr_pos = find_player(lines)
        screen_arr = screen_to_array(lines)
        msg_text = " ".join(messages).lower()

        if self._status.level != prev_status.level:
            self._seen_mask.fill(False)
            self._visited_mask.fill(False)
            self._visit_counts.fill(0)
            self._search_counts.fill(0)
            self._prev_pos = None
            self._tile_under_player = "."
        elif curr_pos is not None and curr_pos != prev_pos:
            self._prev_pos = prev_pos
            stepped_tile = chr(int(self._last_screen[curr_pos[0], curr_pos[1]]))
            if stepped_tile in ("%", "#", "+"):
                self._tile_under_player = stepped_tile
            elif stepped_tile in _ITEM_CHARS and (
                "moved onto" in msg_text or "no room" in msg_text
            ):
                self._tile_under_player = stepped_tile
            else:
                self._tile_under_player = "."
        elif action_idx == 11 and "no room" not in msg_text:
            self._tile_under_player = "."

        # Rogue picks items up automatically when the player walks onto
        # them, so watch every step's messages for food entering the pack.
        if _FOOD_PICKUP_RE.search(msg_text):
            self._has_food = True

        ate_food = False
        if (
            self._status.hungry_state >= 1
            and self._has_food
            and self.client.alive
            and not is_death_or_end_screen(lines)
        ):
            ate_food, eat_msgs = self._try_auto_eat()
            if eat_msgs:
                messages.extend(eat_msgs)
                lines = self.client.get_lines()
                self._last_lines = lines
                parsed_after_eat = parse_status(lines[23])
                if parsed_after_eat is not None:
                    self._status = parsed_after_eat
                curr_pos = find_player(lines)
                screen_arr = screen_to_array(lines)

        self._last_pos = curr_pos
        self._last_screen = screen_arr

        terminated = (
            not self.client.alive
            or is_death_or_end_screen(lines)
            or self._status.hp <= 0
        )
        if terminated and self._status.hp > 0:
            self._status = replace(self._status, hp=0)
        effective_max_steps = self.max_steps + 500 * max(0, self._status.level - 1)
        truncated = (not terminated) and (self._steps >= effective_max_steps)
        if terminated:
            self._end = "death"
            self._killed_by = parse_killer(lines)
        elif truncated:
            self._end = "time_limit"

        # On a terminated step the screen is the tombstone, not the map, so
        # it must not count as newly explored tiles.
        new_tiles = 0
        if not terminated:
            curr_mask = screen_arr[1:23] != 32
            new_tiles = int(np.count_nonzero(curr_mask & ~self._seen_mask))
            self._seen_mask |= curr_mask

        new_visit = False
        revisit_count = 0
        if not terminated and curr_pos is not None and 1 <= curr_pos[0] <= 22:
            r_idx, c_idx = curr_pos[0] - 1, curr_pos[1]
            if action_idx == 9:  # 's' search
                self._search_counts[r_idx, c_idx] += 1
            if not self._visited_mask[r_idx, c_idx]:
                self._visited_mask[r_idx, c_idx] = True
                self._visit_counts[r_idx, c_idx] = 1
                new_visit = True
            elif curr_pos != prev_pos:
                # Only arriving on a tile counts as a revisit; standing on
                # it to fight or rest does not.
                self._visit_counts[r_idx, c_idx] += 1
                revisit_count = int(self._visit_counts[r_idx, c_idx])

        wasted_turn = False
        if action_idx < len(_MOVE_DELTAS):
            wasted_turn = (
                prev_pos is not None
                and curr_pos == prev_pos
                and self._status.exp == prev_status.exp
                and self._status.hp == prev_status.hp
                and not terminated
                and not self._is_attack(prev_screen, prev_pos, action_idx)
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
            revisit_count=revisit_count,
            ate_food=ate_food,
        )
        obs = self._build_obs(screen_arr, curr_pos, self._status)
        info = self._build_info(curr_pos, messages)

        if self.render_mode == "human":
            self.render()

        # Post the finished game to the high score list (no-op without a
        # score file).  A time-limit episode ends with the player alive, so
        # it quits first and is recorded as "quit".
        if terminated or truncated:
            self.client.record_score(quit_game=truncated)

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

    def _try_auto_eat(self) -> tuple[bool, list[str]]:
        """Automatically eat a food ration from the pack when Hungry/Weak/Faint."""
        collected: list[str] = []
        collected.extend(self.client.send_key("e"))
        list_msgs = self.client.send_key("*")
        collected.extend(list_msgs)

        list_text = " ".join(list_msgs).lower()
        if "nothing appropriate" in list_text or "empty handed" in list_text:
            self._has_food = False
            return False, collected

        # With two or more food stacks Rogue draws an overlay and waits for
        # a space; with exactly one it prints the item as a normal message.
        slot = self._overlay_slot(self.client.get_lines())
        if slot is None:
            for m in list_msgs:
                match = _SLOT_RE.match(m)
                if match:
                    slot = match.group(1)
                    break
        collected.extend(self._dismiss_overlay())

        ate = False
        if slot is not None:
            eat_msgs = self.client.send_key(slot)
            collected.extend(eat_msgs)
            ate = bool(_ATE_RE.search(" ".join(eat_msgs).lower()))

        # Leave the game at the command prompt whatever happened above.
        collected.extend(self._dismiss_overlay())
        if "eat what" in self.client.get_lines()[0].lower():
            collected.extend(self.client.send_key("\x1b"))
        if not ate:
            self._has_food = False
        return ate, collected

    @staticmethod
    def _overlay_slot(lines: list[str]) -> Optional[str]:
        """Return the first pack letter listed in an inventory overlay, if any."""
        for row, line in enumerate(lines):
            col = line.find(_OVERLAY_PROMPT)
            if col == -1:
                continue
            # Item lines sit above the prompt, starting at the same column.
            for item_line in lines[:row]:
                match = _SLOT_RE.match(item_line[col:])
                if match:
                    return match.group(1)
            return None
        return None

    def _dismiss_overlay(self) -> list[str]:
        """Send a space if an inventory overlay is waiting for one."""
        if any(_OVERLAY_PROMPT in line for line in self.client.get_lines()):
            return self.client.send_key(" ")
        return []

    @staticmethod
    def _is_attack(
        prev_screen: np.ndarray,
        prev_pos: Optional[tuple[int, int]],
        action_idx: int,
    ) -> bool:
        """True if a move action was aimed at a tile showing a monster."""
        if prev_pos is None or action_idx >= len(_MOVE_DELTAS):
            return False
        dr, dc = _MOVE_DELTAS[action_idx]
        nr, nc = prev_pos[0] + dr, prev_pos[1] + dc
        if not (1 <= nr <= 22 and 0 <= nc < 80):
            return False
        return "A" <= chr(int(prev_screen[nr, nc])) <= "Z"

    def _compute_action_mask(
        self, screen_arr: np.ndarray, player_pos: Optional[tuple[int, int]]
    ) -> np.ndarray:
        mask = np.ones(len(ACTIONS), dtype=np.uint8)
        if player_pos is not None:
            pr, pc = player_pos
            monster_adjacent = False
            searchable_adjacent = False

            for idx, (dr, dc) in enumerate(_MOVE_DELTAS):
                nr, nc = pr + dr, pc + dc
                if not (1 <= nr <= 22 and 0 <= nc < 80):
                    mask[idx] = 0
                else:
                    ch = chr(int(screen_arr[nr, nc]))
                    if "A" <= ch <= "Z":
                        monster_adjacent = True
                    if ch in ("-", "|") or (
                        ch == " "
                        and (dr == 0 or dc == 0)
                        and self._tile_under_player in ("#", "+")
                    ):
                        searchable_adjacent = True

                    if ch in _WALL_CHARS:
                        mask[idx] = 0
                    elif dr != 0 and dc != 0:
                        orth_r = chr(int(screen_arr[nr, pc]))
                        orth_c = chr(int(screen_arr[pr, nc]))
                        if orth_r in _WALL_CHARS or orth_c in _WALL_CHARS:
                            mask[idx] = 0

            # Fight back: when an adjacent monster can be attacked, attacking
            # is the only legal action.  The move toward it must have survived
            # the wall and diagonal rules above, as Rogue refuses a diagonal
            # attack past a wall.  With several targets the policy chooses.
            attack_moves = [
                idx
                for idx, (dr, dc) in enumerate(_MOVE_DELTAS)
                if mask[idx] == 1
                and 1 <= pr + dr <= 22
                and 0 <= pc + dc < 80
                and "A" <= chr(int(screen_arr[pr + dr, pc + dc])) <= "Z"
            ]
            if attack_moves:
                mask[:] = 0
                mask[attack_moves] = 1
                return mask

            # Mask '.' (rest) when at full HP with no adjacent monster
            if self._status.hp >= self._status.max_hp and not monster_adjacent:
                mask[8] = 0

            # Mask 's' (search) unless next to a wall / corridor dead-end and searched < 6 times
            curr_searches = (
                int(self._search_counts[pr - 1, pc]) if 1 <= pr <= 22 else 0
            )
            if not searchable_adjacent or curr_searches >= 6:
                mask[9] = 0

            # Anti-oscillation & dead-end / room-corner escape when not in combat
            if not monster_adjacent and 1 <= pr <= 22:
                valid_moves = [
                    (idx, pr + dr, pc + dc)
                    for idx, (dr, dc) in enumerate(_MOVE_DELTAS)
                    if mask[idx] == 1
                ]

                # 1. Corridor / doorway forward flow: avoid immediate U-turns unless at a dead end
                if (
                    len(valid_moves) > 1
                    and self._prev_pos is not None
                    and self._tile_under_player in ("#", "+")
                ):
                    for idx, nr, nc in valid_moves:
                        if (nr, nc) == self._prev_pos:
                            mask[idx] = 0

                # 2. Room corner / revisit gradient escape: steer toward least-visited
                # neighbor tiles (treating stairs/items as 0 so they are always reachable)
                remaining_moves = [
                    (idx, nr, nc) for idx, nr, nc in valid_moves if mask[idx] == 1
                ]
                if len(remaining_moves) > 1:
                    eff_visits: list[tuple[int, int]] = []
                    for idx, nr, nc in remaining_moves:
                        ch = chr(int(screen_arr[nr, nc]))
                        if ch == "%" or ch in _ITEM_CHARS:
                            eff_v = 0
                        else:
                            eff_v = int(self._visit_counts[nr - 1, nc])
                        eff_visits.append((idx, eff_v))
                    min_v = min(v for _, v in eff_visits)
                    for idx, eff_v in eff_visits:
                        if eff_v > min_v:
                            mask[idx] = 0

        if self._tile_under_player != "%":
            mask[10] = 0  # '>' only valid on stairs
        if self._tile_under_player not in _ITEM_CHARS:
            mask[11] = 0  # ',' only valid on items

        if np.count_nonzero(mask) == 0:
            mask[8] = 1

        return mask

    @staticmethod
    def _compute_reward(
        prev: RogueStatus,
        curr: RogueStatus,
        new_tiles: int,
        new_visit: bool,
        wasted_turn: bool,
        terminated: bool,
        revisit_count: int = 0,
        ate_food: bool = False,
    ) -> float:
        reward = -0.002
        if wasted_turn:
            reward -= 0.05
        if new_visit:
            reward += 0.02
        elif revisit_count >= 3:
            reward -= 0.005 * min(6, revisit_count - 2)
        if ate_food:
            reward += 2.0
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
        action_mask = self._compute_action_mask(screen_arr, player_pos)
        obs_screen = screen_arr.copy()
        dungeon_slice = obs_screen[1:23]
        visited_floor = self._visited_mask & (dungeon_slice == ord("."))
        visited_corr = self._visited_mask & (dungeon_slice == ord("#"))
        dungeon_slice[visited_floor] = ord("~")
        dungeon_slice[visited_corr] = ord("`")
        return {
            "screen": obs_screen,
            "local": extract_local_crop(obs_screen, player_pos, radius=5),
            "stats": status.to_array(),
            "action_mask": action_mask,
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
            "exp_level": self._status.exp_level,
            # "death" or "time_limit" on the final step, None before it.
            "end": self._end,
            # The killer named on the tombstone, when the game ended in death.
            "killed_by": self._killed_by,
            "player_pos": player_pos,
            "messages": messages,
        }
