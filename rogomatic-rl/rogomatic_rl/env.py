"""Gymnasium environment wrapping Rogue 5.4 via a 24x80 PTY."""

from __future__ import annotations

from collections import deque
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
# What the agent walks over to collect before descending: items and gold.
_COLLECT_CHARS = _ITEM_CHARS | {"*"}
_COLLECT_CODES = np.array(sorted(ord(c) for c in _COLLECT_CHARS), dtype=np.uint8)

# Reward terms.  A descent and a death are the same size, so reaching the
# next level is not worth dying for, and every term is within about three
# orders of magnitude of the per-step ones (a descent used to be +50).
REWARD_DESCENT = 10.0  # per dungeon level gained
REWARD_DEATH = -10.0
# Health shaping: the reward tracks the change in hp / max_hp, so going
# from full health to none costs this much and healing earns it back.
# Being a pure difference, it cannot be farmed by getting hurt and resting.
REWARD_HEALTH = 5.0
REWARD_EXP_POINT = 0.2  # per experience point gained ...
REWARD_EXP_STEP_CAP = 5.0  # ... up to this much on one step
REWARD_EXP_LEVEL = 3.0  # per experience level gained

# Prompt at the bottom of Rogue's multi-line inventory overlay (things.c).
_OVERLAY_PROMPT = "--Press space to continue--"
# Pack-add message for food, e.g. "2 rations of food (a)", "a slime-mold (c)".
_FOOD_PICKUP_RE = re.compile(r"(?:food|slime-mold)\s*\([a-z]\)")
# Pack-add message for any item: "<inv_name> (x)" (add_pack in pack.c).
_PICKUP_RE = re.compile(r"^(?!wielding\b|was wearing\b|wearing\b).+\([a-z]\)\s*$", re.IGNORECASE)
# Messages printed by eat() in misc.c when food is actually consumed.
_ATE_RE = re.compile(r"tasted good|tastes awful|that was a yummy")
_SLOT_RE = re.compile(r"^\s*([a-z])\) ")
_ITEM_LINE_RE = re.compile(r"^\s*([a-z])\)\s+(.+?)\s*$")
_WEAPON_BONUS_RE = re.compile(r"^an?\s+(?:([+-]\d+),([+-]\d+)\s+)?(.+?)(?:\s+called\s+.+)?$", re.IGNORECASE)

# Mean melee damage by weapon name (init_dam in weapons.c).
# Ranged/thrown items (short bow, arrow, dart, shuriken) are omitted so they are never wielded.
_WEAPON_BASE_DAMAGE: dict[str, float] = {
    "two handed sword": 10.0,  # 4d4
    "long sword": 7.5,         # 3d4
    "mace": 5.0,               # 2d4
    "spear": 4.0,              # 2d3
    "dagger": 3.5,             # 1d6
}
# Starting weapon is a +1,+1 mace: 5.0 + 0.5*1 + 1.0 = 6.5.
_DEFAULT_WEAPON_SCORE = 6.5

_ARMOR_RE = re.compile(
    r"^(?:[+-]\d+\s+)?([a-z ]+?)(?:\s+\[(?:protection\s+)?(-?\d+)\])?(?:\s+called\s+.+)?$",
    re.IGNORECASE,
)
_WEARING_MSG_RE = re.compile(r"wearing\s+(.+)$", re.IGNORECASE)
# Base Arm rating (10 - a_class, higher is better; extern.c).
_ARMOR_BASE_VALUE: dict[str, int] = {
    "leather armor": 2,
    "ring mail": 3,
    "studded leather armor": 3,
    "scale mail": 4,
    "chain mail": 5,
    "splint mail": 6,
    "banded mail": 6,
    "plate mail": 7,
}

_KNOWN_POTION_RE = re.compile(
    r"^(?:an?|(\d+))\s+potions?\s+of\s+([a-z ]+?)\(([a-z ]+)\)$",
    re.IGNORECASE,
)
_UNKNOWN_POTION_RE = re.compile(
    r"^(?:an?|(\d+))\s+([a-z ]+?)\s+potions?$",
    re.IGNORECASE,
)

_KNOWN_SCROLL_RE = re.compile(r"^(?:an?|(\d+))\s+scrolls?\s+of\s+(.+?)\s*$", re.IGNORECASE)
_TITLED_SCROLL_RE = re.compile(
    r"^(?:an?|(\d+))\s+scrolls?\s+titled\s+'(.*)'\s*$", re.IGNORECASE
)
_CALLED_SCROLL_RE = re.compile(
    r"^(?:an?|(\d+))\s+scrolls?\s+called\s+(.+?)\s*$", re.IGNORECASE
)
# Printed by read_scroll for every identify scroll before the prompt (scrolls.c).
_IDENTIFY_MSG_RE = re.compile(r"this scroll is an (identify [a-z ,]+?) scroll")
# Known scrolls read as soon as no monster is in view.
_SCROLLS_READ_WHEN_SAFE = frozenset(
    {
        "enchant armor",
        "enchant weapon",
        "protect armor",
        "magic mapping",
        "monster confusion",
    }
)
# Known scrolls read only to escape a fight, in this order.
_SCROLLS_FOR_ESCAPE = ("teleportation", "hold monster")
# Scroll effects whose reading leaves oi_know false, so the env infers the
# kind from the message Rogue prints (scrolls.c).
_SCROLL_EFFECT_MESSAGES: tuple[tuple[str, str], ...] = (
    ("your armor glows", "enchant armor"),
    ("shimmering", "protect armor"),
    ("universal oneness", "remove curse"),
    ("somebody is watching over you", "remove curse"),
    ("maniacal laughter", "scare monster"),
    ("your hands begin to glow", "monster confusion"),
)
_WEAPON_GLOWS_RE = re.compile(r"your .+ glows blue")
# Upper bound on keys sent at one `identify what?` prompt. Rogue loops on ESC
# there, so the env keeps offering items until one is accepted.
_MAX_IDENTIFY_KEYS = 30

_KNOWN_RING_RE = re.compile(
    r"^(?:an?|(\d+))\s+rings?\s+of\s+([a-z ]+?)(?:\s+\[([+-]?\d+)\])?\(([a-z ]+)\)$",
    re.IGNORECASE,
)
_UNKNOWN_RING_RE = re.compile(
    r"^(?:an?|(\d+))\s+([a-z ]+?)\s+rings?$",
    re.IGNORECASE,
)
_RING_BASE_SCORE: dict[str, float] = {
    "regeneration": 10.0,
    "slow digestion": 9.0,
    "maintain armor": 4.5,
    "sustain strength": 4.0,
    "see invisible": 3.5,
    "stealth": 3.0,
    "searching": 2.5,
    "adornment": -5.0,
    "aggravate monster": -5.0,
    "teleportation": -5.0,
}
_NUMERIC_RING_TIEBREAK: dict[str, float] = {
    "increase damage": 0.3,
    "dexterity": 0.2,
    "protection": 0.1,
    "add strength": 0.0,
}

_MISSILE_RE = re.compile(
    r"^(?:an?|(\d+))\s+((?:[+-]\d+,[+-]\d+\s+)?)(shuriken|spear|dagger|dart|arrow)s?(?:\s+called\s+.+)?$",
    re.IGNORECASE,
)
_MISSILE_BONUS_RE = re.compile(r"^([+-]\d+),([+-]\d+)\s*$")
# Mean thrown damage when wielding a melee weapon (init_dam in weapons.c, roll_em in fight.c).
_MISSILE_BASE_DAMAGE: dict[str, float] = {
    "shuriken": 5.0,  # 2d4
    "spear": 3.5,     # 1d6
    "dagger": 2.5,    # 1d4
    "dart": 2.0,      # 1d3
    "arrow": 1.0,     # 1d1 without short bow in hand
}
_DEFAULT_PACK: dict[str, str] = {
    "a": "Some food",
    "b": "+1 ring mail [4] (being worn)",
    "c": "A +1,+1 mace (weapon in hand)",
    "d": "A +1,+0 short bow",
    "e": "25 +0,+0 arrows",
}





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
        self._seen_floor_mask: np.ndarray = np.zeros((22, 80), dtype=bool)
        self._visited_mask: np.ndarray = np.zeros((22, 80), dtype=bool)
        self._visit_counts: np.ndarray = np.zeros((22, 80), dtype=np.int32)
        self._search_counts: np.ndarray = np.zeros((22, 80), dtype=np.int32)
        # Item tiles (screen row, col) already stepped on this level.  An
        # item still drawn there could not be picked up (full pack), so it
        # is never steered to again.
        self._item_skip: set[tuple[int, int]] = set()
        self._has_food: bool = True
        self._pack: dict[str, str] = {}
        self._inventory_dirty: bool = False
        self._weapon_score: float = _DEFAULT_WEAPON_SCORE
        self._weapon_cursed: bool = False
        self._armor_cursed: bool = False
        self._ring_cursed: dict[str, bool] = {"left": False, "right": False}
        self._known_ring_stones: dict[str, tuple[str, Optional[int]]] = {}
        # Stones of unidentified rings found cursed (so harmful) or found
        # removable (so safe). Rogue never names a ring when it is worn.
        self._bad_ring_stones: set[str] = set()
        self._safe_ring_stones: set[str] = set()
        self._tried_potion_colors: set[str] = set()
        self._known_potion_colors: dict[str, str] = {}
        self._tried_scroll_keys: set[str] = set()
        self._known_scroll_titles: dict[str, str] = {}
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
        self._seen_floor_mask.fill(False)
        self._visited_mask.fill(False)
        self._visit_counts.fill(0)
        self._search_counts.fill(0)
        self._item_skip.clear()
        self._prev_pos = None
        self._tile_under_player = "."
        self._has_food = True
        self._pack = dict(_DEFAULT_PACK)
        self._inventory_dirty = False
        self._weapon_score = _DEFAULT_WEAPON_SCORE
        self._weapon_cursed = False
        self._armor_cursed = False
        self._ring_cursed = {"left": False, "right": False}
        self._known_ring_stones = {}
        self._bad_ring_stones = set()
        self._safe_ring_stones = set()
        self._tried_potion_colors = set()
        self._known_potion_colors = {}
        self._tried_scroll_keys = set()
        self._known_scroll_titles = {}
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
        self._seen_floor_mask |= screen_arr[1:23] == ord(".")
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
            self._seen_floor_mask.fill(False)
            self._visited_mask.fill(False)
            self._visit_counts.fill(0)
            self._search_counts.fill(0)
            self._item_skip.clear()
            self._prev_pos = None
            self._tile_under_player = "."
        elif curr_pos is not None and curr_pos != prev_pos:
            self._prev_pos = prev_pos
            stepped_tile = chr(int(self._last_screen[curr_pos[0], curr_pos[1]]))
            if stepped_tile in _COLLECT_CHARS:
                self._item_skip.add(curr_pos)
            picked_up_or_stepped_item = (
                any(_PICKUP_RE.search(m) for m in messages)
                or "gold pieces" in msg_text
                or "moved onto" in msg_text
                or "no room" in msg_text
            )
            if stepped_tile in ("#", "+"):
                self._tile_under_player = stepped_tile
            elif stepped_tile == "%" and not picked_up_or_stepped_item:
                self._tile_under_player = "%"
            elif (stepped_tile in _ITEM_CHARS or stepped_tile == "*") and not messages:
                # Under hallucination, an unseen staircase is drawn as a random
                # item/gold glyph (daemons.c:260), so stepping onto it produces
                # no pickup or "moved onto" message.
                self._tile_under_player = "%"
            elif stepped_tile in _ITEM_CHARS and (
                "moved onto" in msg_text or "no room" in msg_text
            ):
                self._tile_under_player = stepped_tile
            else:
                self._tile_under_player = "."
        elif action_idx == 10 and self._status.level == prev_status.level:
            self._tile_under_player = "."
        elif action_idx == 11 and "no room" not in msg_text:
            self._tile_under_player = "."

        # Rogue picks items up automatically when the player walks onto
        # them, so watch every step's messages for food entering the pack.
        if _FOOD_PICKUP_RE.search(msg_text):
            self._has_food = True
        if any(_PICKUP_RE.search(m) for m in messages):
            self._inventory_dirty = True

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

        if self.client.alive and not is_death_or_end_screen(lines):
            auto_msgs = self._run_auto_actions()
            if auto_msgs:
                messages.extend(auto_msgs)
                lines = self.client.get_lines()
                self._last_lines = lines
                parsed_after_auto = parse_status(lines[23])
                if parsed_after_auto is not None:
                    self._status = parsed_after_auto
                curr_pos = find_player(lines)
                screen_arr = screen_to_array(lines)

        self._last_pos = curr_pos

        terminated = (
            not self.client.alive
            or is_death_or_end_screen(lines)
            or self._status.hp <= 0
        )
        if terminated and self._status.hp > 0:
            self._status = replace(self._status, hp=0)
        per_level_steps = 500 if self.max_steps >= 500 else 0
        effective_max_steps = self.max_steps + per_level_steps * max(
            0, self._status.level - 1
        )
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
            self._seen_floor_mask |= screen_arr[1:23] == ord(".")
            erased_floor = (screen_arr[1:23] == ord(" ")) & self._seen_floor_mask
            screen_arr[1:23][erased_floor] = ord(".")
            curr_mask = screen_arr[1:23] != 32
            new_tiles = int(np.count_nonzero(curr_mask & ~self._seen_mask))
            self._seen_mask |= curr_mask

        self._last_screen = screen_arr

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
    def _overlay_items(lines: list[str], list_msgs: list[str]) -> dict[str, str]:
        """Extract all 'x) description' pack items from an overlay or single-line message."""
        items: dict[str, str] = {}
        for row, line in enumerate(lines):
            col = line.find(_OVERLAY_PROMPT)
            if col == -1:
                continue
            for item_line in lines[:row]:
                match = _ITEM_LINE_RE.match(item_line[col:])
                if match:
                    items[match.group(1)] = match.group(2)
            return items
        for m in list_msgs:
            match = _ITEM_LINE_RE.match(m)
            if match:
                items[match.group(1)] = match.group(2)
        return items

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

    def _refresh_inventory(self) -> list[str]:
        """Query the full pack with 'i' (zero game turns) and update cached pack state."""
        collected: list[str] = []
        inv_msgs = self.client.send_key("i")
        collected.extend(inv_msgs)
        self._pack = self._overlay_items(self.client.get_lines(), inv_msgs)
        collected.extend(self._dismiss_overlay())
        self._inventory_dirty = False
        for desc in self._pack.values():
            if desc.endswith("(weapon in hand)"):
                score = self._weapon_value(desc)
                if score is not None:
                    self._weapon_score = score
        return collected

    @staticmethod
    def _weapon_value(desc: str) -> Optional[float]:
        """Score a melee weapon description (higher is better), or None if not a melee weapon."""
        clean = desc
        if clean.endswith("(weapon in hand)"):
            clean = clean[: -len("(weapon in hand)")].strip()
        match = _WEAPON_BONUS_RE.match(clean)
        if not match:
            return None
        hplus_s, dplus_s, name = match.group(1), match.group(2), match.group(3).lower()
        base = _WEAPON_BASE_DAMAGE.get(name)
        if base is None:
            return None
        hplus = int(hplus_s) if hplus_s is not None else 0
        dplus = int(dplus_s) if dplus_s is not None else 0
        return base + 0.5 * hplus + dplus

    @staticmethod
    def _armor_value(desc: str) -> Optional[int]:
        """Score an armor description on the Arm scale (10 - AC, higher is better), or None."""
        clean = desc
        if clean.endswith("(being worn)"):
            clean = clean[: -len("(being worn)")].strip()
        match = _ARMOR_RE.match(clean)
        if not match:
            return None
        name = match.group(1).strip().lower()
        base = _ARMOR_BASE_VALUE.get(name)
        if base is None:
            return None
        bracket = match.group(2)
        return int(bracket) if bracket is not None else base

    def _parse_potion(self, desc: str) -> Optional[tuple[int, Optional[str], str]]:
        """Parse a potion pack entry into (count, kind_or_None, color), or None."""
        m_known = _KNOWN_POTION_RE.match(desc.strip())
        if m_known:
            count = int(m_known.group(1)) if m_known.group(1) else 1
            kind = m_known.group(2).strip().lower()
            color = m_known.group(3).strip().lower()
            self._tried_potion_colors.add(color)
            self._known_potion_colors[color] = kind
            return count, kind, color
        m_unk = _UNKNOWN_POTION_RE.match(desc.strip())
        if m_unk:
            count = int(m_unk.group(1)) if m_unk.group(1) else 1
            color = m_unk.group(2).strip().lower()
            return count, self._known_potion_colors.get(color), color
        return None

    def _quaff_slot(self, slot: str, count: int, color: str) -> list[str]:
        """Quaff one potion from `slot`, dismiss any 'call it:' prompt, and update `_pack`."""
        collected: list[str] = []
        collected.extend(self.client.send_key("q"))
        quaff_msgs = self.client.send_key(slot)
        collected.extend(quaff_msgs)
        self._tried_potion_colors.add(color)

        qtext = " ".join(quaff_msgs).lower()
        inferred: Optional[str] = None
        if "begin to feel much better" in qtext:
            inferred = "extra healing"
        elif "begin to feel better" in qtext:
            inferred = "healing"
        elif "what bulging muscles" in qtext:
            inferred = "gain strength"
        elif "much more skillful" in qtext:
            inferred = "raise level"
        elif "feel warm all over" in qtext:
            inferred = "restore strength"
        if inferred is not None:
            self._known_potion_colors[color] = inferred

        top = self.client.get_lines()[0].lower()
        if "call it" in top or "quaff what" in top:
            collected.extend(self.client.send_key("\x1b"))

        if count <= 1:
            self._pack.pop(slot, None)
        else:
            rem = count - 1
            kind = self._known_potion_colors.get(color)
            if kind is not None:
                prefix = "A potion" if rem == 1 else f"{rem} potions"
                self._pack[slot] = f"{prefix} of {kind}({color})"
            else:
                prefix = "A" if rem == 1 else str(rem)
                noun = "potion" if rem == 1 else "potions"
                self._pack[slot] = f"{prefix} {color} {noun}"
        return collected

    def _try_heal_quaff(self) -> list[str]:
        """Drink a known healing potion when below 25% health (even in combat)."""
        if self._status.hp <= 0 or self._status.hp >= 0.25 * max(1, self._status.max_hp):
            return []
        collected: list[str] = []
        if self._inventory_dirty:
            collected.extend(self._refresh_inventory())
        for target_kind in ("extra healing", "healing"):
            for slot, desc in list(self._pack.items()):
                parsed = self._parse_potion(desc)
                if parsed is None:
                    continue
                count, kind, color = parsed
                if kind == target_kind:
                    collected.extend(self._quaff_slot(slot, count, color))
                    return collected
        return collected

    def _try_safe_quaff(self) -> list[str]:
        """Drink known buff potions or (at full health) unknown potions when no monster is in view."""
        for slot, desc in list(self._pack.items()):
            parsed = self._parse_potion(desc)
            if parsed is None:
                continue
            count, kind, color = parsed
            if kind in ("gain strength", "raise level"):
                return self._quaff_slot(slot, count, color)
            if (
                kind == "restore strength"
                and self._status.strength < self._status.max_strength
            ):
                return self._quaff_slot(slot, count, color)
            if (
                kind is None
                and color not in self._tried_potion_colors
                and self._status.hp >= self._status.max_hp
            ):
                return self._quaff_slot(slot, count, color)
        return []

    def _parse_scroll(self, desc: str) -> Optional[tuple[int, Optional[str], str]]:
        """Parse a scroll pack entry into (count, kind_or_None, key), or None.

        `key` names the scroll's title (or its user-given name) so repeated
        copies of an unidentified scroll are recognised as the same kind.
        """
        clean = desc.strip()
        match = _KNOWN_SCROLL_RE.match(clean)
        if match:
            kind = match.group(2).strip().lower()
            return (int(match.group(1)) if match.group(1) else 1, kind, f"of:{kind}")
        match = _TITLED_SCROLL_RE.match(clean)
        if match:
            key = f"titled:{match.group(2).strip().lower()}"
        else:
            match = _CALLED_SCROLL_RE.match(clean)
            if not match:
                return None
            key = f"called:{match.group(2).strip().lower()}"
        count = int(match.group(1)) if match.group(1) else 1
        return count, self._known_scroll_titles.get(key), key

    def _identify_class(self, desc: str) -> Optional[tuple[str, bool]]:
        """Classify a pack item for an identify scroll: (scroll name, unidentified)."""
        clean = desc.strip()
        potion = self._parse_potion(clean)
        if potion is not None:
            return "identify potion", potion[1] is None
        scroll = self._parse_scroll(clean)
        if scroll is not None:
            return "identify scroll", scroll[1] is None
        ring = self._ring_value(clean)
        if ring is not None:
            return "identify ring, wand or staff", ring[0] == "unknown"
        if re.search(r"\b(?:wand|staff)\b", clean, re.IGNORECASE):
            return "identify ring, wand or staff", " of " not in clean
        if (
            self._weapon_value(clean) is not None
            or self._missile_value(clean) is not None
        ):
            return "identify weapon", re.search(r"[+-]\d+,[+-]\d+", clean) is None
        if self._armor_value(clean) is not None:
            return "identify armor", re.match(r"^[+-]\d+\s", clean) is None
        return None

    def _identify_candidates(
        self, kind: Optional[str], read_slot: str, read_count: int
    ) -> tuple[list[str], list[str]]:
        """Pack slots to offer an identify scroll: (unidentified, already known)."""
        preferred: list[str] = []
        fallback: list[str] = []
        for slot, desc in sorted(self._pack.items()):
            if slot == read_slot and read_count <= 1:
                continue  # the scroll being read has already left the pack
            cls = self._identify_class(desc)
            if cls is None or (kind is not None and cls[0] != kind):
                continue
            (preferred if cls[1] else fallback).append(slot)
        return preferred, fallback

    @staticmethod
    def _infer_scroll_kind(text: str) -> Optional[str]:
        """Guess a scroll kind from its reading message (for kinds Rogue leaves unnamed)."""
        for fragment, kind in _SCROLL_EFFECT_MESSAGES:
            if fragment in text:
                return kind
        if _WEAPON_GLOWS_RE.search(text):
            return "enchant weapon"
        return None

    def _scroll_prompts(
        self, identify_kind: Optional[str], read_slot: str, read_count: int
    ) -> list[str]:
        """Answer the prompts a scroll can leave behind, so the game is back at its command prompt.

        Rogue re-asks `identify what?` after ESC (wizard.c, whatis with
        insist), so that prompt is answered with a real item; `*` is the
        only way out when the pack holds nothing of the asked type.
        """
        collected: list[str] = []
        queue: Optional[list[str]] = None
        for _ in range(_MAX_IDENTIFY_KEYS):
            top = self.client.get_lines()[0].lower()
            if "identify what" in top:
                if queue is None:
                    preferred, fallback = self._identify_candidates(
                        identify_kind, read_slot, read_count
                    )
                    chosen = preferred + fallback
                    rest = [s for s in sorted(self._pack) if s not in chosen]
                    queue = chosen + ["*"] + rest
                if not queue:
                    break
                collected.extend(self.client.send_key(queue.pop(0)))
                collected.extend(self._dismiss_overlay())
            elif "call it" in top or "read what" in top:
                collected.extend(self.client.send_key("\x1b"))
            else:
                break
        return collected

    def _read_scroll_slot(
        self, slot: str, desc: str, count: int, kind: Optional[str], key: str
    ) -> list[str]:
        """Read one scroll from `slot` and keep the pack and curse flags in step."""
        collected: list[str] = []
        collected.extend(self.client.send_key("r"))
        read_msgs = self.client.send_key(slot)
        collected.extend(read_msgs)
        self._tried_scroll_keys.add(key)

        text = " ".join(read_msgs).lower()
        identify_match = _IDENTIFY_MSG_RE.search(text)
        identify_kind = identify_match.group(1) if identify_match else None
        if kind is not None and kind.startswith("identify"):
            identify_kind = kind
        inferred = self._infer_scroll_kind(text)
        if kind is None and inferred is not None:
            self._known_scroll_titles[key] = inferred
        effect = kind or inferred
        collected.extend(self._scroll_prompts(identify_kind, slot, count))

        if effect == "remove curse":
            self._armor_cursed = False
            self._weapon_cursed = False
            self._ring_cursed = {"left": False, "right": False}
        elif effect in ("enchant armor", "protect armor"):
            self._armor_cursed = False
        elif effect == "enchant weapon":
            self._weapon_cursed = False

        if count <= 1:
            self._pack.pop(slot, None)
        else:
            noun = "A scroll" if count - 1 == 1 else f"{count - 1} scrolls"
            self._pack[slot] = re.sub(
                r"^(?:an?|\d+)\s+scrolls?", noun, desc.strip(), count=1, flags=re.IGNORECASE
            )
        self._inventory_dirty = True
        return collected

    def _should_read_known_scroll(
        self, kind: str, slot: str, count: int
    ) -> bool:
        """True if a known scroll of `kind` is worth reading right now (no monster in view)."""
        if kind in ("enchant armor", "protect armor"):
            return any(d.endswith("(being worn)") for d in self._pack.values())
        if kind == "enchant weapon":
            return any(d.endswith("(weapon in hand)") for d in self._pack.values())
        if kind in _SCROLLS_READ_WHEN_SAFE:
            return True
        if kind == "remove curse":
            return (
                self._armor_cursed
                or self._weapon_cursed
                or any(self._ring_cursed.values())
            )
        if kind.startswith("identify"):
            return bool(self._identify_candidates(kind, slot, count)[0])
        return False

    def _try_read_scrolls(self) -> list[str]:
        """Read one scroll when no monster is in view: known useful ones, or each unknown title once."""
        for slot, desc in list(self._pack.items()):
            parsed = self._parse_scroll(desc)
            if parsed is None:
                continue
            count, kind, key = parsed
            if kind is None:
                if (
                    key not in self._tried_scroll_keys
                    and self._status.hp >= self._status.max_hp
                ):
                    return self._read_scroll_slot(slot, desc, count, kind, key)
            elif self._should_read_known_scroll(kind, slot, count):
                return self._read_scroll_slot(slot, desc, count, kind, key)
        return []

    def _try_emergency_scroll(self) -> list[str]:
        """Read a known teleportation or hold monster scroll when nearly dead beside a monster."""
        parsed_status = parse_status(self.client.get_lines()[23])
        status = parsed_status if parsed_status is not None else self._status
        if status.hp <= 0 or status.hp >= 0.25 * max(1, status.max_hp):
            return []
        lines = self.client.get_lines()
        pos = find_player(lines)
        if pos is None or not any(
            "A" <= lines[r][c] <= "Z"
            for r in range(max(1, pos[0] - 1), min(22, pos[0] + 1) + 1)
            for c in range(max(0, pos[1] - 1), min(len(lines[r]) - 1, pos[1] + 1) + 1)
        ):
            return []
        collected: list[str] = []
        if self._inventory_dirty:
            collected.extend(self._refresh_inventory())
        for target in _SCROLLS_FOR_ESCAPE:
            for slot, desc in list(self._pack.items()):
                scroll = self._parse_scroll(desc)
                if scroll is not None and scroll[1] == target:
                    collected.extend(
                        self._read_scroll_slot(slot, desc, scroll[0], target, scroll[2])
                    )
                    return collected
        return collected

    @staticmethod
    def _missile_value(desc: str) -> Optional[tuple[int, float]]:
        """Parse a throwable missile pack entry into (count, score), or None."""
        clean = desc.strip()
        if clean.endswith("(weapon in hand)") or clean.endswith("(being worn)"):
            return None
        match = _MISSILE_RE.match(clean)
        if not match:
            return None
        count = int(match.group(1)) if match.group(1) else 1
        bonus_str = match.group(2)
        kind = match.group(3).lower()
        base = _MISSILE_BASE_DAMAGE.get(kind)
        if base is None:
            return None
        hplus = 0
        dplus = 0
        if bonus_str:
            bmatch = _MISSILE_BONUS_RE.match(bonus_str)
            if bmatch:
                hplus = int(bmatch.group(1))
                dplus = int(bmatch.group(2))
        return count, base + 0.5 * hplus + dplus

    @staticmethod
    def _find_missile_target(
        screen_arr: np.ndarray, player_pos: Optional[tuple[int, int]]
    ) -> Optional[str]:
        """Return the direction key toward the nearest unobstructed aligned monster at dist >= 2."""
        if player_pos is None:
            return None
        pr, pc = player_pos
        for dr, dc in _MOVE_DELTAS:
            nr, nc = pr + dr, pc + dc
            if 1 <= nr <= 22 and 0 <= nc < 80:
                if "A" <= chr(int(screen_arr[nr, nc])) <= "Z":
                    return None

        best_dist: Optional[int] = None
        best_dir: Optional[str] = None
        for idx, (dr, dc) in enumerate(_MOVE_DELTAS):
            k = 1
            while True:
                nr, nc = pr + k * dr, pc + k * dc
                if not (1 <= nr <= 22 and 0 <= nc < 80):
                    break
                ch = chr(int(screen_arr[nr, nc]))
                if ch in _WALL_CHARS or ch == "+":
                    break
                if "A" <= ch <= "Z":
                    if k >= 2 and (best_dist is None or k < best_dist):
                        best_dist = k
                        best_dir = ACTIONS[idx]
                    break
                k += 1
        return best_dir

    def _try_throw_missile(self) -> list[str]:
        """Throw the highest-scored missile at an aligned monster at range >= 2."""
        lines = self.client.get_lines()
        player_pos = find_player(lines)
        screen_arr = screen_to_array(lines)
        erased_floor = (screen_arr[1:23] == ord(" ")) & self._seen_floor_mask
        screen_arr[1:23][erased_floor] = ord(".")
        dir_key = self._find_missile_target(screen_arr, player_pos)
        if dir_key is None:
            return []

        collected: list[str] = []
        if self._inventory_dirty:
            collected.extend(self._refresh_inventory())

        best_slot: Optional[str] = None
        best_count: int = 0
        best_score: float = -1e9
        for slot, desc in self._pack.items():
            parsed = self._missile_value(desc)
            if parsed is None:
                continue
            count, score = parsed
            if score > best_score:
                best_score = score
                best_slot = slot
                best_count = count

        if best_slot is None:
            return collected

        desc = self._pack[best_slot]
        collected.extend(self.client.send_key(f"t{dir_key}"))
        throw_msgs = self.client.send_key(best_slot)
        collected.extend(throw_msgs)
        top = self.client.get_lines()[0].lower()
        if "throw what" in top or "direction" in top:
            collected.extend(self.client.send_key("\x1b"))
            self._pack.pop(best_slot, None)
            return collected

        if best_count <= 1:
            self._pack.pop(best_slot, None)
        else:
            rem = best_count - 1
            match = _MISSILE_RE.match(desc.strip())
            if match:
                bonus_part = match.group(2) or ""
                kind = match.group(3).lower()
                if rem == 1:
                    article = "An" if kind[0] in "aeiou" else "A"
                    self._pack[best_slot] = f"{article} {bonus_part}{kind}"
                else:
                    self._pack[best_slot] = f"{rem} {bonus_part}{kind}s"
            else:
                self._pack.pop(best_slot, None)
        return collected

    @staticmethod
    def _monster_in_view(lines: list[str]) -> bool:
        """True if any monster letter ('A'..'Z') is visible on the dungeon map."""
        return any("A" <= ch <= "Z" for row in lines[1:23] for ch in row)

    def _run_auto_actions(self) -> list[str]:
        """Run scripted inventory refresh and auto-equip/quaff/read/missile/ring actions."""
        collected: list[str] = []
        collected.extend(self._try_heal_quaff())
        if self._monster_in_view(self.client.get_lines()):
            escaped = self._try_emergency_scroll()
            collected.extend(escaped)
            if not escaped:
                collected.extend(self._try_throw_missile())
            return collected
        if self._inventory_dirty:
            collected.extend(self._refresh_inventory())
        collected.extend(self._try_auto_wield())
        collected.extend(self._try_auto_wear())
        collected.extend(self._try_auto_rings())
        quaffed = self._try_safe_quaff()
        collected.extend(quaffed)
        if not quaffed:
            collected.extend(self._try_read_scrolls())
        return collected

    def _ring_value(
        self, desc: str
    ) -> Optional[tuple[str, str, Optional[str], float]]:
        """Parse a ring pack entry into (kind, stone, hand, score), or None."""
        clean = desc.strip()
        hand: Optional[str] = None
        if clean.endswith("(on left hand)"):
            hand = "left"
            clean = clean[: -len("(on left hand)")].strip()
        elif clean.endswith("(on right hand)"):
            hand = "right"
            clean = clean[: -len("(on right hand)")].strip()
        elif clean.endswith("(being worn)") or clean.endswith("(weapon in hand)"):
            return None

        bonus: Optional[int] = None
        match = _KNOWN_RING_RE.match(clean)
        if match:
            kind = match.group(2).strip().lower()
            if match.group(3) is not None:
                bonus = int(match.group(3))
            stone = match.group(4).strip().lower()
        else:
            umatch = _UNKNOWN_RING_RE.match(clean)
            if not umatch:
                return None
            stone = umatch.group(2).strip().lower()
            known = self._known_ring_stones.get(stone)
            if known is not None:
                kind, bonus = known
            else:
                kind = "unknown"

        if kind == "unknown":
            # About a quarter of rings are cursed and cannot be taken off
            # (aggravate monster, teleportation, and a third of the numeric
            # ones), so an unidentified ring is worn only with a remove curse
            # scroll in the pack, or once its stone is known to be removable.
            if stone in self._bad_ring_stones:
                score = -5.0
            elif self._status.hungry_state != 0:
                score = 0.0
            elif stone in self._safe_ring_stones or self._has_remove_curse():
                score = 0.5
            else:
                score = 0.0
        elif kind in _NUMERIC_RING_TIEBREAK:
            if bonus is not None:
                if bonus <= 0:
                    score = -5.0
                else:
                    score = 4.0 + 2.0 * bonus + _NUMERIC_RING_TIEBREAK[kind]
            else:
                score = 5.0 + _NUMERIC_RING_TIEBREAK[kind]
        else:
            score = _RING_BASE_SCORE.get(kind, 0.0)
        return kind, stone, hand, score

    def _has_remove_curse(self) -> bool:
        """True if the pack holds a known scroll of remove curse."""
        for desc in self._pack.values():
            scroll = self._parse_scroll(desc)
            if scroll is not None and scroll[1] == "remove curse":
                return True
        return False

    def _worn_ring_stone(self, hand: str) -> Optional[str]:
        """The stone of the ring on `hand`, from the pack, or None."""
        for desc in self._pack.values():
            parsed = self._ring_value(desc)
            if parsed is not None and parsed[2] == hand:
                return parsed[1]
        return None

    def _remove_ring(
        self, hand: str, slot: str, both_worn: bool
    ) -> tuple[bool, list[str]]:
        """Attempt to remove the ring on `hand`; returns (removed, messages)."""
        collected: list[str] = []
        r_msgs = self.client.send_key("R")
        collected.extend(r_msgs)
        top = self.client.get_lines()[0].lower()
        if both_worn or "or right" in top:
            hand_msgs = self.client.send_key("l" if hand == "left" else "r")
            collected.extend(hand_msgs)
            msg_text = " ".join(hand_msgs).lower()
        else:
            msg_text = " ".join(r_msgs).lower()

        if "appears to be cursed" in msg_text:
            self._ring_cursed[hand] = True
            # Only harmful rings are cursed, so remember the stone as bad.
            stone = self._worn_ring_stone(hand)
            if stone is not None:
                self._bad_ring_stones.add(stone)
            return False, collected

        if slot in self._pack:
            suffix = f"(on {hand} hand)"
            d = self._pack[slot]
            if d.endswith(suffix):
                self._pack[slot] = d[: -len(suffix)].strip()
        return True, collected

    def _put_on_ring(
        self, slot: str, stone: str, target_hand: str, both_empty: bool
    ) -> tuple[bool, list[str]]:
        """Put the ring in `slot` onto `target_hand`; returns (put_on, messages)."""
        prev_str = self._status.strength
        collected: list[str] = []
        collected.extend(self.client.send_key("P"))
        slot_msgs = self.client.send_key(slot)
        collected.extend(slot_msgs)
        top = self.client.get_lines()[0].lower()
        if both_empty or "or right" in top:
            hand_msgs = self.client.send_key("l" if target_hand == "left" else "r")
            collected.extend(hand_msgs)
            wear_text = " ".join(hand_msgs).lower()
        else:
            wear_text = " ".join(slot_msgs).lower()

        top_after = self.client.get_lines()[0].lower()
        if "put on what" in top_after or "or right" in top_after:
            collected.extend(self.client.send_key("\x1b"))
            return False, collected

        if (
            "wearing" not in wear_text
            and "(on left hand)" not in wear_text
            and "(on right hand)" not in wear_text
            and not slot_msgs
            and not both_empty
        ):
            return False, collected

        if slot in self._pack:
            base = self._pack[slot]
            for suf in ("(on left hand)", "(on right hand)"):
                if base.endswith(suf):
                    base = base[: -len(suf)].strip()
            self._pack[slot] = f"{base} (on {target_hand} hand)"

        parsed_after = parse_status(self.client.get_lines()[23])
        if parsed_after is not None:
            delta = parsed_after.strength - prev_str
            self._status = parsed_after
            if delta != 0:
                self._known_ring_stones[stone] = ("add strength", delta)
                if delta < 0:
                    self._ring_cursed[target_hand] = True
        return True, collected

    def _probe_ring(self, hand: str, slot: str, stone: str) -> list[str]:
        """Learn whether a freshly worn unidentified ring is cursed by trying to take it off.

        Rogue never names a ring when it is worn, but it refuses to remove a
        cursed one. A removable ring is harmless, so its stone is marked safe
        and the ring goes back on; a cursed one is marked bad and, with the
        hand flagged, a remove curse scroll can free it.
        """
        both_worn = sum(
            1
            for desc in self._pack.values()
            if (parsed := self._ring_value(desc)) is not None and parsed[2] is not None
        ) >= 2
        removed, msgs = self._remove_ring(hand, slot, both_worn=both_worn)
        if removed:
            self._safe_ring_stones.add(stone)
        return msgs

    def _try_auto_rings(self) -> list[str]:
        """Manage putting on, upgrading, and removing rings out of combat."""
        collected: list[str] = []
        starving = self._status.hungry_state >= 1 and not self._has_food

        # 1. Remove any worn uncursed rings that are harmful (< 0) or draining food when starving.
        for hand in ("left", "right"):
            if self._ring_cursed.get(hand, False):
                continue
            worn_items = {
                parsed[2]: (slot, parsed[0], parsed[1], parsed[3])
                for slot, desc in self._pack.items()
                if (parsed := self._ring_value(desc)) is not None and parsed[2] is not None
            }
            if hand not in worn_items:
                continue
            slot, kind, _stone, score = worn_items[hand]
            should_remove = score < 0.0 or (
                starving and kind not in ("slow digestion", "regeneration")
            )
            if should_remove:
                _, msgs = self._remove_ring(hand, slot, both_worn=len(worn_items) == 2)
                collected.extend(msgs)

        # 2. Put on or upgrade rings from unworn candidates in descending score order.
        while True:
            worn: dict[str, tuple[str, str, str, float]] = {}
            unworn: list[tuple[float, str, str, str]] = []
            for slot, desc in self._pack.items():
                parsed = self._ring_value(desc)
                if parsed is None:
                    continue
                kind, stone, hand, score = parsed
                if hand is not None:
                    worn[hand] = (slot, kind, stone, score)
                elif score > 0.0 and (
                    not starving or kind in ("slow digestion", "regeneration")
                ):
                    unworn.append((score, slot, kind, stone))

            if not unworn:
                break
            unworn.sort(key=lambda item: item[0], reverse=True)
            cand_score, cand_slot, _cand_kind, cand_stone = unworn[0]

            free_hands = [h for h in ("left", "right") if h not in worn]
            if free_hands:
                target_hand = free_hands[0]
                ok, msgs = self._put_on_ring(
                    cand_slot,
                    cand_stone,
                    target_hand,
                    both_empty=len(free_hands) == 2,
                )
                collected.extend(msgs)
                if not ok:
                    break
                if (
                    _cand_kind == "unknown"
                    and cand_stone not in self._known_ring_stones
                    and cand_stone not in self._safe_ring_stones
                ):
                    collected.extend(self._probe_ring(target_hand, cand_slot, cand_stone))
                continue

            replaceable = [
                (worn[h][3], h, worn[h][0])
                for h in ("left", "right")
                if h in worn and not self._ring_cursed.get(h, False)
            ]
            if not replaceable:
                break
            replaceable.sort(key=lambda item: item[0])
            weak_score, weak_hand, weak_slot = replaceable[0]
            if cand_score <= weak_score:
                break

            removed, r_msgs = self._remove_ring(weak_hand, weak_slot, both_worn=True)
            collected.extend(r_msgs)
            if not removed:
                continue

            ok, p_msgs = self._put_on_ring(
                cand_slot, cand_stone, weak_hand, both_empty=False
            )
            collected.extend(p_msgs)
            if not ok:
                break
            if (
                _cand_kind == "unknown"
                and cand_stone not in self._known_ring_stones
                and cand_stone not in self._safe_ring_stones
            ):
                collected.extend(self._probe_ring(weak_hand, cand_slot, cand_stone))

        return collected

    def _try_auto_wear(self) -> list[str]:
        """Wear the best armor in the pack if it strictly beats the worn armor."""
        if self._armor_cursed:
            return []
        worn_slot: Optional[str] = None
        current_score = self._status.armor
        for slot, desc in self._pack.items():
            if desc.endswith("(being worn)"):
                worn_slot = slot
                score = self._armor_value(desc)
                if score is not None:
                    current_score = score
                break

        best_slot: Optional[str] = None
        best_score = current_score
        for slot, desc in self._pack.items():
            if slot == worn_slot or desc.endswith("(being worn)"):
                continue
            score = self._armor_value(desc)
            if score is not None and score > best_score:
                best_score = score
                best_slot = slot

        if best_slot is None:
            return []

        collected: list[str] = []
        if worn_slot is not None or self._status.armor > 0:
            take_off_msgs = self.client.send_key("T")
            collected.extend(take_off_msgs)
            if "appears to be cursed" in " ".join(take_off_msgs).lower():
                self._armor_cursed = True
                return collected
            if worn_slot is not None and worn_slot in self._pack:
                d = self._pack[worn_slot]
                if d.endswith("(being worn)"):
                    self._pack[worn_slot] = d[: -len("(being worn)")].strip()

        collected.extend(self.client.send_key("W"))
        wear_msgs = self.client.send_key(best_slot)
        collected.extend(wear_msgs)
        for m in wear_msgs:
            match = _WEARING_MSG_RE.search(m)
            if match:
                self._pack[best_slot] = f"{match.group(1).strip()} (being worn)"
                break
        if "wear what" in self.client.get_lines()[0].lower():
            collected.extend(self.client.send_key("\x1b"))
        return collected

    def _try_auto_wield(self) -> list[str]:
        """Wield the best melee weapon in the pack if it beats the current one."""
        if self._weapon_cursed:
            return []
        best_slot: Optional[str] = None
        best_score = self._weapon_score
        for slot, desc in self._pack.items():
            if desc.endswith("(weapon in hand)"):
                continue
            score = self._weapon_value(desc)
            if score is not None and score > best_score:
                best_score = score
                best_slot = slot

        if best_slot is None:
            return []

        collected: list[str] = []
        w_msgs = self.client.send_key("w")
        collected.extend(w_msgs)
        if "appears to be cursed" in " ".join(w_msgs).lower():
            self._weapon_cursed = True
            return collected
        wield_msgs = self.client.send_key(best_slot)
        collected.extend(wield_msgs)
        wield_text = " ".join(wield_msgs).lower()
        if "appears to be cursed" in wield_text:
            self._weapon_cursed = True
        elif "wielding" in wield_text:
            self._weapon_score = best_score
            for s, d in list(self._pack.items()):
                if d.endswith("(weapon in hand)"):
                    self._pack[s] = d[: -len("(weapon in hand)")].strip()
            self._pack[best_slot] = f"{self._pack[best_slot]} (weapon in hand)"
        if "wield what" in self.client.get_lines()[0].lower():
            collected.extend(self.client.send_key("\x1b"))
        return collected

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

    @staticmethod
    def _path_distances(
        screen_arr: np.ndarray,
        goals: list[tuple[int, int]],
        avoid: frozenset[str] = frozenset(),
    ) -> dict[tuple[int, int], int]:
        """Shortest-path step distances from every reachable tile to the goals.

        Walls, unseen tiles and monsters block, as do diagonals past a
        wall corner (Rogue refuses those moves).  Tiles showing a
        character in `avoid` are not walked through either.
        """
        dist: dict[tuple[int, int], int] = {}
        queue: deque[tuple[int, int]] = deque()
        for goal in goals:
            dist[goal] = 0
            queue.append(goal)
        while queue:
            r, c = queue.popleft()
            d = dist[(r, c)]
            for dr, dc in _MOVE_DELTAS:
                nr, nc = r - dr, c - dc
                if not (1 <= nr <= 22 and 0 <= nc < 80) or (nr, nc) in dist:
                    continue
                ch = chr(int(screen_arr[nr, nc]))
                if ch in _WALL_CHARS or ch in avoid or ("A" <= ch <= "Z"):
                    continue
                if dr != 0 and dc != 0:
                    orth_r = chr(int(screen_arr[r, nc]))
                    orth_c = chr(int(screen_arr[nr, c]))
                    if orth_r in _WALL_CHARS or orth_c in _WALL_CHARS:
                        continue
                dist[(nr, nc)] = d + 1
                queue.append((nr, nc))
        return dist

    @classmethod
    def _stairs_distances(cls, screen_arr: np.ndarray) -> dict[tuple[int, int], int]:
        """Shortest-path step distances to any visible '%' staircase."""
        rows, cols = np.nonzero(screen_arr[1:23] == ord("%"))
        stairs = [(int(r) + 1, int(c)) for r, c in zip(rows, cols)]
        if not stairs:
            return {}
        return cls._path_distances(screen_arr, stairs)

    def _item_distances(self, screen_arr: np.ndarray) -> dict[tuple[int, int], int]:
        """Shortest-path step distances to the visible items worth collecting.

        Items and gold count, except tiles already stepped on this level.
        Weak or fainting with no food, only food counts.  Paths avoid
        known traps ('^').
        """
        if self._status.hungry_state >= 2 and not self._has_food:
            wanted = screen_arr[1:23] == ord(":")
        else:
            wanted = np.isin(screen_arr[1:23], _COLLECT_CODES)
        rows, cols = np.nonzero(wanted)
        goals = [
            (int(r) + 1, int(c))
            for r, c in zip(rows, cols)
            if (int(r) + 1, int(c)) not in self._item_skip
        ]
        if not goals:
            return {}
        return self._path_distances(screen_arr, goals, avoid=frozenset("^"))

    @staticmethod
    def _steer(
        mask: np.ndarray, pos: tuple[int, int], dist: dict[tuple[int, int], int]
    ) -> bool:
        """Keep only the legal moves on a shortest path in `dist`.

        Also masks resting and searching.  Returns False, leaving `mask`
        unchanged, when no legal move reaches a tile in `dist`.
        """
        pr, pc = pos
        reachable = [
            (idx, dist[(pr + dr, pc + dc)])
            for idx, (dr, dc) in enumerate(_MOVE_DELTAS)
            if mask[idx] == 1 and (pr + dr, pc + dc) in dist
        ]
        if not reachable:
            return False
        min_d = min(d for _, d in reachable)
        best = {idx for idx, d in reachable if d == min_d}
        for idx in range(len(_MOVE_DELTAS)):
            if idx not in best:
                mask[idx] = 0
        mask[8] = 0
        mask[9] = 0
        return True

    def _compute_action_mask(
        self, screen_arr: np.ndarray, player_pos: Optional[tuple[int, int]]
    ) -> np.ndarray:
        mask = np.ones(len(ACTIONS), dtype=np.uint8)
        if player_pos is not None:
            pr, pc = player_pos
            monster_adjacent = False
            monster_in_view = bool(
                np.any((screen_arr[1:23] >= ord("A")) & (screen_arr[1:23] <= ord("Z")))
            )
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

            # Visible items to collect before going down.  Not while a
            # monster is in view: fighting or fleeing comes first.
            item_dist = {} if monster_in_view else self._item_distances(screen_arr)

            if self._tile_under_player == "%":
                if (
                    self._status.hp < self._status.max_hp
                    and not monster_in_view
                    and self._status.hungry_state == 0
                ):
                    mask[:] = 0
                    mask[8] = 1
                    return mask
                to_item = mask.copy()
                to_item[8:] = 0
                if item_dist and self._steer(to_item, player_pos, item_dist):
                    return to_item
                mask[:] = 0
                mask[10] = 1
                return mask

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

                if item_dist and self._steer(mask, player_pos, item_dist):
                    pass  # collect visible items before anything else
                elif self._steer(mask, player_pos, self._stairs_distances(screen_arr)):
                    pass
                elif (
                    self._status.hp < 0.6 * self._status.max_hp
                    and not monster_in_view
                    and self._status.hungry_state == 0
                    and self._tile_under_player not in _ITEM_CHARS
                ):
                    mask[:] = 0
                    mask[8] = 1
                    return mask
                else:
                    # 1. Corridor / doorway forward flow: avoid immediate U-turns unless at a dead end;
                    # at a '#' corridor dead end, finish searching up to the cap before turning back.
                    if (
                        len(valid_moves) == 1
                        and self._prev_pos is not None
                        and self._tile_under_player == "#"
                        and (valid_moves[0][1], valid_moves[0][2]) == self._prev_pos
                        and mask[9] == 1
                    ):
                        mask[valid_moves[0][0]] = 0
                        mask[8] = 0
                    elif (
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
        reward += min(
            REWARD_EXP_STEP_CAP, REWARD_EXP_POINT * max(0, curr.exp - prev.exp)
        )
        reward += REWARD_EXP_LEVEL * max(0, curr.exp_level - prev.exp_level)
        reward += REWARD_DESCENT * max(0, curr.level - prev.level)
        reward += 0.02 * new_tiles
        # On death hp has been set to 0, so the remaining health is lost here.
        reward += REWARD_HEALTH * (
            curr.hp / max(1, curr.max_hp) - prev.hp / max(1, prev.max_hp)
        )
        if terminated:
            reward += REWARD_DEATH
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
