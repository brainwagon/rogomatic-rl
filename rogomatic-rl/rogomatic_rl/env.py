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
# Rogue 5.4 places each room inside one cell of a 3x3 grid, NUMCOLS / 3
# columns by NUMLINES / 3 rows (do_rooms in rooms.c).  Cell column k
# holds room columns 26k+1 .. 26k+25 and cell row k screen rows
# 8k .. 8k+7.
_CELL_COLS = 26
_CELL_ROWS = 8
# Tiles that are not room floor when deciding which side of a wall is
# the inside of the room.
_NOT_FLOOR = frozenset(" #+-|")

# The most damage each monster can do in one turn: every attack hitting
# for its maximum (the damage strings in monsters[], extern.c).  The
# venus flytrap's grows while it holds the player; 8 stands in for it.
_MONSTER_MAX_HIT: dict[str, int] = {
    "A": 0, "B": 2, "C": 12, "D": 46, "E": 2, "F": 8, "G": 27, "H": 8,
    "I": 0, "J": 32, "K": 4, "L": 1, "M": 34, "N": 0, "O": 8, "P": 16,
    "Q": 10, "R": 6, "S": 3, "T": 28, "U": 36, "V": 10, "W": 6, "X": 16,
    "Y": 12, "Z": 8,
}
# Below this max hp, rest back to full after any damage: one hobgoblin
# hit (up to 8) is a large part of a 12 hp character.
_REST_TO_FULL_MAX_HP = 20
# Monsters without ISMEAN in Rogue 5.4 (monsters[] in extern.c).
# wake_monster (monsters.c) wakes only ISMEAN monsters, so one of these
# placed asleep stays asleep until it is attacked or hit.  Each is
# dangerous to wake: centaur, ice monster (freezes: hypothermia),
# jabberwock, leprechaun (steals gold), nymph (steals an item), wraith
# (drains a level), xeroc, yeti.  Bats, orcs (ISGREED: they wake near
# gold) and phantoms (invisible) are fought as usual.
_SLEEPY_MONSTERS = frozenset("CIJLNWXY")
# Monster names, to tell which letters a message is about (extern.c).
_MONSTER_NAMES: dict[str, str] = {
    "A": "aquator", "B": "bat", "C": "centaur", "D": "dragon", "E": "emu",
    "F": "venus flytrap", "G": "griffin", "H": "hobgoblin", "I": "ice monster",
    "J": "jabberwock", "K": "kestrel", "L": "leprechaun", "M": "medusa",
    "N": "nymph", "O": "orc", "P": "phantom", "Q": "quagga", "R": "rattlesnake",
    "S": "snake", "T": "troll", "U": "black unicorn", "V": "vampire",
    "W": "wraith", "X": "xeroc", "Y": "yeti", "Z": "zombie",
}
# Steps on a level after which sleeping monsters are no longer left
# alone, in case one is blocking the only way on.
_SLEEPER_PATIENCE_STEPS = 800
# With no stairs found after this many steps on a level, go and wake a
# sleeping monster even while some walls are still to be searched: the
# stairs may be under it (#33).
_SLEEPER_STAIRS_STEPS = 500
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
# A short bow with known bonuses, e.g. "A +1,+0 short bow" (the starting
# one).  An unidentified bow could be cursed, and a cursed bow could not
# be put down again before melee, so only a known one is used.
_KNOWN_BOW_RE = re.compile(
    r"^an?\s+([+-]\d+),([+-]\d+)\s+short bow(?:\s+\(weapon in hand\))?$", re.IGNORECASE
)
# Archery ranges, counted after the policy's own move each step.  The
# bow is wielded for a monster at least _BOW_MIN_RANGE away, arrows are
# shot at one _BOW_SHOOT_RANGE or more away, and the melee weapon goes
# back in hand once any monster is within _BOW_PUT_AWAY_RANGE, so it is
# ready when the monster arrives.  Each weapon change costs a turn.
_BOW_MIN_RANGE = 6
_BOW_SHOOT_RANGE = 4
_BOW_PUT_AWAY_RANGE = 3
# Not worth arrows: flying monsters move twice a turn while 3 or more
# squares away (runners in chase.c), and bats do only 1d2 anyway.
_NO_ARCHERY = frozenset("BGK")
# Mean thrown damage when wielding a melee weapon (init_dam in weapons.c, roll_em in fight.c).
_MISSILE_BASE_DAMAGE: dict[str, float] = {
    "shuriken": 5.0,  # 2d4
    "spear": 3.5,     # 1d6
    "dagger": 2.5,    # 1d4
    "dart": 2.0,      # 1d3
    "arrow": 1.0,     # 1d1 by hand: its 2x3 needs the bow in hand
}
# A missile gets its thrown damage only from its launcher (roll_em in
# fight.c), so an arrow thrown by hand does 1x1 (plus the strength bonus,
# +1 at 16).  Far enough away the bow is wielded instead
# (`_try_throw_missile`).
# Wand and staff pack entries (nameit in things.c).  Charges show only once
# the stick itself is identified: "A wand of fire [4 charges](brass)".
_KNOWN_STICK_RE = re.compile(
    r"^an?\s+(?:wand|staff)\s+of\s+([a-z ]+?)(?:\s+\[(\d+)(?:\s+charges)?\])?\(([a-z ]+)\)$",
    re.IGNORECASE,
)
_CALLED_STICK_RE = re.compile(
    r"^an?\s+(?:wand|staff)\s+called\s+.+?\(([a-z ]+)\)$", re.IGNORECASE
)
_UNKNOWN_STICK_RE = re.compile(r"^an?\s+([a-z ]+?)\s+(?:wand|staff)$", re.IGNORECASE)
# Bolts travel BOLT_LENGTH tiles (rogue.h) and bounce off walls and doors;
# a bounce within 3 tiles can come back and hit the hero for 6d6.
_BOLT_STICKS = ("lightning", "fire", "cold")
_BOLT_LENGTH = 6
_BOLT_MIN_BOUNCE = 4
# Monsters whose special attacks cancellation removes (fight.c, monsters.c):
# aquator and rust monster rust armour, venus flytrap holds, ice monster
# freezes, leprechaun and nymph steal, medusa confuses, vampire and wraith
# drain.
_CANCEL_LETTERS = frozenset("AFILMNRVW")
# Food pack entries (inv_name in things.c): "Some food", "3 rations of
# food", "A slime-mold", "2 slime-molds".
_FOOD_COUNT_RE = re.compile(
    r"^(?:some food|an?\s+slime-mold|(\d+)\s+(?:rations of food|slime-molds))$",
    re.IGNORECASE,
)

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
        # Searches made next to each tile, so a wall tile counts as searched
        # whichever neighbouring spot the search was made from.
        self._wall_search_counts: np.ndarray = np.zeros((22, 80), dtype=np.int32)
        self._level_steps: int = 0  # steps taken on the current level
        # Set when resting starts; resting then goes on until full hp.
        self._resting: bool = False
        # Monster letters seen awake on this level: one moved, attacked,
        # was attacked or was named in a fight message.
        self._awake_letters: set[str] = set()
        # Monsters on the last screen, and those that moved onto their
        # tile in the last step, with the distance they came from.
        self._monster_tiles: dict[tuple[int, int], str] = {}
        self._approaching: dict[tuple[int, int], int] = {}
        # Per tile: the monster letter last seen there, and how many times
        # it has been seen there without moving (0 if it just moved in).
        # Kept while the tile is out of sight, so a sleeper seen again
        # from a doorway is still known.
        self._sightings: dict[tuple[int, int], tuple[str, int]] = {}
        # The mask allowed only waiting for an approaching monster, and
        # the agent has just waited once (`_lie_in_wait`).
        self._wait_forced: bool = False
        self._lying_in_wait: bool = False
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
        # Stick materials Rogue has named (material -> kind), and materials
        # zapped once to identify them.  A stick Rogue does not name on use
        # stays unknown, so it is not zapped again.
        self._stick_kinds: dict[str, str] = {}
        self._tried_stick_materials: set[str] = set()
        # Slots found out of charges ("nothing happens"), with the entry then.
        self._empty_sticks: dict[str, str] = {}
        # (kind, monster letter) already slowed or cancelled on this level.
        self._zapped_at: set[tuple[str, str]] = set()
        # Where a light stick was zapped on this level.
        self._lit_spots: list[tuple[int, int]] = []
        # A wand or staff entered the pack since the last inventory listing.
        self._stick_picked_up: bool = False
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
        self._wall_search_counts.fill(0)
        self._level_steps = 0
        self._resting = False
        self._clear_monster_tracking()
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
        self._stick_kinds = {}
        self._tried_stick_materials = set()
        self._empty_sticks = {}
        self._zapped_at = set()
        self._lit_spots = []
        self._stick_picked_up = False
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
        self._level_steps += 1
        action_idx = int(action)
        # Waited for an approaching monster: do it once, then act (#30).
        self._lying_in_wait = self._wait_forced and action_idx in (8, 9)
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
            self._wall_search_counts.fill(0)
            self._level_steps = 0
            self._resting = False
            self._clear_monster_tracking()
            self._item_skip.clear()
            self._zapped_at.clear()
            self._lit_spots.clear()
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
            if re.search(r"\b(?:wand|staff)\b", msg_text):
                self._stick_picked_up = True

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

        self._track_monsters(
            prev_screen,
            prev_pos,
            action_idx,
            screen_arr,
            curr_pos,
            messages,
            same_level=self._status.level == prev_status.level,
        )

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
                self._wall_search_counts[
                    max(0, r_idx - 1) : r_idx + 2, max(0, c_idx - 1) : c_idx + 2
                ] += 1
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
            self._inventory_dirty = True

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

    def _try_last_chance_quaff(self) -> tuple[bool, list[str]]:
        """Drink an untried unknown potion when the next hit could kill.

        That is when hp is at most the largest `_MONSTER_MAX_HIT` among
        the adjacent monsters.  A gamble: of Rogue 5.4's potions, 25% help
        in a fight (healing 13, extra healing 5, haste self 5, raise level
        2) and about 20% hurt (confusion 7, poison 8, blindness 5).
        Known healing potions are drunk first, by `_try_heal_quaff`.
        Returns whether a potion was drunk, and the messages.
        """
        lines = self.client.get_lines()
        pos = find_player(lines)
        if pos is None or self._status.hp <= 0:
            return False, []
        ignored = self._ignored_monsters()
        worst = 0
        for dr, dc in _MOVE_DELTAS:
            r, c = pos[0] + dr, pos[1] + dc
            if 1 <= r <= 22 and 0 <= c < len(lines[r]):
                ch = lines[r][c]
                if "A" <= ch <= "Z" and ch not in ignored:
                    worst = max(worst, _MONSTER_MAX_HIT.get(ch, 8))
        if self._status.hp > worst:
            return False, []
        collected: list[str] = []
        if self._inventory_dirty:
            collected.extend(self._refresh_inventory())
        for slot, desc in list(self._pack.items()):
            parsed = self._parse_potion(desc)
            if parsed is None:
                continue
            count, kind, color = parsed
            if kind is None and color not in self._tried_potion_colors:
                collected.extend(self._quaff_slot(slot, count, color))
                return True, collected
        return False, collected

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
        screen_arr: np.ndarray,
        player_pos: Optional[tuple[int, int]],
        skip: frozenset[str] = frozenset(),
    ) -> Optional[str]:
        """Return the direction key toward the nearest unobstructed aligned monster at dist >= 2.

        Monsters in `skip` are never targets; they block the line like a wall.
        """
        target = RogueEnv._missile_target(screen_arr, player_pos, skip)
        return None if target is None else target[0]

    @staticmethod
    def _missile_target(
        screen_arr: np.ndarray,
        player_pos: Optional[tuple[int, int]],
        skip: frozenset[str] = frozenset(),
    ) -> Optional[tuple[str, int]]:
        """`_find_missile_target`'s direction key, with the monster's distance."""
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
                if ch in skip:
                    break
                if "A" <= ch <= "Z":
                    if k >= 2 and (best_dist is None or k < best_dist):
                        best_dist = k
                        best_dir = ACTIONS[idx]
                    break
                k += 1
        if best_dir is None or best_dist is None:
            return None
        return best_dir, best_dist

    def _try_throw_missile(self) -> list[str]:
        """Shoot or throw at an aligned monster 2 or more squares away.

        From the wielded bow an arrow does 2x3; by hand, 1x1.  Like
        Rog-O-Matic's archery (strategy.c there): wield the bow for a
        monster at least `_BOW_MIN_RANGE` away, shoot while one is in line
        `_BOW_SHOOT_RANGE` or more away, and wield the melee weapon again
        as soon as any monster is within `_BOW_PUT_AWAY_RANGE`, so the
        bow's 1x1 is never used in melee.  A monster out of line keeps the
        bow in hand, so weapons are not swapped back and forth; with none
        in view `_try_auto_wield` puts the melee weapon back.  Flying
        monsters (`_NO_ARCHERY`) close too fast to be worth it.  Closer
        than that, the best missile is thrown by hand: daggers, darts,
        shuriken and spears need no launcher, and arrows come last.
        """
        lines = self.client.get_lines()
        player_pos = find_player(lines)
        screen_arr = screen_to_array(lines)
        erased_floor = (screen_arr[1:23] == ord(" ")) & self._seen_floor_mask
        screen_arr[1:23][erased_floor] = ord(".")
        ignored = self._ignored_monsters()
        target = self._missile_target(screen_arr, player_pos, ignored)

        if self._wielding_bow():
            arrows = self._arrow_slot()
            close = self._nearest_monster(screen_arr, player_pos, ignored)
            if arrows is None or (close is not None and close <= _BOW_PUT_AWAY_RANGE):
                melee = self._best_melee()
                return [] if melee is None else self._wield(*melee)
            if (
                target is not None
                and target[1] >= _BOW_SHOOT_RANGE
                and self._target_letter(screen_arr, player_pos, target) not in _NO_ARCHERY
            ):
                return self._throw(arrows, target[0])
            return []

        if target is None:
            return []
        dir_key, dist = target
        collected: list[str] = []
        if self._inventory_dirty:
            collected.extend(self._refresh_inventory())

        bow = self._bow_slot()
        if (
            dist >= _BOW_MIN_RANGE
            and self._target_letter(screen_arr, player_pos, target) not in _NO_ARCHERY
            and bow is not None
            and self._arrow_slot() is not None
            and not self._weapon_cursed
        ):
            collected.extend(self._wield(bow, 0.0))
            return collected

        best_slot: Optional[str] = None
        best_score: float = -1e9
        for slot, desc in self._pack.items():
            parsed = self._missile_value(desc)
            if parsed is None:
                continue
            _, score = parsed
            if score > best_score:
                best_score = score
                best_slot = slot
        if best_slot is None:
            return collected
        collected.extend(self._throw(best_slot, dir_key))
        return collected

    @staticmethod
    def _nearest_monster(
        screen_arr: np.ndarray, player_pos: Optional[tuple[int, int]], ignored: frozenset[str]
    ) -> Optional[int]:
        """Distance in moves to the nearest visible monster not in `ignored`."""
        if player_pos is None:
            return None
        dungeon = screen_arr[1:23]
        rows, cols = np.nonzero((dungeon >= ord("A")) & (dungeon <= ord("Z")))
        dists = [
            max(abs(int(r) + 1 - player_pos[0]), abs(int(c) - player_pos[1]))
            for r, c in zip(rows, cols)
            if chr(int(dungeon[r, c])) not in ignored
        ]
        return min(dists) if dists else None

    @staticmethod
    def _target_letter(
        screen_arr: np.ndarray, player_pos: Optional[tuple[int, int]], target: tuple[str, int]
    ) -> str:
        """The monster letter at a `_missile_target` result."""
        assert player_pos is not None
        dr, dc = _MOVE_DELTAS[ACTIONS.index(target[0])]
        return chr(int(screen_arr[player_pos[0] + dr * target[1], player_pos[1] + dc * target[1]]))

    def _throw(self, slot: str, dir_key: str) -> list[str]:
        """Throw (or shoot) one missile from `slot`; update the pack entry."""
        desc = self._pack[slot]
        count = 1
        match = _MISSILE_RE.match(desc.strip())
        if match and match.group(1):
            count = int(match.group(1))
        collected: list[str] = []
        collected.extend(self.client.send_key(f"t{dir_key}"))
        throw_msgs = self.client.send_key(slot)
        collected.extend(throw_msgs)
        top = self.client.get_lines()[0].lower()
        if "throw what" in top or "direction" in top:
            collected.extend(self.client.send_key("\x1b"))
            self._pack.pop(slot, None)
            return collected

        if count <= 1:
            self._pack.pop(slot, None)
        else:
            rem = count - 1
            if match:
                bonus_part = match.group(2) or ""
                kind = match.group(3).lower()
                if rem == 1:
                    article = "An" if kind[0] in "aeiou" else "A"
                    self._pack[slot] = f"{article} {bonus_part}{kind}"
                else:
                    self._pack[slot] = f"{rem} {bonus_part}{kind}s"
            else:
                self._pack.pop(slot, None)
        return collected

    def _food_count(self) -> int:
        """Rations and slime-molds in the pack, as last listed."""
        total = 0
        for desc in self._pack.values():
            match = _FOOD_COUNT_RE.match(desc.strip())
            if match:
                total += int(match.group(1)) if match.group(1) else 1
        return total

    def _parse_stick(self, desc: str) -> Optional[tuple[Optional[str], str, Optional[int]]]:
        """Parse a wand or staff pack entry into (kind, material, charges).

        `kind` is None while the stick is unidentified (a material named
        earlier in the game counts as identified); `charges` is None
        unless Rogue shows them.
        """
        clean = desc.strip()
        match = _KNOWN_STICK_RE.match(clean)
        if match:
            kind = match.group(1).strip().lower()
            material = match.group(3).strip().lower()
            self._stick_kinds[material] = kind
            charges = int(match.group(2)) if match.group(2) is not None else None
            return kind, material, charges
        match = _CALLED_STICK_RE.match(clean) or _UNKNOWN_STICK_RE.match(clean)
        if match:
            material = match.group(1).strip().lower()
            return self._stick_kinds.get(material), material, None
        return None

    def _usable_sticks(self) -> list[tuple[str, Optional[str], str]]:
        """(slot, kind, material) for each stick that may still have charges."""
        out = []
        for slot, desc in sorted(self._pack.items()):
            parsed = self._parse_stick(desc)
            if parsed is None:
                continue
            kind, material, charges = parsed
            if charges == 0 or self._empty_sticks.get(slot) == desc:
                continue
            out.append((slot, kind, material))
        return out

    @staticmethod
    def _zap_targets(
        screen_arr: np.ndarray, player_pos: tuple[int, int]
    ) -> list[tuple[int, int, str, int]]:
        """The first monster along each of the 8 directions a stick can aim.

        Returns (direction index, distance, monster letter, bounce), where
        `bounce` is the distance to the first wall or door a bolt would
        bounce off (fire_bolt in sticks.c), or a large number if none is
        within a bolt's reach.  Rays pass doors and items and stop at the
        first monster or wall (step_ok in io.c).
        """
        pr, pc = player_pos
        targets = []
        for idx, (dr, dc) in enumerate(_MOVE_DELTAS):
            target: Optional[tuple[int, str]] = None
            bounce = 99
            for k in range(1, 80):
                nr, nc = pr + k * dr, pc + k * dc
                ch = (
                    chr(int(screen_arr[nr, nc]))
                    if 1 <= nr <= 22 and 0 <= nc < 80
                    else " "
                )
                if ch in _WALL_CHARS or ch == "+":
                    bounce = min(bounce, k)
                    if ch != "+":
                        break
                if target is None and "A" <= ch <= "Z":
                    target = (k, ch)
                if target is not None and k >= _BOLT_LENGTH:
                    break
            if target is not None:
                targets.append((idx, target[0], target[1], bounce))
        return targets

    def _choose_zap(
        self,
        sticks: list[tuple[str, Optional[str], str]],
        targets: list[tuple[int, int, str, int]],
    ) -> Optional[tuple[str, Optional[str], str, tuple[int, int, str, int]]]:
        """Pick (slot, kind, material, target) to zap now, or None.

        In order: teleport away when hurt, cancellation against a monster
        with a special attack, a bolt that cannot bounce back, slow
        monster when hurt, magic missile, polymorph as a last resort, then
        an unknown stick to identify it while healthy.  Haste monster,
        invisibility, teleport to, drain life and nothing are never used.
        """
        status = self._status
        hp_frac = status.hp / status.max_hp if status.max_hp > 0 else 1.0
        ignored = self._ignored_monsters()
        targets = [t for t in targets if t[2] not in ignored]
        by_kind: dict[str, tuple[str, str]] = {}
        unknown: list[tuple[str, str]] = []
        for slot, kind, material in sticks:
            if kind is None:
                if material not in self._tried_stick_materials:
                    unknown.append((slot, material))
            else:
                by_kind.setdefault(kind, (slot, material))

        def nearest(cands: list[tuple[int, int, str, int]]) -> Optional[tuple[int, int, str, int]]:
            return min(cands, key=lambda t: t[1]) if cands else None

        def pick(kind: str, cands: list[tuple[int, int, str, int]]):
            if kind not in by_kind:
                return None
            target = nearest(cands)
            if target is None:
                return None
            slot, material = by_kind[kind]
            return slot, kind, material, target

        bolt_safe = [
            t for t in targets if t[1] <= _BOLT_LENGTH and t[3] >= _BOLT_MIN_BOUNCE
        ]
        choices = []
        if hp_frac < 0.4:
            choices.append(pick("teleport away", targets))
        choices.append(
            pick(
                "cancellation",
                [
                    t
                    for t in targets
                    if t[2] in _CANCEL_LETTERS and ("cancellation", t[2]) not in self._zapped_at
                ],
            )
        )
        for kind in _BOLT_STICKS:
            cands = [t for t in bolt_safe if not (kind == "fire" and t[2] == "D")]
            choices.append(pick(kind, cands))
        if hp_frac < 0.6:
            choices.append(
                pick(
                    "slow monster",
                    [t for t in targets if ("slow monster", t[2]) not in self._zapped_at],
                )
            )
        choices.append(pick("magic missile", targets))
        if hp_frac < 0.25:
            choices.append(pick("polymorph", [t for t in targets if t[1] == 1]))
        for choice in choices:
            if choice is not None:
                return choice

        # Identify an unknown stick on a monster at range while healthy:
        # drain life halves hp, teleport to brings the monster adjacent,
        # and a bolt must not be able to bounce back.
        if unknown and hp_frac >= 0.7:
            target = nearest([t for t in bolt_safe if t[1] >= 2])
            if target is not None:
                slot, material = unknown[0]
                return slot, None, material, target
        return None

    def _current_screen(self) -> tuple[np.ndarray, Optional[tuple[int, int]]]:
        """The live screen with erased dark-room floor restored, and the player."""
        lines = self.client.get_lines()
        screen_arr = screen_to_array(lines)
        erased_floor = (screen_arr[1:23] == ord(" ")) & self._seen_floor_mask
        screen_arr[1:23][erased_floor] = ord(".")
        return screen_arr, find_player(lines)

    def _zap(self, slot: str, dir_key: str) -> tuple[bool, list[str]]:
        """Zap the stick in `slot` toward `dir_key`; True if a turn was used.

        An empty stick ("nothing happens") still uses the turn, and is
        remembered so it is not zapped again.
        """
        desc = self._pack.get(slot, "")
        collected = self.client.send_key(f"z{dir_key}")
        zap_msgs = self.client.send_key(slot)
        collected.extend(zap_msgs)
        top = self.client.get_lines()[0].lower()
        if "zap with" in top or "direction" in top:
            collected.extend(self.client.send_key("\x1b"))
            self._empty_sticks[slot] = desc
            return False, collected
        self._inventory_dirty = True
        if "nothing happens" in " ".join(zap_msgs).lower():
            self._empty_sticks[slot] = desc
        return True, collected

    def _try_zap(self) -> tuple[bool, list[str]]:
        """Zap a wand or staff at a monster in line, if one is worth it.

        Returns whether a game turn was used, and the messages.
        """
        screen_arr, player_pos = self._current_screen()
        if player_pos is None:
            return False, []
        targets = self._zap_targets(screen_arr, player_pos)
        if not targets:
            return False, []
        collected: list[str] = []
        has_stick = any(self._parse_stick(d) for d in self._pack.values())
        if self._inventory_dirty and (has_stick or self._stick_picked_up):
            collected.extend(self._refresh_inventory())
            self._stick_picked_up = False
        choice = self._choose_zap(self._usable_sticks(), targets)
        if choice is None:
            return False, collected
        slot, kind, material, (dir_idx, _dist, letter, _bounce) = choice
        if kind is None:
            self._tried_stick_materials.add(material)
        elif kind in ("slow monster", "cancellation"):
            self._zapped_at.add((kind, letter))
        used, msgs = self._zap(slot, ACTIONS[dir_idx])
        collected.extend(msgs)
        if kind is None and used:
            # Rogue names light, bolts, magic missile and (when seen)
            # polymorph on use; listing the pack takes no game turn.
            collected.extend(self._refresh_inventory())
        return used, collected

    def _in_dark_room(self, player_pos: tuple[int, int]) -> bool:
        """True when standing on room floor that Rogue erases behind the player."""
        if self._tile_under_player != "." or not (1 <= player_pos[0] <= 22):
            return False
        raw = screen_to_array(self.client.get_lines())[1:23]
        erased = (raw == ord(" ")) & self._seen_floor_mask
        r, c = player_pos[0] - 1, player_pos[1]
        return bool(np.any(erased[max(0, r - 3) : r + 4, max(0, c - 3) : c + 4]))

    def _try_light(self) -> list[str]:
        """Light a dark room with a known light stick, once per spot."""
        _, player_pos = self._current_screen()
        if player_pos is None or not self._in_dark_room(player_pos):
            return []
        if any(
            max(abs(player_pos[0] - r), abs(player_pos[1] - c)) <= 5
            for r, c in self._lit_spots
        ):
            return []
        for slot, kind, _material in self._usable_sticks():
            if kind == "light":
                self._lit_spots.append(player_pos)
                return self._zap(slot, "h")[1]
        return []

    def _monster_in_view(self, lines: list[str]) -> bool:
        """True if any monster letter ('A'..'Z') is visible on the dungeon map.

        A sleeping ice monster does not count (see `_ignored_monsters`).
        """
        ignored = self._ignored_monsters()
        return any("A" <= ch <= "Z" and ch not in ignored for row in lines[1:23] for ch in row)

    def _ignored_monsters(self) -> frozenset[str]:
        """Monster letters to leave alone: `_SLEEPY_MONSTERS` not seen awake.

        These never wake on their own (wake_monster in monsters.c), so one
        left alone stays harmless; an ice monster's freezing hits, for
        one, are what kill by hypothermia.  The agent does not attack,
        throw at, zap or count as a monster in view any such letter until
        one has been seen awake on this level (`_track_monsters`), or
        until `_SLEEPER_PATIENCE_STEPS` on the level, in case one blocks
        the way.  Seeing a letter awake applies to all of that letter on
        the level: coarse, but it errs toward fighting.
        """
        if self._level_steps >= _SLEEPER_PATIENCE_STEPS:
            return frozenset()
        return _SLEEPY_MONSTERS - self._awake_letters

    def _lie_in_wait(self, player_pos: tuple[int, int], ignored: frozenset[str]) -> bool:
        """Whether to wait a turn so an approaching monster steps in first.

        The player moves, then the monsters do.  Stepping next to a
        monster gives it the first swing; letting it step next to the
        player gives the player the first swing.  Rog-O-Matic waits when a
        monster is an even number of squares away (tomonster and
        fightmonster in strategy.c there).  Here: a monster that has just
        moved closer and is now 2 squares away, waited for once.  Not a
        dragon, whose flame reaches from a distance.
        """
        if self._lying_in_wait:
            return False
        pr, pc = player_pos
        for pos, old_dist in self._approaching.items():
            letter = self._monster_tiles.get(pos)
            if letter is None or letter in ignored or letter == "D":
                continue
            dist = max(abs(pos[0] - pr), abs(pos[1] - pc))
            if dist == 2 and old_dist > dist:
                return True
        return False

    def _clear_monster_tracking(self) -> None:
        self._awake_letters.clear()
        self._monster_tiles = {}
        self._approaching = {}
        self._sightings = {}
        self._wait_forced = False
        self._lying_in_wait = False

    def _track_monsters(
        self,
        prev_screen: np.ndarray,
        prev_pos: Optional[tuple[int, int]],
        action_idx: int,
        screen_arr: np.ndarray,
        curr_pos: Optional[tuple[int, int]],
        messages: list[str],
        *,
        same_level: bool,
    ) -> None:
        """Note which monsters moved, and which letters are awake.

        Like Rog-O-Matic's sleepmonster / wakemonster (monsters.c there):
        a monster that moved is awake.  So is one that the agent attacked,
        or whose name appears in a message ("the centaur hit", "the arrow
        misses the yeti"); being frozen means an ice monster.  A move is a
        letter on a new tile with the same letter within 2 tiles of it on
        the last screen, and no longer on its old tile.
        """
        text = " ".join(messages).lower()
        for letter, name in _MONSTER_NAMES.items():
            if name in text:
                self._awake_letters.add(letter)
        if "frozen" in text:
            self._awake_letters.add("I")
        if same_level and self._is_attack(prev_screen, prev_pos, action_idx):
            dr, dc = _MOVE_DELTAS[action_idx]
            assert prev_pos is not None
            self._awake_letters.add(chr(int(prev_screen[prev_pos[0] + dr, prev_pos[1] + dc])))

        dungeon = screen_arr[1:23]
        rows, cols = np.nonzero((dungeon >= ord("A")) & (dungeon <= ord("Z")))
        now = {(int(r) + 1, int(c)): chr(int(dungeon[r, c])) for r, c in zip(rows, cols)}
        approaching: dict[tuple[int, int], int] = {}
        moved: set[tuple[int, int]] = set()
        if same_level:
            for pos, letter in now.items():
                if self._monster_tiles.get(pos) == letter:
                    continue
                for old, old_letter in self._monster_tiles.items():
                    if (
                        old_letter == letter
                        and now.get(old) != letter
                        and max(abs(old[0] - pos[0]), abs(old[1] - pos[1])) <= 2
                    ):
                        self._awake_letters.add(letter)
                        moved.add(pos)
                        if curr_pos is not None:
                            approaching[pos] = max(
                                abs(old[0] - curr_pos[0]), abs(old[1] - curr_pos[1])
                            )
                        break
        for pos, letter in now.items():
            seen_letter, count = self._sightings.get(pos, ("", 0))
            if pos in moved:
                count = 0
            elif seen_letter == letter:
                count += 1
            else:
                count = 1
            self._sightings[pos] = (letter, count)
        self._monster_tiles = now
        self._approaching = approaching

    def _sleeping_monsters(
        self, player_pos: Optional[tuple[int, int]]
    ) -> dict[tuple[int, int], str]:
        """Visible monsters that look asleep: seen twice on their tile, not adjacent.

        Rogue wakes a mean monster only 2 times in 3 when the player
        enters its lit room or moves next to it (door_open in move.c,
        look in misc.c), so sleeping monsters in view are common.  One
        that stays put is treated as asleep, as Rog-O-Matic does
        (sleepmonster in monsters.c there).  Adjacent ones never count:
        an awake monster next to the player also stands still.
        """
        if player_pos is None:
            return {}
        pr, pc = player_pos
        return {
            pos: letter
            for pos, letter in self._monster_tiles.items()
            if self._sightings.get(pos, ("", 0))[1] >= 2
            and self._sightings[pos][0] == letter
            and max(abs(pos[0] - pr), abs(pos[1] - pc)) > 1
        }

    def _steer_to_sleeper(
        self, mask: np.ndarray, screen_arr: np.ndarray, player_pos: tuple[int, int]
    ) -> bool:
        """With no stairs in sight, walk up to a sleeping monster to fight it.

        A monster standing on the stairs hides them, and a sleeping one
        can stand there for good.  Rog-O-Matic, after failing to find the
        stairs, attacks sleeping monsters for this reason (strategize in
        strategy.c there, "No stairs, attacking sleeping monster").  It
        also clears a doorway a sleeper blocks.  Only monsters the player
        can survive one maximum hit from, never an ice monster, and not
        the ones `_ignored_monsters` leaves alone.  The attack itself
        comes from the fight-back rule once the agent is adjacent.
        Returns False, leaving `mask` unchanged, when there is no such
        monster in reach.
        """
        if np.any(screen_arr[1:23] == ord("%")):
            return False
        ignored = self._ignored_monsters()
        hp = self._status.hp
        goals = [
            pos
            for pos, letter in self._sleeping_monsters(player_pos).items()
            if letter not in ignored
            and letter != "I"
            and hp > _MONSTER_MAX_HIT.get(letter, 8)
        ]
        if not goals:
            return False
        return self._steer(mask, player_pos, self._path_distances(screen_arr, goals, avoid=frozenset("^")))

    def _run_auto_actions(self) -> list[str]:
        """Run scripted inventory refresh and auto-equip/quaff/read/missile/ring actions."""
        collected: list[str] = []
        collected.extend(self._try_heal_quaff())
        if self._monster_in_view(self.client.get_lines()):
            escaped = self._try_emergency_scroll()
            collected.extend(escaped)
            if not escaped:
                zapped, zap_msgs = self._try_zap()
                collected.extend(zap_msgs)
                if not zapped:
                    quaffed, quaff_msgs = self._try_last_chance_quaff()
                    collected.extend(quaff_msgs)
                    if not quaffed:
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
        collected.extend(self._try_light())
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
        # Regeneration burns food three times as fast (ring_eat in rings.c),
        # so it is worn only with two rations or more in the pack.
        food_short = self._food_count() < 2

        def unwanted(kind: str) -> bool:
            return (starving and kind != "slow digestion") or (
                food_short and kind == "regeneration"
            )

        # 1. Remove any worn uncursed rings that are harmful (< 0) or draining food when short.
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
            should_remove = score < 0.0 or unwanted(kind)
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
                elif score > 0.0 and not unwanted(kind):
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
        return self._wield(best_slot, best_score)

    def _wield(self, slot: str, score: float) -> list[str]:
        """Wield the weapon in `slot`, scored `score` for `_try_auto_wield`."""
        collected: list[str] = []
        w_msgs = self.client.send_key("w")
        collected.extend(w_msgs)
        if "appears to be cursed" in " ".join(w_msgs).lower():
            self._weapon_cursed = True
            return collected
        wield_msgs = self.client.send_key(slot)
        collected.extend(wield_msgs)
        wield_text = " ".join(wield_msgs).lower()
        if "appears to be cursed" in wield_text:
            self._weapon_cursed = True
        elif "wielding" in wield_text:
            self._weapon_score = score
            for s, d in list(self._pack.items()):
                if d.endswith("(weapon in hand)"):
                    self._pack[s] = d[: -len("(weapon in hand)")].strip()
            self._pack[slot] = f"{self._pack[slot]} (weapon in hand)"
        if "wield what" in self.client.get_lines()[0].lower():
            collected.extend(self.client.send_key("\x1b"))
        return collected

    def _bow_slot(self) -> Optional[str]:
        """The slot of a short bow with known, not negative, bonuses."""
        for slot, desc in self._pack.items():
            match = _KNOWN_BOW_RE.match(desc.strip())
            if match and int(match.group(1)) >= 0 and int(match.group(2)) >= 0:
                return slot
        return None

    def _wielding_bow(self) -> bool:
        return any(
            "short bow" in desc.lower() and desc.endswith("(weapon in hand)")
            for desc in self._pack.values()
        )

    def _arrow_slot(self) -> Optional[str]:
        for slot, desc in self._pack.items():
            match = _MISSILE_RE.match(desc.strip())
            if match and match.group(3).lower() == "arrow":
                return slot
        return None

    def _best_melee(self) -> Optional[tuple[str, float]]:
        """The best melee weapon in the pack, wielded or not, and its score."""
        best: Optional[tuple[str, float]] = None
        for slot, desc in self._pack.items():
            score = self._weapon_value(desc.replace("(weapon in hand)", "").strip())
            if score is not None and (best is None or score > best[1]):
                best = (slot, score)
        return best

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

    def _frontier_distances(self, screen_arr: np.ndarray) -> dict[tuple[int, int], int]:
        """Shortest-path step distances to the nearest unexplored frontier.

        The frontier is every seen tile the player has not stood on that
        leads somewhere new: a corridor or door tile, or room floor next
        to a tile never seen (the edge of a dark room).  Paths avoid
        known traps.
        """
        dungeon = screen_arr[1:23]
        unseen = np.pad(~self._seen_mask, 1, constant_values=False)
        next_to_unseen = (
            unseen[:-2, 1:-1] | unseen[2:, 1:-1] | unseen[1:-1, :-2] | unseen[1:-1, 2:]
        )
        frontier = ~self._visited_mask & (
            (dungeon == ord("#"))
            | (dungeon == ord("+"))
            | ((dungeon == ord(".")) & next_to_unseen)
        )
        rows, cols = np.nonzero(frontier)
        if len(rows) == 0:
            return {}
        goals = [(int(r) + 1, int(c)) for r, c in zip(rows, cols)]
        return self._path_distances(screen_arr, goals, avoid=frozenset("^"))

    def _dead_end_distances(
        self, screen_arr: np.ndarray, player_pos: tuple[int, int]
    ) -> dict[tuple[int, int], int]:
        """Distances to corridor dead ends not yet searched the full 6 times.

        Corridors in Rogue lead somewhere, so a dead end usually hides a
        passage.  Empty when the player is standing on one, so the dead-end
        search rule can act there.
        """
        goals = self._dead_end_tiles(screen_arr, player_pos)
        if not goals or player_pos in goals:
            return {}
        return self._path_distances(screen_arr, goals, avoid=frozenset("^"))

    def _dead_end_tiles(
        self, screen_arr: np.ndarray, player_pos: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Corridor dead ends (screen row, col) still under the search cap.

        Rogue digs corridors in straight orthogonal runs (conn in
        passages.c), so a dead end is a corridor tile with one walkable
        tile beside it orthogonally.  Diagonal neighbours are not
        counted: one step past a bend, the tile before the bend is a
        diagonal neighbour of the dead end.
        """
        dungeon = screen_arr[1:23]
        corridor = dungeon == ord("#")
        pr, pc = player_pos
        if 1 <= pr <= 22:
            corridor[pr - 1, pc] = self._tile_under_player == "#"
        walkable = np.pad(
            ~np.isin(dungeon, [ord(" "), ord("-"), ord("|")]), 1, constant_values=False
        )
        neighbours = sum(
            walkable[1 + dr : 23 + dr, 1 + dc : 81 + dc].astype(np.int8)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        dead_end = corridor & (neighbours == 1) & (self._search_counts < self._search_cap())
        rows, cols = np.nonzero(dead_end)
        return [(int(r) + 1, int(c)) for r, c in zip(rows, cols)]

    def _search_cap(self) -> int:
        """Searches allowed per spot: 6, and 6 more for every 500 steps on the level.

        Six searches find a secret door with about 74% chance (1 in 5 each,
        search in command.c), so a level with no stairs found after a long
        time gets searched again.
        """
        return 6 * (1 + self._level_steps // 500)

    def _rest_action(self) -> int:
        """The action to rest with: search ('s') where something could be hidden.

        Rog-O-Matic always rests by searching (restup in tactics.c there).
        A search takes the same turn, still heals (doctor in daemons.c
        counts every quiet turn) and may find a secret door or corridor
        next to the player.  Level 1 hides nothing (#25), so there it is
        '.'.
        """
        return 9 if self._status.level >= 2 else 8

    def _wants_rest(self, monster_in_view: bool) -> bool:
        """Whether to rest now; once started, resting goes on to full hp.

        Resting starts below 60% hp, or after any damage while max hp is
        under `_REST_TO_FULL_MAX_HP`.  It stops at full hp, or when a
        monster comes into view or the player gets hungry.
        """
        s = self._status
        if monster_in_view or s.hungry_state != 0 or s.hp >= s.max_hp:
            self._resting = False
            return False
        if s.hp < 0.6 * s.max_hp or s.max_hp < _REST_TO_FULL_MAX_HP:
            self._resting = True
        return self._resting

    def _map_rows(
        self, screen_arr: np.ndarray, player_pos: Optional[tuple[int, int]]
    ) -> list[str]:
        """The screen as 24 strings, with the tile under the player in place of '@'."""
        rows = [bytes(row).decode("ascii", "replace") for row in screen_arr]
        if player_pos is not None:
            pr, pc = player_pos
            rows[pr] = rows[pr][:pc] + self._tile_under_player + rows[pr][pc + 1 :]
        return rows

    @staticmethod
    def _door_wall(
        rows: list[str], r: int, c: int
    ) -> Optional[tuple[tuple[int, int], Optional[tuple[int, int]]]]:
        """Whether the wall tile at (r, c) could be a secret door.

        Rogue 5.4 puts a door on a room's wall only where it faces a room
        in the next grid cell (conn in passages.c): the bottom wall toward
        the cell below, the right wall toward the cell to the right, and
        so on, never on a corner.  Each pair of cells is joined at most
        once, so a wall side already showing a door has no other.

        Returns None when the tile cannot be a door, and otherwise the
        (dr, dc) step from the wall into the room and the (row, col) grid
        cell the wall faces.  The facing cell is None when the tile's cell
        cannot be worked out.
        """

        def at(y: int, x: int) -> str:
            if 1 <= y <= 22 and 0 <= x < 80:
                return rows[y][x]
            return " "

        ch = at(r, c)
        if ch == "-":
            if at(r - 1, c) == "|" or at(r + 1, c) == "|":
                return None  # a corner
            seg = [c]
            for step in (-1, 1):
                x = c + step
                while at(r, x) in "-+":
                    seg.append(x)
                    x += step
            if any(at(r, x) == "+" for x in seg):
                return None
            inward: Optional[tuple[int, int]] = None
            if at(r + 1, c) not in _NOT_FLOOR:
                inward = (1, 0)
            elif at(r - 1, c) not in _NOT_FLOOR:
                inward = (-1, 0)
            else:
                # Inside not seen: a corner tells which way the room lies.
                for x in seg:
                    if at(r + 1, x) == "|":
                        inward = (1, 0)
                        break
                    if at(r - 1, x) == "|":
                        inward = (-1, 0)
                        break
        elif ch == "|":
            seg = [r]
            ends = []
            for step in (-1, 1):
                y = r + step
                while at(y, c) in "|+":
                    seg.append(y)
                    y += step
                ends.append(y)
            if any(at(y, c) == "+" for y in seg):
                return None
            inward = None
            if at(r, c + 1) not in _NOT_FLOOR:
                inward = (0, 1)
            elif at(r, c - 1) not in _NOT_FLOOR:
                inward = (0, -1)
            else:
                for y in ends:
                    if at(y, c) == "-" and at(y, c + 1) == "-":
                        inward = (0, 1)
                        break
                    if at(y, c) == "-" and at(y, c - 1) == "-":
                        inward = (0, -1)
                        break
        else:
            return None
        if inward is None:
            return None
        cell_r = min(2, r // _CELL_ROWS)
        cell_c = min(2, max(0, (c - 1) // _CELL_COLS))
        # The wall faces away from the room: opposite to `inward`.
        face_r, face_c = cell_r - inward[0], cell_c - inward[1]
        if not (0 <= face_r <= 2 and 0 <= face_c <= 2):
            return None  # faces the edge of the map
        return inward, (face_r, face_c)

    def _wall_search_tiers(
        self, screen_arr: np.ndarray, player_pos: tuple[int, int]
    ) -> list[set[tuple[int, int]]]:
        """Spots to search from for secret doors, best first.

        A spot is the floor tile just inside a wall tile that could be a
        door (`_door_wall`) and has been searched fewer than
        `_search_cap()` times.  Walls facing a grid cell where no room
        floor has been seen come first, as a door there leads somewhere
        new.  Within each, spots whose wall neighbours along the wall are
        also unsearched come first, so searches 3 tiles apart cover the
        wall with little overlap.  Empty tiers are left out.
        """
        rows = self._map_rows(screen_arr, player_pos)
        cap = self._search_cap()
        floor_r, floor_c = np.nonzero(self._seen_floor_mask)
        reached = {
            (min(2, (int(r) + 1) // _CELL_ROWS), min(2, max(0, (int(c) - 1) // _CELL_COLS)))
            for r, c in zip(floor_r, floor_c)
        }
        walls: dict[tuple[int, int], tuple[tuple[int, int], Optional[tuple[int, int]]]] = {}
        wall_r, wall_c = np.nonzero(np.isin(screen_arr[1:23], [ord("-"), ord("|")]))
        for r0, c0 in zip(wall_r, wall_c):
            r, c = int(r0) + 1, int(c0)
            info = self._door_wall(rows, r, c)
            if info is not None:
                walls[(r, c)] = info

        def done(pos: tuple[int, int]) -> bool:
            return pos in walls and self._wall_search_counts[pos[0] - 1, pos[1]] >= cap

        tiers: list[set[tuple[int, int]]] = [set(), set(), set(), set()]
        for (r, c), ((dr, dc), facing) in walls.items():
            if done((r, c)):
                continue
            sr, sc = r + dr, c + dc
            if not (1 <= sr <= 22 and 0 <= sc < 80):
                continue
            ch = rows[sr][sc]
            if ch in _NOT_FLOOR or ch == "^" or "A" <= ch <= "Z":
                continue
            # Along the wall is perpendicular to the inward step.
            spaced = not done((r + dc, c + dr)) and not done((r - dc, c - dr))
            new_cell = facing not in reached
            tiers[(0 if new_cell else 2) + (0 if spaced else 1)].add((sr, sc))
        return [t for t in tiers if t]

    def _wall_searchable(
        self, screen_arr: np.ndarray, player_pos: tuple[int, int]
    ) -> bool:
        """Whether a wall tile next to the player could be a door still worth searching."""
        rows = self._map_rows(screen_arr, player_pos)
        cap = self._search_cap()
        pr, pc = player_pos
        for dr, dc in _MOVE_DELTAS:
            r, c = pr + dr, pc + dc
            if not (1 <= r <= 22 and 0 <= c < 80) or rows[r][c] not in "-|":
                continue
            if (
                self._wall_search_counts[r - 1, c] < cap
                and self._door_wall(rows, r, c) is not None
            ):
                return True
        return False

    def _steer_wall_search(
        self, mask: np.ndarray, screen_arr: np.ndarray, player_pos: tuple[int, int]
    ) -> bool:
        """Walk to the best spot for finding a secret door, and search there.

        Returns False, leaving `mask` unchanged, when no spot is reachable.
        """
        for spots in self._wall_search_tiers(screen_arr, player_pos):
            if player_pos in spots:
                mask[:] = 0
                mask[9] = 1
                return True
            dist = self._path_distances(screen_arr, list(spots), avoid=frozenset("^"))
            if self._steer(mask, player_pos, dist):
                return True
        return False

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
        self._wait_forced = False
        if player_pos is not None:
            pr, pc = player_pos
            monster_adjacent = False
            ignored = self._ignored_monsters()
            letters = (screen_arr[1:23] >= ord("A")) & (screen_arr[1:23] <= ord("Z"))
            for ch in ignored:
                letters &= screen_arr[1:23] != ord(ch)
            # Sleeping monsters do not stop resting, collecting, exploring
            # or searching; paths still go around them (#33).
            for r, c in self._sleeping_monsters(player_pos):
                letters[r - 1, c] = False
            monster_in_view = bool(np.any(letters))
            searchable_adjacent = False

            for idx, (dr, dc) in enumerate(_MOVE_DELTAS):
                nr, nc = pr + dr, pc + dc
                if not (1 <= nr <= 22 and 0 <= nc < 80):
                    mask[idx] = 0
                else:
                    ch = chr(int(screen_arr[nr, nc]))
                    if ch in ignored:
                        mask[idx] = 0  # moving there would attack it
                    elif "A" <= ch <= "Z":
                        monster_adjacent = True
                    if (
                        ch == " "
                        and (dr == 0 or dc == 0)
                        and self._tile_under_player in ("#", "+")
                    ):
                        searchable_adjacent = True  # a hidden corridor tile

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
                    mask[self._rest_action()] = 1
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
                and chr(int(screen_arr[pr + dr, pc + dc])) not in ignored
            ]
            if attack_moves:
                mask[:] = 0
                mask[attack_moves] = 1
                return mask

            if self._lie_in_wait(player_pos, ignored):
                self._wait_forced = True
                mask[:] = 0
                mask[self._rest_action()] = 1
                return mask

            # Mask '.' (rest) when at full HP with no adjacent monster
            if self._status.hp >= self._status.max_hp and not monster_adjacent:
                mask[8] = 0

            # Mask 's' (search) unless something next to the player could be
            # hidden: a blank beside a corridor, under the per-spot search
            # cap, or a wall tile that could be a secret door, searched
            # fewer times than the cap.  Level 1 hides nothing: door() and
            # putpass() in passages.c hide only when rnd(10) + 1 < level.
            curr_searches = (
                int(self._search_counts[pr - 1, pc]) if 1 <= pr <= 22 else 0
            )
            secrets_possible = self._status.level >= 2
            if not secrets_possible or not (
                (searchable_adjacent and curr_searches < self._search_cap())
                or self._wall_searchable(screen_arr, player_pos)
            ):
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
                    secrets_possible
                    and not monster_in_view
                    and mask[9] == 1
                    and player_pos in self._dead_end_tiles(screen_arr, player_pos)
                ):
                    # At a corridor dead end, search it out before going
                    # anywhere else: dead ends usually hide a passage, and
                    # coming back once the rest of the level is explored
                    # is a long walk.
                    mask[:] = 0
                    mask[9] = 1
                    return mask
                elif (
                    self._wants_rest(monster_in_view)
                    and self._tile_under_player not in _ITEM_CHARS
                ):
                    mask[:] = 0
                    mask[self._rest_action()] = 1
                    return mask
                elif not monster_in_view and self._steer(
                    mask, player_pos, self._frontier_distances(screen_arr)
                ):
                    # Explore toward the nearest unexplored corridor, door or
                    # dark floor.  Like item steering, not with a monster in
                    # view: one seen only from a doorway would otherwise
                    # switch the two rules back and forth.
                    pass
                elif (
                    not monster_in_view
                    and self._level_steps >= _SLEEPER_STAIRS_STEPS
                    and self._steer_to_sleeper(mask, screen_arr, player_pos)
                ):
                    pass  # long on this level: the stairs may be under a sleeper
                elif (
                    secrets_possible
                    and not monster_in_view
                    and self._steer(
                        mask, player_pos, self._dead_end_distances(screen_arr, player_pos)
                    )
                ):
                    # Nothing left to explore: go and search a dead end.  Not
                    # with a monster in view, for the same reason as above:
                    # one seen only from a doorway sent the agent back and
                    # forth between this rule and item steering.
                    pass
                elif (
                    secrets_possible
                    and not monster_in_view
                    and player_pos not in self._dead_end_tiles(screen_arr, player_pos)
                    and self._steer_wall_search(mask, screen_arr, player_pos)
                ):
                    # No dead ends left either: search the room walls that
                    # could hide a door, those facing unexplored cells first.
                    pass
                elif not monster_in_view and self._steer_to_sleeper(
                    mask, screen_arr, player_pos
                ):
                    pass  # nothing left to search: the stairs may be under a sleeper
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
