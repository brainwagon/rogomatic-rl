"""Screen parser for Rogue 5.4's 24x80 terminal output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


_STATUS_RE = re.compile(
    r"Level:\s*(\d+)\s+"
    r"Gold:\s*(\d+)\s+"
    r"Hp:\s*(-?\d+)\((\d+)\)\s+"
    r"Str:\s*(\d+)\((\d+)\)\s+"
    r"Arm:\s*(-?\d+)\s+"
    r"Exp:\s*(\d+)/(\d+)"
    r"(?:\s+(Hungry|Weak|Faint))?"
)

_HUNGER_MAP = {
    None: 0,
    "": 0,
    "Hungry": 1,
    "Weak": 2,
    "Faint": 3,
}

_DEATH_MARKERS = (
    "Rest",
    "In",
    "Peace",
    "--Press return to continue--",
    "Top Ten",
)


@dataclass(frozen=True)
class RogueStatus:
    """Parsed status bar fields from row 23 of Rogue 5.4."""

    level: int = 1
    gold: int = 0
    hp: int = 12
    max_hp: int = 12
    strength: int = 16
    max_strength: int = 16
    armor: int = 4
    exp_level: int = 1
    exp: int = 0
    hungry_state: int = 0

    def to_array(self) -> np.ndarray:
        """Return status fields as a float32 numpy array of shape (10,)."""
        return np.array(
            [
                self.level,
                self.gold,
                self.hp,
                self.max_hp,
                self.strength,
                self.max_strength,
                self.armor,
                self.exp_level,
                self.exp,
                self.hungry_state,
            ],
            dtype=np.float32,
        )


def parse_status(status_line: str) -> Optional[RogueStatus]:
    """Parse row 23 into a RogueStatus object, or None if not a status line."""
    match = _STATUS_RE.search(status_line)
    if not match:
        return None

    return RogueStatus(
        level=int(match.group(1)),
        gold=int(match.group(2)),
        hp=int(match.group(3)),
        max_hp=int(match.group(4)),
        strength=int(match.group(5)),
        max_strength=int(match.group(6)),
        armor=int(match.group(7)),
        exp_level=int(match.group(8)),
        exp=int(match.group(9)),
        hungry_state=_HUNGER_MAP.get(match.group(10), 0),
    )


def screen_to_array(lines: list[str]) -> np.ndarray:
    """Convert 24x80 screen strings into a (24, 80) uint8 ASCII array."""
    padded = "".join(
        line[:80].ljust(80) if len(line) < 80 else line[:80]
        for line in lines[:24]
    )
    if len(lines) < 24:
        padded += " " * (80 * (24 - len(lines)))
    raw = padded.encode("ascii", errors="replace")
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(24, 80).copy()
    arr[arr > 127] = ord("?")
    return arr


def extract_local_crop(
    screen: np.ndarray,
    player_pos: Optional[tuple[int, int]],
    radius: int = 5,
) -> np.ndarray:
    """Extract a (2*radius+1, 2*radius+1) uint8 crop centered on the player."""
    size = 2 * radius + 1
    crop = np.full((size, size), ord(" "), dtype=np.uint8)
    if player_pos is None:
        return crop

    pr, pc = player_pos
    r0, r1 = pr - radius, pr + radius + 1
    c0, c1 = pc - radius, pc + radius + 1

    sr0, sr1 = max(1, r0), min(23, r1)
    sc0, sc1 = max(0, c0), min(80, c1)
    if sr0 >= sr1 or sc0 >= sc1:
        return crop

    dr0, dc0 = sr0 - r0, sc0 - c0
    crop[dr0 : dr0 + (sr1 - sr0), dc0 : dc0 + (sc1 - sc0)] = screen[sr0:sr1, sc0:sc1]
    return crop


def find_player(lines: list[str]) -> Optional[tuple[int, int]]:
    """Return (row, col) of '@' in the map area (rows 1..22), or None."""
    for r in range(1, min(23, len(lines))):
        c = lines[r].find("@")
        if c != -1:
            return (r, c)
    return None


def parse_killer(lines: list[str]) -> Optional[str]:
    """Return what killed the player, read from the tombstone, or None.

    Rogue 5.4 draws "killed by a" on one tombstone row and the killer's
    name centred on the next (rip.c ``death``), e.g. "centaur" or
    "starvation".  Returns None when the screen is not a tombstone.
    """
    for row, line in enumerate(lines[:-1]):
        if "killed by" in line.lower():
            killer = lines[row + 1].replace("|", " ").strip().lower()
            return killer or None
    return None


def is_death_or_end_screen(lines: list[str]) -> bool:
    """Detect tombstone / scoreboard / press-return end screens."""
    joined_lower = "\n".join(lines).lower()
    if "press return to continue" in joined_lower:
        return True
    # The winner screen.  Its "--Press space to continue--" prompt is not
    # used as the marker, because inventory overlays show the same prompt.
    if "made it to the light of day" in joined_lower:
        return True
    if "rest" in joined_lower and "peace" in joined_lower and "____" in joined_lower:
        return True
    if "killed by" in joined_lower and "gold" in joined_lower:
        return True
    if "top ten" in joined_lower:
        return True
    return False
