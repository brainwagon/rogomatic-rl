"""Per-episode outcome records and summary statistics.

Shared by training (rolling statistics over recent episodes) and fixed
evaluation, so both report play quality in the same reward-independent
terms: how deep a game got, how long it lasted and how it ended.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class EpisodeRecord:
    """The outcome of one finished game."""

    level: int  # dungeon level the game ended on
    steps: int  # env steps taken
    gold: int
    exp: int  # experience points
    exp_level: int
    ret: float  # undiscounted return; changes scale with the reward shaping
    end: str  # "death" or "time_limit"
    killed_by: Optional[str] = None  # tombstone killer, deaths only
    seed: Optional[int] = None  # dungeon number, fixed evaluation only

    @classmethod
    def from_info(
        cls, info: Mapping[str, Any], ret: float, seed: Optional[int] = None
    ) -> "EpisodeRecord":
        """Build a record from a RogueEnv final-step ``info`` dict."""
        killed_by = info.get("killed_by")
        return cls(
            level=int(info["level"]),
            steps=int(info["steps"]),
            gold=int(info["gold"]),
            exp=int(info["exp"]),
            exp_level=int(info.get("exp_level", 1)),
            ret=float(ret),
            end=str(info.get("end") or "death"),
            killed_by=str(killed_by) if killed_by else None,
            seed=seed,
        )


def summarize(records: Iterable[EpisodeRecord]) -> dict[str, Any]:
    """Summary statistics over a set of finished games (empty dict if none)."""
    recs = list(records)
    if not recs:
        return {}
    levels = np.array([r.level for r in recs])
    deaths = [r for r in recs if r.end == "death"]
    return {
        "episodes": len(recs),
        "level_mean": float(levels.mean()),
        "level_median": float(np.median(levels)),
        "level_max": int(levels.max()),
        "steps_mean": float(np.mean([r.steps for r in recs])),
        "gold_mean": float(np.mean([r.gold for r in recs])),
        "exp_mean": float(np.mean([r.exp for r in recs])),
        "exp_level_mean": float(np.mean([r.exp_level for r in recs])),
        "return_mean": float(np.mean([r.ret for r in recs])),
        "death_share": len(deaths) / len(recs),
        "time_limit_share": sum(r.end == "time_limit" for r in recs) / len(recs),
        # Games ending on each dungeon level, and what killed the player.
        "level_counts": {
            str(k): v for k, v in sorted(Counter(r.level for r in recs).items())
        },
        "killers": dict(
            Counter(r.killed_by or "unknown" for r in deaths).most_common()
        ),
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    """One-line ``key=value`` rendering of :func:`summarize` for the text log."""
    if not summary:
        return "n=0"
    return (
        f"n={summary['episodes']} "
        f"lvl_mean={summary['level_mean']:.2f} "
        f"lvl_med={summary['level_median']:g} "
        f"lvl_max={summary['level_max']} "
        f"len={summary['steps_mean']:.0f} "
        f"gold={summary['gold_mean']:.1f} "
        f"exp={summary['exp_mean']:.1f} "
        f"death={summary['death_share']:.2f} "
        f"time_limit={summary['time_limit_share']:.2f}"
    )


_STAT_FIELDS = (
    ("dungeon level", "level"),
    ("player level", "exp_level"),
    ("gold", "gold"),
    ("steps", "steps"),
)


def format_stats(records: Iterable[EpisodeRecord]) -> str:
    """A table of how deep, how strong, how rich and how long the games got.

    Mean, median, minimum and maximum of the dungeon level reached, the
    player (experience) level, gold and steps, then how many games ended
    on each dungeon level.
    """
    recs = list(records)
    if not recs:
        return "0 games"
    ends = Counter(r.end for r in recs)
    lines = [
        f"{len(recs)} games: " + ", ".join(f"{n} {e}" for e, n in sorted(ends.items())),
        f"{'':14}{'mean':>8}{'median':>8}{'min':>8}{'max':>8}",
    ]
    for label, field in _STAT_FIELDS:
        values = np.array([getattr(r, field) for r in recs])
        lines.append(
            f"{label:14}{values.mean():8.2f}{np.median(values):8g}"
            f"{values.min():8d}{values.max():8d}"
        )
    counts = sorted(Counter(r.level for r in recs).items())
    lines.append(
        "games ending on each dungeon level: "
        + " ".join(f"{level}:{n}" for level, n in counts)
    )
    return "\n".join(lines)


class EpisodeWindow:
    """The most recent ``size`` finished games."""

    def __init__(self, size: int = 100) -> None:
        self._records: deque[EpisodeRecord] = deque(maxlen=size)

    def add(self, record: EpisodeRecord) -> None:
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def mean_return(self, last: int = 20) -> Optional[float]:
        """Mean return of the last ``last`` games, or None before the first."""
        if not self._records:
            return None
        return float(np.mean([r.ret for r in list(self._records)[-last:]]))

    def summary(self) -> dict[str, Any]:
        return summarize(self._records)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one JSON object as a line, creating the file and its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def episode_row(record: EpisodeRecord, **extra: Any) -> dict[str, Any]:
    """A JSONL row for one game: ``extra`` keys first, then the record."""
    return {**extra, **asdict(record)}
