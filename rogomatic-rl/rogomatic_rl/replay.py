"""Record a game as text frames, and turn a recording into an HTML replay.

A recording is every 24x80 screen of one game plus the key that produced
it.  It is stored compactly (only the rows that changed between screens)
as JSON.  :func:`render_html` turns one into a self-contained HTML and
JavaScript snippet that replays the game, for pasting into a blog post.
"""

from __future__ import annotations

import html
import json
import multiprocessing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from .env import ACTIONS, RogueEnv
from .episodes import EpisodeRecord
from .model import RogueActorCritic

ROWS = 24
COLS = 80
FORMAT_VERSION = 1


@dataclass
class Recording:
    """One recorded game: its screens, the keys pressed and how it ended."""

    frames: list[list[str]] = field(default_factory=list)  # 24 rows per screen
    keys: list[str] = field(default_factory=list)  # keys[i] led to frames[i + 1]
    outcome: Optional[EpisodeRecord] = None

    def add_frame(self, screen: str, key: Optional[str] = None) -> None:
        rows = [row.rstrip() for row in screen.split("\n")[:ROWS]]
        rows += [""] * (ROWS - len(rows))
        self.frames.append(rows)
        if key is not None:
            self.keys.append(key)

    def score(self) -> tuple[int, int, int]:
        """How good the game was: deepest level first, then gold, then length."""
        if self.outcome is None:
            return (0, 0, 0)
        return (self.outcome.level, self.outcome.gold, self.outcome.steps)


def encode(recording: Recording) -> dict[str, Any]:
    """Compact JSON form: the first screen, then only the rows that changed."""
    frames: list[list[Any]] = []
    previous: Optional[list[str]] = None
    for rows in recording.frames:
        if previous is None:
            frames.append(list(rows))
        else:
            # A flat [row, text, row, text, ...] list of the changed rows.
            delta: list[Any] = []
            for r, (old, new) in enumerate(zip(previous, rows)):
                if old != new:
                    delta += [r, new]
            frames.append(delta)
        previous = rows
    outcome = recording.outcome
    return {
        "v": FORMAT_VERSION,
        "rows": ROWS,
        "cols": COLS,
        "outcome": None if outcome is None else asdict(outcome),
        "keys": "".join(recording.keys),
        "frames": frames,
    }


def decode(data: dict[str, Any]) -> Recording:
    """Rebuild full screens from :func:`encode`'s output."""
    if data.get("v") != FORMAT_VERSION:
        raise ValueError(f"Unsupported recording version: {data.get('v')!r}")
    recording = Recording(keys=list(data["keys"]))
    rows: list[str] = []
    for i, frame in enumerate(data["frames"]):
        if i == 0:
            rows = list(frame)
        else:
            rows = list(rows)
            for r, text in zip(frame[0::2], frame[1::2]):
                rows[r] = text
        recording.frames.append(rows)
    if data.get("outcome"):
        recording.outcome = EpisodeRecord(**data["outcome"])
    return recording


def save(recording: Recording, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(encode(recording), separators=(",", ":")))


def load(path: str | Path) -> Recording:
    return decode(json.loads(Path(path).read_text()))


def record_game(
    agent: RogueActorCritic,
    seed: Optional[int] = None,
    max_steps: int = 2000,
    deterministic: bool = False,
) -> Recording:
    """Play one game on the CPU and record every screen.

    With ``seed`` the game is played in that numbered dungeon; without it
    the dungeon is random.
    """
    env = RogueEnv(
        max_steps=max_steps, render_mode="ansi", fixed_dungeons=seed is not None
    )
    recording = Recording()
    try:
        obs, info = env.reset(seed=seed)
        recording.add_frame(env.render() or "")
        hx = torch.zeros((1, 1, agent.hidden_dim), dtype=torch.float32)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            with torch.no_grad():
                action, _, _, _, hx = agent.get_action_and_value(
                    torch.as_tensor(obs["screen"], dtype=torch.uint8).unsqueeze(0),
                    torch.as_tensor(obs["local"], dtype=torch.uint8).unsqueeze(0),
                    torch.as_tensor(obs["stats"], dtype=torch.float32).unsqueeze(0),
                    hx,
                    action_mask=torch.as_tensor(
                        obs["action_mask"], dtype=torch.uint8
                    ).unsqueeze(0),
                    deterministic=deterministic,
                )
            key = ACTIONS[int(action.item())]
            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            total_reward += reward
            recording.add_frame(env.render() or "", key=key)
        recording.outcome = EpisodeRecord.from_info(info, ret=total_reward, seed=seed)
    finally:
        env.close()
    return recording


def _record_worker(job: tuple[str, Optional[int], int, bool]) -> dict[str, Any]:
    checkpoint, seed, max_steps, deterministic = job
    torch.set_num_threads(1)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    agent = RogueActorCritic(num_actions=len(ACTIONS))
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return encode(record_game(agent, seed, max_steps, deterministic))


def record_best(
    checkpoint: str | Path,
    seeds: Sequence[Optional[int]],
    max_steps: int = 2000,
    deterministic: bool = False,
    workers: int = 8,
) -> tuple[Recording, list[EpisodeRecord]]:
    """Record one game per entry of ``seeds`` and keep the best.

    Games are played in parallel worker processes on the CPU.  Returns the
    best recording (see :meth:`Recording.score`) and every game's outcome,
    in the order of ``seeds``.
    """
    if not seeds:
        raise ValueError("record_best needs at least one game to play")
    jobs = [(str(checkpoint), seed, max_steps, deterministic) for seed in seeds]
    workers = max(1, min(workers, len(jobs)))
    if workers == 1:
        encoded = [_record_worker(job) for job in jobs]
    else:
        # Spawned, not forked: the caller may already hold a CUDA context.
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            encoded = pool.map(_record_worker, jobs, chunksize=1)
    recordings = [decode(data) for data in encoded]
    outcomes = [r.outcome for r in recordings if r.outcome is not None]
    return max(recordings, key=Recording.score), outcomes


def describe_outcome(outcome: Optional[EpisodeRecord]) -> str:
    """A reader-facing sentence about how the game went."""
    if outcome is None:
        return "A game of Rogue"
    if outcome.end == "time_limit":
        ending = "ran out of time"
    elif outcome.killed_by in ("starvation", "hypothermia"):
        ending = f"died of {outcome.killed_by}"
    elif outcome.killed_by:
        article = "an" if outcome.killed_by[0] in "aeiou" else "a"
        ending = f"killed by {article} {outcome.killed_by}"
    else:
        ending = "died"
    dungeon = f"Dungeon {outcome.seed}: " if outcome.seed is not None else ""
    return (
        f"{dungeon}reached level {outcome.level} with {outcome.gold} gold, "
        f"{ending} after {outcome.steps:,} moves"
    )


_TEMPLATE = (Path(__file__).parent / "replay_template.html").read_text


def render_html(
    recording: Recording,
    element_id: str = "rogue-replay",
    caption: Optional[str] = None,
    fps: int = 12,
) -> str:
    """A self-contained HTML + JavaScript snippet that replays ``recording``.

    ``element_id`` must be unique on the page; every style and listener is
    scoped to it.  The snippet has no blank lines, because a blank line
    would end a raw-HTML block in Markdown and let the renderer mangle the
    rest.
    """
    if not element_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"element_id must be letters, digits, - or _: {element_id!r}")
    if caption is None:
        caption = describe_outcome(recording.outcome)
    # Inline JSON must not be able to close its <script> element.
    data = json.dumps(encode(recording), separators=(",", ":"))
    data = data.replace("<", "\\u003c")
    snippet = (
        _TEMPLATE()
        .replace("__ID__", element_id)
        .replace("__CAPTION__", html.escape(caption))
        .replace("__FPS__", str(int(fps)))
        .replace("__DATA__", data)
    )
    lines = [line.rstrip() for line in snippet.splitlines()]
    snippet = "\n".join(line for line in lines if line) + "\n"
    assert "\n\n" not in snippet
    # Some Markdown pipelines escape a leading "#" (a heading marker) even
    # inside raw HTML, which would break a CSS id selector.
    assert not any(line.startswith("#") for line in snippet.splitlines())
    return snippet
