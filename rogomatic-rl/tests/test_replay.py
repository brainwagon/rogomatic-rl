"""Tests for recording a game and rendering it as an HTML replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rogomatic_rl import ACTIONS
from rogomatic_rl.episodes import EpisodeRecord
from rogomatic_rl.model import RogueActorCritic
from rogomatic_rl.replay import (
    Recording,
    decode,
    describe_outcome,
    encode,
    load,
    record_best,
    record_game,
    render_html,
    save,
)


def _outcome(level: int = 3, gold: int = 50, steps: int = 400, **kw) -> EpisodeRecord:
    fields = dict(exp=10, exp_level=2, ret=1.0, end="death", killed_by="hobgoblin", seed=1000)
    fields.update(kw)
    return EpisodeRecord(level=level, steps=steps, gold=gold, **fields)


def _recording() -> Recording:
    rec = Recording()
    rec.add_frame("\n".join(["first message", "  ---", "  |@|", "  ---"]))
    rec.add_frame("\n".join(["", "  ---", "  |@|   H", "  ---"]), key="l")
    rec.add_frame("\n".join(["You hit", "  ---", "  |@|  H", "  ---"]), key="s")
    rec.outcome = _outcome()
    return rec


def test_encoding_stores_only_changed_rows_and_round_trips() -> None:
    rec = _recording()
    data = encode(rec)

    assert len(data["frames"][0]) == 24  # the first screen is stored whole
    assert data["frames"][1] == [0, "", 2, "  |@|   H"]
    assert data["frames"][2] == [0, "You hit", 2, "  |@|  H"]
    assert data["keys"] == "ls"

    restored = decode(json.loads(json.dumps(data)))
    assert restored.frames == rec.frames
    assert restored.keys == rec.keys
    assert restored.outcome == rec.outcome


def test_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "games" / "game.json"
    save(_recording(), path)

    assert load(path).frames == _recording().frames


def test_decode_rejects_an_unknown_version() -> None:
    with pytest.raises(ValueError, match="version"):
        decode({"v": 99, "frames": [], "keys": ""})


def test_best_game_is_the_deepest_then_the_richest() -> None:
    def scored(**kw) -> tuple[int, int, int]:
        return Recording(outcome=_outcome(**kw)).score()

    assert scored(level=4, gold=0) > scored(level=3, gold=900)
    assert scored(level=3, gold=60) > scored(level=3, gold=50)
    assert Recording().score() == (0, 0, 0)


def test_describe_outcome_reads_as_a_sentence() -> None:
    assert describe_outcome(_outcome(level=7, gold=265, steps=1214, killed_by="rattlesnake", seed=1046)) == (
        "Dungeon 1046: reached level 7 with 265 gold, killed by a rattlesnake after 1,214 moves"
    )
    assert "killed by an emu" in describe_outcome(_outcome(killed_by="emu"))
    assert "died of starvation" in describe_outcome(_outcome(killed_by="starvation"))
    assert "ran out of time" in describe_outcome(_outcome(end="time_limit", killed_by=None))
    assert describe_outcome(_outcome(seed=None)).startswith("reached level 3")


def test_html_snippet_is_one_self_contained_block() -> None:
    snippet = render_html(_recording(), element_id="rogue-1", caption="A <test> game")

    # A blank line would end the raw-HTML block in Markdown.
    assert "\n\n" not in snippet
    assert snippet.startswith('<div id="rogue-1"')
    assert snippet.rstrip().endswith("</div>")
    # Nothing is fetched from anywhere.
    assert "http" not in snippet and " src=" not in snippet
    # Everything is scoped to the id, so several replays can share a page.
    assert "__" not in snippet.split('class="rr-data">')[0]
    assert snippet.count(".rogue-replay#rogue-1 ") >= 10
    # No line starts with "#": Markdown pipelines may escape it as a heading.
    assert not any(line.startswith("#") for line in snippet.splitlines())
    assert "A &lt;test&gt; game" in snippet

    # The embedded data is the recording.
    payload = snippet.split('class="rr-data">')[1].split("</script>")[0]
    assert decode(json.loads(payload)).frames == _recording().frames


def test_screen_text_cannot_break_out_of_the_data_script() -> None:
    rec = Recording()
    rec.add_frame("</script><script>alert(1)</script><!-- x")

    snippet = render_html(rec)
    payload = snippet.split('class="rr-data">')[1].split("</script>")[0]

    assert "<" not in payload  # nothing in the data can open or close a tag
    assert json.loads(payload)["frames"][0][0] == "</script><script>alert(1)</script><!-- x"


def test_html_snippet_rejects_an_unsafe_id() -> None:
    with pytest.raises(ValueError, match="element_id"):
        render_html(_recording(), element_id='x" onload="y')


def test_record_game_captures_every_screen_and_the_outcome() -> None:
    agent = RogueActorCritic(num_actions=len(ACTIONS))

    rec = record_game(agent, seed=1000, max_steps=20)

    assert rec.outcome is not None and rec.outcome.seed == 1000
    # One screen at the start and one per move.
    assert len(rec.frames) == rec.outcome.steps + 1
    assert len(rec.keys) == rec.outcome.steps
    assert all(key in ACTIONS for key in rec.keys)
    assert all(len(frame) == 24 for frame in rec.frames)
    assert any("@" in row for row in rec.frames[0])
    assert "Level: 1" in rec.frames[0][23]


def test_record_best_keeps_the_best_of_several_dungeons(tmp_path: Path) -> None:
    ckpt = tmp_path / "agent.pt"
    agent = RogueActorCritic(num_actions=len(ACTIONS))
    torch.save({"model_state_dict": agent.state_dict()}, ckpt)

    best, outcomes = record_best(ckpt, seeds=[1000, 1001, 1002], max_steps=15, workers=2)

    assert [o.seed for o in outcomes] == [1000, 1001, 1002]
    top = max(outcomes, key=lambda o: (o.level, o.gold, o.steps))
    assert best.outcome == top
    assert len(best.frames) == top.steps + 1
