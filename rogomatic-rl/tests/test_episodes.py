"""Tests for per-episode records, their summary and fixed evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rogomatic_rl import ACTIONS
from rogomatic_rl.episodes import (
    EpisodeRecord,
    EpisodeWindow,
    append_jsonl,
    episode_row,
    format_stats,
    format_summary,
    summarize,
)
from rogomatic_rl.evaluate import evaluate_policy
from rogomatic_rl.model import RogueActorCritic
from rogomatic_rl.ppo import PPOConfig, _finished_episodes, train_ppo


def _record(level: int, end: str = "death", killed_by: str | None = "bat", **kw) -> EpisodeRecord:
    fields = dict(steps=100, gold=10, exp=5, exp_level=1, ret=1.0)
    fields.update(kw)
    return EpisodeRecord(
        level=level, end=end, killed_by=killed_by if end == "death" else None, **fields
    )


def test_summarize_reports_depth_and_how_games_ended() -> None:
    summary = summarize(
        [
            _record(1, killed_by="hobgoblin", steps=100),
            _record(1, killed_by="hobgoblin", steps=200),
            _record(2, killed_by=None, steps=300),
            _record(8, end="time_limit", steps=400),
        ]
    )

    assert summary["episodes"] == 4
    assert summary["level_mean"] == pytest.approx(3.0)
    assert summary["level_median"] == pytest.approx(1.5)
    assert summary["level_max"] == 8
    assert summary["steps_mean"] == pytest.approx(250.0)
    assert summary["death_share"] == pytest.approx(0.75)
    assert summary["time_limit_share"] == pytest.approx(0.25)
    assert summary["level_counts"] == {"1": 2, "2": 1, "8": 1}
    assert summary["killers"] == {"hobgoblin": 2, "unknown": 1}
    # The summary is written to JSONL, so it must serialise as is.
    json.dumps(summary)


def test_summary_of_no_games_is_empty() -> None:
    assert summarize([]) == {}
    assert format_summary({}) == "n=0"


def test_format_summary_is_one_key_value_line() -> None:
    line = format_summary(summarize([_record(1), _record(4, end="time_limit")]))

    assert line == (
        "n=2 lvl_mean=2.50 lvl_med=2.5 lvl_max=4 len=100 gold=10.0 exp=5.0 "
        "death=0.50 time_limit=0.50"
    )


def test_format_stats_tabulates_depth_player_level_gold_and_steps() -> None:
    table = format_stats(
        [
            _record(1, exp_level=1, gold=0, steps=100),
            _record(3, exp_level=4, gold=50, steps=900),
            _record(8, end="time_limit", exp_level=7, gold=400, steps=2000),
        ]
    )

    assert table.splitlines() == [
        "3 games: 2 death, 1 time_limit",
        "                  mean  median     min     max",
        "dungeon level     4.00       3       1       8",
        "player level      4.00       4       1       7",
        "gold            150.00      50       0     400",
        "steps          1000.00     900     100    2000",
        "games ending on each dungeon level: 1:1 3:1 8:1",
    ]
    assert format_stats([]) == "0 games"


def test_window_keeps_only_the_most_recent_games() -> None:
    window = EpisodeWindow(size=3)
    assert window.mean_return() is None

    for level in (1, 2, 3, 4):
        window.add(_record(level, ret=float(level)))

    assert len(window) == 3
    assert window.summary()["level_counts"] == {"2": 1, "3": 1, "4": 1}
    assert window.mean_return(last=2) == pytest.approx(3.5)


def test_append_jsonl_creates_the_directory_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "episodes.jsonl"
    append_jsonl(path, episode_row(_record(2), update=7))
    append_jsonl(path, episode_row(_record(3), update=8))

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["update"], r["level"]) for r in rows] == [(7, 2), (8, 3)]
    assert rows[0]["killed_by"] == "bat"


def test_finished_episodes_reads_the_final_step_not_the_next_game() -> None:
    # Two envs; only env 1 finished.  Top-level values belong to the game
    # that autoreset has already started.
    infos = {
        "level": np.array([3, 1]),
        "final_info": {
            "level": np.array([0, 5]),
            "steps": np.array([0, 1234]),
            "gold": np.array([0, 77]),
            "exp": np.array([0, 40]),
            "exp_level": np.array([0, 4]),
            "end": np.array([None, "death"], dtype=object),
            "killed_by": np.array([None, "centaur"], dtype=object),
            "episode": {"r": np.array([0.0, 12.5])},
            "_episode": np.array([False, True]),
        },
    }

    assert _finished_episodes(infos) == [
        EpisodeRecord(
            level=5, steps=1234, gold=77, exp=40, exp_level=4, ret=12.5,
            end="death", killed_by="centaur",
        )
    ]
    assert _finished_episodes({"level": np.array([3, 1])}) == []


def test_evaluate_policy_plays_each_numbered_dungeon_once() -> None:
    agent = RogueActorCritic(num_actions=len(ACTIONS))

    # More dungeons than envs, so slots are reused for later dungeons.
    records = evaluate_policy(
        agent, episodes=5, num_envs=2, max_steps=25, seed_base=1000
    )

    assert [r.seed for r in records] == [1000, 1001, 1002, 1003, 1004]
    for r in records:
        assert r.end in ("death", "time_limit")
        assert 1 <= r.steps <= 25
        if r.end == "time_limit":
            assert r.steps == 25


def test_training_logs_entropy_ceiling_episodes_and_evaluations(
    tmp_path: Path, capsys
) -> None:
    episode_log = tmp_path / "logs" / "episodes.jsonl"
    eval_log = tmp_path / "logs" / "evals.jsonl"
    cfg = PPOConfig(
        total_timesteps=128,
        num_envs=2,
        num_steps=32,
        max_episode_steps=20,
        update_epochs=1,
        num_minibatches=2,
        checkpoint_dir=str(tmp_path / "ckpt"),
        episode_log=str(episode_log),
        eval_interval=2,
        eval_episodes=3,
        eval_log=str(eval_log),
    )
    train_ppo(cfg)
    out = capsys.readouterr().out

    update_lines = [l for l in out.splitlines() if l.startswith("update=")]
    assert len(update_lines) == 2
    fields = dict(part.split("=", 1) for part in update_lines[-1].split())
    # The ceiling is ln(legal actions), so entropy can never exceed it.
    assert 0.0 < float(fields["ent_max"]) <= np.log(len(ACTIONS))
    assert float(fields["ent"]) <= float(fields["ent_max"]) + 1e-3
    assert 0.0 <= float(fields["ent_ratio"]) <= 1.001
    assert 1.0 <= float(fields["legal"]) <= len(ACTIONS)
    assert 0.0 <= float(fields["forced"]) <= 1.0

    # 20-step games over 64 steps per env: several games have finished.
    assert any(l.startswith("episodes update=2 n=") for l in out.splitlines())
    rows = [json.loads(line) for line in episode_log.read_text().splitlines()]
    assert len(rows) >= 4
    assert {"update", "global_step", "level", "steps", "end", "ret"} <= rows[0].keys()

    # One evaluation, after update 2 of 2.
    assert sum(l.startswith("eval update=2 ") for l in out.splitlines()) == 1
    evals = [json.loads(line) for line in eval_log.read_text().splitlines()]
    assert len(evals) == 1
    assert evals[0]["source"] == "train"
    assert evals[0]["update"] == 2
    assert evals[0]["episodes"] == 3
    assert evals[0]["seed_base"] == cfg.eval_seed


def test_evaluation_is_off_by_default(tmp_path: Path, capsys) -> None:
    cfg = PPOConfig(
        total_timesteps=32,
        num_envs=2,
        num_steps=16,
        max_episode_steps=50,
        update_epochs=1,
        num_minibatches=2,
        checkpoint_dir=str(tmp_path),
    )
    train_ppo(cfg)

    assert "eval update=" not in capsys.readouterr().out
    assert [p.name for p in tmp_path.iterdir()] == ["rogomatic_ppo_latest.pt"]
