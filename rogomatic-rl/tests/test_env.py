"""Smoke tests for RogueEnv and the Rogue 5.4 screen parser."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rogomatic_rl import ACTIONS, RogueEnv
from rogomatic_rl.parser import (
    is_death_or_end_screen,
    parse_killer,
    parse_status,
    screen_to_array,
)


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


def test_is_death_or_end_screen_rogue54_tombstone() -> None:
    lines = [" " * 80 for _ in range(24)]
    lines[8] = "                       __________".ljust(80)
    lines[10] = "                     /    REST    \\".ljust(80)
    lines[11] = "                    /      IN      \\".ljust(80)
    lines[12] = "                   /     PEACE      \\".ljust(80)
    lines[16] = "                  |   killed by a    |".ljust(80)
    lines[17] = "                  |      hobgoblin   |".ljust(80)
    lines[20] = "         ________)/\\\\_//(\\/(/\\)/\\//\\/|_)_______".ljust(80)
    lines[23] = "[Press return to continue]".ljust(80)
    assert is_death_or_end_screen(lines) is True

    score_lines = [" " * 80 for _ in range(24)]
    score_lines[1] = "Top Ten Scores:".ljust(80)
    assert is_death_or_end_screen(score_lines) is True


def test_breadcrumbs_and_anti_oscillation_mask() -> None:
    env = RogueEnv(max_steps=10)
    try:
        lines = [" " * 80 for _ in range(24)]
        # Corridor '#' from (5, 4) -> (5, 5) -> (5, 6), player at (5, 5), came from (5, 4)
        row5 = list(lines[5])
        row5[4], row5[5], row5[6] = "#", "@", "#"
        lines[5] = "".join(row5)

        screen_arr = screen_to_array(lines)
        env._tile_under_player = "#"
        env._prev_pos = (5, 4)
        env._visited_mask[4, 4] = True
        env._visit_counts[4, 4] = 1
        env._visited_mask[4, 5] = True
        env._visit_counts[4, 5] = 1

        obs = env._build_obs(screen_arr, (5, 5), env._status)
        mask = obs["action_mask"]

        # Visited corridor at (5, 4) is encoded as '`' (96), unvisited at (5, 6) stays '#' (35)
        assert obs["screen"][5, 4] == ord("`")
        assert obs["screen"][5, 6] == ord("#")

        # Forward move right ('l', idx 3) is valid; backward move left ('h', idx 0) is masked
        assert mask[3] == 1
        assert mask[0] == 0

        # Search ('s', idx 9) is valid up to 6 times at a corridor bordering blank space, then masked
        assert mask[9] == 1
        env._search_counts[4, 5] = 6
        mask_after_search = env._compute_action_mask(screen_arr, (5, 5))
        assert mask_after_search[9] == 0
    finally:
        env.close()


def test_auto_eat_and_per_level_step_budget() -> None:
    env = RogueEnv(max_steps=100)
    try:
        env.reset()
        # Eating starting ration in slot 'a' should succeed, and second attempt should report no food left
        ate_first, msgs_first = env._try_auto_eat()
        assert ate_first is True
        assert any("food" in m.lower() or "yum" in m.lower() or "yuk" in m.lower() for m in msgs_first)

        ate_second, _ = env._try_auto_eat()
        assert ate_second is False
        assert env._has_food is False
    finally:
        env.close()


class _FakeClient:
    """Scripted stand-in for RoguePTYClient: one screen per key sent."""

    def __init__(self, screens: dict[str, list[str]], initial: list[str]) -> None:
        self._screens = screens
        self._lines = initial
        self.keys: list[str] = []
        self.alive = True

    def send_key(self, key: str) -> list[str]:
        self.keys.append(key)
        self._lines = self._screens.get(key, self._lines)
        top = self._lines[0].strip()
        return [top] if top else []

    def get_lines(self) -> list[str]:
        return list(self._lines)

    def record_score(self, quit_game: bool = False) -> bool:
        return False

    def close(self) -> None:
        self.alive = False


def _blank_screen() -> list[str]:
    return [" " * 80 for _ in range(24)]


def _with(lines: list[str], row: int, col: int, text: str) -> list[str]:
    out = list(lines)
    out[row] = (out[row][:col] + text + out[row][col + len(text) :])[:80].ljust(80)
    return out


def _fake_env(screens: dict[str, list[str]], initial: list[str]) -> RogueEnv:
    env = RogueEnv(max_steps=100)
    env.client.close()
    env.client = _FakeClient(screens, initial)  # type: ignore[assignment]
    env._last_lines = initial
    env._last_screen = screen_to_array(initial)
    return env


def test_inventory_overlay_is_not_an_end_screen() -> None:
    overlay = _with(_blank_screen(), 2, 50, "--Press space to continue--")
    assert is_death_or_end_screen(overlay) is False

    winner = _with(
        _blank_screen(), 10, 5, "Congratulations, you have made it to the light of day!"
    )
    winner = _with(winner, 23, 0, "--Press space to continue--")
    assert is_death_or_end_screen(winner) is True


def test_auto_eat_with_multiple_food_stacks_dismisses_overlay() -> None:
    prompt = _with(_blank_screen(), 0, 0, "eat what? (* for list): ")
    overlay = _with(_blank_screen(), 0, 50, "b) 2 rations of food")
    overlay = _with(overlay, 1, 50, "d) A slime-mold")
    overlay = _with(overlay, 2, 50, "--Press space to continue--")
    eaten = _with(_blank_screen(), 0, 0, "yum, that tasted good")

    env = _fake_env({"e": prompt, "*": overlay, " ": prompt, "b": eaten}, _blank_screen())
    ate, _ = env._try_auto_eat()

    assert ate is True
    # The overlay must be dismissed with a space before the slot letter is sent.
    assert env.client.keys == ["e", "*", " ", "b"]
    assert env._has_food is True
    assert is_death_or_end_screen(env.client.get_lines()) is False


def test_auto_eat_reports_failure_when_nothing_is_eaten() -> None:
    prompt = _with(_blank_screen(), 0, 0, "eat what? (* for list): ")
    listing = _with(_blank_screen(), 0, 0, "a) Some food")
    # The slot key is ignored: the prompt stays up and nothing is eaten.
    env = _fake_env(
        {"e": prompt, "*": listing, "a": prompt, "\x1b": _blank_screen()},
        _blank_screen(),
    )
    ate, _ = env._try_auto_eat()

    assert ate is False
    assert env._has_food is False
    assert env.client.keys[-1] == "\x1b"


def test_food_picked_up_by_walking_re_enables_auto_eat() -> None:
    before = _with(_blank_screen(), 5, 5, "@:")
    after = _with(_blank_screen(), 5, 6, "@")
    after = _with(after, 0, 0, "2 rations of food (a)")

    env = _fake_env({"l": after}, before)
    env._last_pos = (5, 5)
    env._has_food = False
    env.step(3)  # 'l'

    assert env._has_food is True


def test_death_step_gets_no_exploration_reward() -> None:
    tomb = _blank_screen()
    tomb = _with(tomb, 8, 23, "__________")
    tomb = _with(tomb, 10, 21, "/    REST    \\")
    tomb = _with(tomb, 12, 19, "/     PEACE      \\")
    tomb = _with(tomb, 16, 18, "|   killed by a    |")

    env = _fake_env({"h": tomb}, _blank_screen())
    _, reward, terminated, _, _ = env.step(0)

    assert terminated is True
    assert reward == pytest.approx(-10.002)


def test_attacking_and_resting_are_not_penalised() -> None:
    screen = _with(_blank_screen(), 5, 5, "@K")
    env = _fake_env({}, screen)
    env._last_pos = (5, 5)
    env._visited_mask[4, 5] = True
    env._visit_counts[4, 5] = 1
    env._seen_mask |= screen_to_array(screen)[1:23] != ord(" ")

    # Repeated attacks on the monster to the right, with no hp or exp change.
    for _ in range(6):
        _, reward, _, _, _ = env.step(3)  # 'l'
        assert reward == -0.002

    # Resting in place does not accumulate a revisit penalty either.
    for _ in range(6):
        _, reward, _, _, _ = env.step(8)  # '.'
        assert reward == -0.002

    # A move into empty floor that goes nowhere is still a wasted turn.
    _, reward, _, _, _ = env.step(0)  # 'h'
    assert reward == pytest.approx(-0.052)


def test_close_kills_rogue_before_closing_the_pty(monkeypatch) -> None:
    import rogomatic_rl.pty_client as pty_client

    env = RogueEnv(max_steps=10)
    env.reset()
    proc = env.client._proc
    master_fd = env.client._master_fd
    assert proc is not None and proc.poll() is None

    exited_at_close: list[bool] = []
    real_close = pty_client.os.close

    def spy_close(fd: int) -> None:
        if fd == master_fd:
            exited_at_close.append(proc.poll() is not None)
        real_close(fd)

    monkeypatch.setattr(pty_client.os, "close", spy_close)
    env.close()

    # If the PTY master closed while Rogue was alive, Rogue would auto-save
    # to ~/rogue.save.
    assert exited_at_close and all(exited_at_close)


def test_time_limit_episode_is_posted_to_shared_score_file(tmp_path) -> None:
    score_file = tmp_path / "scores" / "rogue.scr"
    env = RogueEnv(max_steps=3, score_file=score_file)
    try:
        obs, _ = env.reset()
        proc = env.client._proc
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(np.flatnonzero(obs["action_mask"])[0])
            obs, _, terminated, truncated, _ = env.step(action)

        # Rogue was allowed to write the score and exit by itself.
        assert proc is not None and proc.returncode == 0
        assert env.client.alive is False
        # The shared file exists; Rogue only writes an entry for a game that
        # beats one already listed, so a zero-gold game leaves it empty.
        assert score_file.exists()
    finally:
        env.close()


def test_no_score_file_means_no_quit_at_time_limit() -> None:
    env = RogueEnv(max_steps=3)
    try:
        obs, _ = env.reset()
        truncated = False
        while not truncated:
            action = int(np.flatnonzero(obs["action_mask"])[0])
            obs, _, terminated, truncated, _ = env.step(action)
            assert terminated is False
        # Without a score file the game is left running, as before.
        assert env.client.alive is True
    finally:
        env.close()


def _tombstone(article: str, killer: str) -> list[str]:
    tomb = _blank_screen()
    tomb = _with(tomb, 8, 23, "__________")
    tomb = _with(tomb, 10, 21, "/    REST    \\")
    tomb = _with(tomb, 12, 19, "/     PEACE      \\")
    tomb = _with(tomb, 16, 18, f"|   killed by {article:<2}   |")
    tomb = _with(tomb, 17, 18, "|" + killer.center(18) + "|")
    tomb = _with(tomb, 23, 0, "[Press return to continue]")
    return tomb


def test_parse_killer_reads_the_tombstone() -> None:
    assert parse_killer(_tombstone("a", "hobgoblin")) == "hobgoblin"
    assert parse_killer(_tombstone("an", "ice monster")) == "ice monster"
    # Starvation has no article on the tombstone.
    assert parse_killer(_tombstone("", "starvation")) == "starvation"
    assert parse_killer(_blank_screen()) is None


def test_final_step_info_says_how_the_game_ended() -> None:
    env = _fake_env({"h": _tombstone("a", "centaur")}, _blank_screen())
    _, _, terminated, _, info = env.step(0)

    assert terminated is True
    assert info["end"] == "death"
    assert info["killed_by"] == "centaur"

    env = _fake_env({}, _with(_blank_screen(), 5, 5, "@"))
    env.max_steps = 2
    _, _, _, truncated, info = env.step(8)
    assert truncated is False
    assert info["end"] is None
    _, _, _, truncated, info = env.step(8)
    assert truncated is True
    assert info["end"] == "time_limit"
    assert info["killed_by"] is None


def test_fixed_dungeons_makes_reset_seed_pick_the_dungeon() -> None:
    env = RogueEnv(max_steps=10, fixed_dungeons=True)
    try:
        first, _ = env.reset(seed=1000)
        again, _ = env.reset(seed=1000)
        other, _ = env.reset(seed=1001)
    finally:
        env.close()

    assert np.array_equal(first["screen"], again["screen"])
    assert not np.array_equal(first["screen"], other["screen"])


def test_dungeons_are_random_unless_fixed_dungeons_is_set() -> None:
    env = RogueEnv(max_steps=10)
    try:
        first, _ = env.reset(seed=1000)
        again, _ = env.reset(seed=1000)
    finally:
        env.close()

    assert not np.array_equal(first["screen"], again["screen"])


def _mask_at(lines: list[str], pos: tuple[int, int]) -> np.ndarray:
    env = _fake_env({}, lines)
    return env._compute_action_mask(screen_to_array(lines), pos)


def _room(center: str = "@") -> list[str]:
    """A 3x3 patch of room floor around the player at (5, 5)."""
    lines = _blank_screen()
    lines = _with(lines, 4, 4, "...")
    lines = _with(lines, 5, 4, f".{center}.")
    lines = _with(lines, 6, 4, "...")
    return lines


def test_only_attacking_is_legal_next_to_a_monster() -> None:
    lines = _with(_room(), 5, 6, "H")  # hobgoblin to the right

    mask = _mask_at(lines, (5, 5))

    assert [ACTIONS[i] for i in np.flatnonzero(mask)] == ["l"]


def test_policy_chooses_between_several_adjacent_monsters() -> None:
    lines = _with(_room(), 5, 6, "H")  # right
    lines = _with(lines, 4, 4, "K")  # up-left

    mask = _mask_at(lines, (5, 5))

    assert [ACTIONS[i] for i in np.flatnonzero(mask)] == ["l", "y"]


def test_monster_across_a_blocked_diagonal_does_not_force_an_attack() -> None:
    # The snake is up-right, but a wall above the player blocks that
    # diagonal, so Rogue would refuse the attack.
    lines = _with(_room(), 4, 5, "-S")

    mask = _mask_at(lines, (5, 5))

    assert mask[5] == 0  # u: the blocked diagonal toward the snake
    assert mask[0] == 1 and mask[3] == 1  # h and l: free to move


def test_mask_is_unchanged_with_no_monster_adjacent() -> None:
    lines = _with(_room(), 5, 8, "H")  # two tiles away, with floor between
    lines = _with(lines, 5, 7, ".")

    mask = _mask_at(lines, (5, 5))

    assert int(mask[:8].sum()) == 8  # every direction is open floor
