"""Smoke tests for RogueEnv and the Rogue 5.4 screen parser."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rogomatic_rl import ACTIONS, RogueEnv
from rogomatic_rl.env import (
    REWARD_DEATH,
    REWARD_DESCENT,
    REWARD_EXP_LEVEL,
    REWARD_EXP_POINT,
    REWARD_EXP_STEP_CAP,
    REWARD_HEALTH,
)
from rogomatic_rl.parser import RogueStatus
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
    # Step penalty, death, and the loss of all remaining health (it was full).
    assert reward == pytest.approx(-0.002 + REWARD_DEATH - REWARD_HEALTH)


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


def _reward(prev: RogueStatus, curr: RogueStatus, terminated: bool = False) -> float:
    """The reward for a step with no exploration, minus the step penalty."""
    return RogueEnv._compute_reward(prev, curr, 0, False, False, terminated) + 0.002


def test_a_descent_is_worth_no_more_than_a_death() -> None:
    start = RogueStatus(level=1)

    assert _reward(start, RogueStatus(level=2)) == pytest.approx(REWARD_DESCENT)
    assert REWARD_DESCENT <= -REWARD_DEATH


def test_health_lost_is_charged_and_health_regained_is_paid_back() -> None:
    full = RogueStatus(hp=12, max_hp=12)
    hurt = RogueStatus(hp=3, max_hp=12)

    assert _reward(full, hurt) == pytest.approx(-0.75 * REWARD_HEALTH)
    # Getting hurt and healing back nets to nothing, so it cannot be farmed.
    assert _reward(full, hurt) + _reward(hurt, full) == pytest.approx(0.0)


def test_dying_costs_the_remaining_health_as_well_as_the_death_penalty() -> None:
    hurt = RogueStatus(hp=3, max_hp=12)
    dead = RogueStatus(hp=0, max_hp=12)

    assert _reward(hurt, dead, terminated=True) == pytest.approx(
        REWARD_DEATH - 0.25 * REWARD_HEALTH
    )


def test_experience_is_rewarded_per_point_up_to_a_cap_and_per_level() -> None:
    start = RogueStatus(exp=0, exp_level=1)

    # A hobgoblin is worth 3 points.
    assert _reward(start, RogueStatus(exp=3)) == pytest.approx(3 * REWARD_EXP_POINT)
    # One huge kill is capped, so deep monsters do not swamp everything else.
    assert _reward(start, RogueStatus(exp=500)) == pytest.approx(REWARD_EXP_STEP_CAP)
    assert _reward(start, RogueStatus(exp=10, exp_level=2)) == pytest.approx(
        10 * REWARD_EXP_POINT + REWARD_EXP_LEVEL
    )
    # Losing a level (a wraith's drain) is not rewarded or charged here.
    assert _reward(RogueStatus(exp=10, exp_level=2), start) == pytest.approx(0.0)


def _status_line(
    *,
    level: int = 1,
    gold: int = 0,
    hp: int = 12,
    max_hp: int = 12,
    strength: int = 16,
    max_strength: int = 16,
    armor: int = 4,
    exp_level: int = 1,
    exp: int = 0,
) -> str:
    return (
        f"Level: {level}  Gold: {gold:<6d} Hp: {hp}({max_hp})  "
        f"Str: {strength}({max_strength})  Arm: {armor}   Exp: {exp_level}/{exp}"
    )


def _inv_overlay(*items: str) -> list[str]:
    lines = _blank_screen()
    for row, item in enumerate(items):
        lines = _with(lines, row, 40, item)
    lines = _with(lines, len(items), 40, "--Press space to continue--")
    return _with(lines, 23, 0, _status_line())


def test_auto_wield_upgrades_to_better_melee_weapon() -> None:
    before = _with(_room(), 5, 6, ")")
    before = _with(before, 23, 0, _status_line())

    picked_up = _with(_room(), 5, 6, "@")
    picked_up = _with(picked_up, 5, 5, ".")
    picked_up = _with(picked_up, 0, 0, "A long sword (f)")
    picked_up = _with(picked_up, 23, 0, _status_line())

    inv = _inv_overlay(
        "a) Some food",
        "b) +1 ring mail [4] (being worn)",
        "c) A +1,+1 mace (weapon in hand)",
        "f) A long sword",
        "g) A dagger",
    )
    wield_prompt = _with(picked_up, 0, 0, "wield what? (* for list): ")
    wielded = _with(picked_up, 0, 0, "wielding a long sword (f)")

    env = _fake_env(
        {
            "l": picked_up,
            "i": inv,
            " ": picked_up,
            "w": wield_prompt,
            "f": wielded,
        },
        before,
    )
    env._last_pos = (5, 5)
    env._status = parse_status(before[23]) or RogueStatus()

    _, _, _, _, info = env.step(3)  # 'l' onto the long sword

    assert env.client.keys == ["l", "i", " ", "w", "f"]
    assert any("wielding a long sword" in m.lower() for m in info["messages"])
    # Stepping again with no new pickup does not re-wield or wield the weaker dagger.
    env.client.keys.clear()
    env.step(0)  # 'h'
    assert "w" not in env.client.keys


def test_auto_wear_takes_off_old_armor_and_wears_strictly_better_armor() -> None:
    before = _with(_room(), 5, 6, "]")
    before = _with(before, 23, 0, _status_line(armor=4))

    picked_up = _with(_room(), 5, 6, "@")
    picked_up = _with(picked_up, 5, 5, ".")
    picked_up = _with(picked_up, 0, 0, "Chain mail (f)")
    picked_up = _with(picked_up, 23, 0, _status_line(armor=4))

    # Scale mail (base Arm 4) ties +1 ring mail [4], so only chain mail (base Arm 5) beats it.
    inv = _inv_overlay(
        "b) +1 ring mail [4] (being worn)",
        "c) A +1,+1 mace (weapon in hand)",
        "e) Scale mail",
        "f) Chain mail",
    )
    took_off = _with(picked_up, 0, 0, "was wearing b) +1 ring mail [4]")
    took_off = _with(took_off, 23, 0, _status_line(armor=0))
    wear_prompt = _with(took_off, 0, 0, "wear what? (* for list): ")
    worn = _with(took_off, 0, 0, "wearing +0 chain mail [5]")
    worn = _with(worn, 23, 0, _status_line(armor=5))

    env = _fake_env(
        {
            "l": picked_up,
            "i": inv,
            " ": picked_up,
            "T": took_off,
            "W": wear_prompt,
            "f": worn,
        },
        before,
    )
    env._last_pos = (5, 5)
    env._status = parse_status(before[23]) or RogueStatus()

    _, _, _, _, info = env.step(3)  # 'l' onto chain mail

    assert env.client.keys == ["l", "i", " ", "T", "W", "f"]
    assert env._status.armor == 5
    assert any("wearing +0 chain mail [5]" in m.lower() for m in info["messages"])

    # Next step does not swap back to +1 ring mail [4] or try scale mail (base 4).
    env.client.keys.clear()
    env.step(0)  # 'h'
    assert "T" not in env.client.keys and "W" not in env.client.keys


def test_auto_equip_waits_until_no_monster_in_view_and_stops_when_cursed() -> None:
    # A hobgoblin 'H' is in the room when the player picks up plate mail and a two-handed sword.
    with_monster = _with(_room(), 5, 6, "@")
    with_monster = _with(with_monster, 4, 4, "H")
    with_monster = _with(with_monster, 0, 0, "Plate mail (f)")
    with_monster = _with(with_monster, 23, 0, _status_line(armor=4))

    inv = _inv_overlay(
        "b) -1 ring mail [2] (being worn)",
        "c) A -1,-1 mace (weapon in hand)",
        "f) Plate mail",
        "g) A two handed sword",
    )
    clear_room = _with(_room(), 5, 6, "@")
    clear_room = _with(clear_room, 23, 0, _status_line(armor=2))
    cursed_msg = _with(clear_room, 0, 0, "you can't.  It appears to be cursed")

    env = _fake_env(
        {
            "l": with_monster,
            "i": inv,
            " ": clear_room,
            "w": cursed_msg,
            "T": cursed_msg,
            ".": clear_room,
        },
        _with(_room(), 23, 0, _status_line(armor=2)),
    )
    env._last_pos = (5, 5)
    env._status = parse_status(with_monster[23]) or RogueStatus()

    # Step 1: monster is in view, so no equip happens yet.
    env.step(3)  # 'l'
    assert env.client.keys == ["l"]

    # Step 2: monster is gone; env tries to wield and take off armor, sees both are cursed.
    env.client.keys.clear()
    env.step(8)  # '.'
    assert env.client.keys == [".", "i", " ", "w", "T"]

    # Step 3: now that both slots are known cursed, env does not retry either.
    env.client.keys.clear()
    env.step(8)  # '.'
    assert env.client.keys == ["."]


def test_quaffs_unknown_potion_when_safe_and_dismisses_call_it_prompt() -> None:
    before = _with(_room(), 5, 6, "!")
    before = _with(before, 23, 0, _status_line(hp=12, max_hp=12))

    picked_up = _with(_room(), 5, 6, "@")
    picked_up = _with(picked_up, 5, 5, ".")
    picked_up = _with(picked_up, 0, 0, "2 pink potions (f)")
    picked_up = _with(picked_up, 23, 0, _status_line(hp=12, max_hp=12))

    inv = _inv_overlay(
        "b) +1 ring mail [4] (being worn)",
        "c) A +1,+1 mace (weapon in hand)",
        "f) 2 pink potions",
    )
    quaff_prompt = _with(picked_up, 0, 0, "quaff what? (* for list): ")
    # Monster detection with no monsters leaves oi_know false and prompts "call it: ".
    call_it = _with(picked_up, 0, 0, "call it: ")
    dismissed = _with(picked_up, 0, 0, "")

    env = _fake_env(
        {
            "l": picked_up,
            "i": inv,
            " ": picked_up,
            "q": quaff_prompt,
            "f": call_it,
            "\x1b": dismissed,
            ".": dismissed,
        },
        before,
    )
    env._last_pos = (5, 5)
    env._status = parse_status(before[23]) or RogueStatus()

    env.step(3)  # 'l' onto the 2 pink potions

    assert env.client.keys == ["l", "i", " ", "q", "f", "\x1b"]

    # The remaining pink potion in the stack is now known to be tried, so it is not drunk again.
    env.client.keys.clear()
    env.step(8)  # '.'
    assert "q" not in env.client.keys


def test_quaffs_healing_potion_below_quarter_health_even_in_combat() -> None:
    combat = _with(_room(), 5, 6, "H")
    half_hp = _with(combat, 23, 0, _status_line(hp=6, max_hp=12))
    low_hp = _with(combat, 23, 0, _status_line(hp=2, max_hp=12))
    healed = _with(combat, 0, 0, "you begin to feel better")
    healed = _with(healed, 23, 0, _status_line(hp=8, max_hp=12))
    quaff_prompt = _with(low_hp, 0, 0, "quaff what? (* for list): ")

    env = _fake_env(
        {
            "l": half_hp,
            "q": quaff_prompt,
            "d": healed,
        },
        half_hp,
    )
    env._last_pos = (5, 5)
    env._status = parse_status(half_hp[23]) or RogueStatus()
    env._pack = {"d": "A potion of healing(blue)"}

    # At 50% HP (6/12), healing potion is saved.
    env.step(3)  # 'l' attack
    assert env.client.keys == ["l"]

    # When an attack step leaves HP at 2/12 (< 25%), the healing potion is drunk immediately.
    env.client._screens["l"] = low_hp
    env.client.keys.clear()
    _, reward, _, _, info = env.step(3)  # 'l' attack

    assert env.client.keys == ["l", "q", "d"]
    assert info["hp"] == 8
    assert "d" not in env._pack
    # Reward reflects net HP change from 6/12 -> 8/12 (+2/12 * REWARD_HEALTH) minus step penalty.
    assert reward == pytest.approx(-0.002 + (2 / 12) * REWARD_HEALTH)


def test_quaffs_known_buff_potions_and_ignores_known_bad_potions() -> None:
    room = _with(_room(), 23, 0, _status_line(hp=12, max_hp=12, strength=14, max_strength=16))
    quaff_prompt = _with(room, 0, 0, "quaff what? (* for list): ")
    restored = _with(room, 0, 0, "hey, this tastes great.  It make you feel warm all over")
    restored = _with(restored, 23, 0, _status_line(hp=12, max_hp=12, strength=16, max_strength=16))

    env = _fake_env(
        {
            ".": room,
            "q": quaff_prompt,
            "e": restored,
        },
        room,
    )
    env._last_pos = (5, 5)
    env._status = parse_status(room[23]) or RogueStatus()
    env._pack = {
        "d": "A potion of poison(red)",
        "e": "A murky potion",
        "f": "A potion of blindness(black)",
    }
    # Suppose murky was previously discovered to be restore strength.
    env._tried_potion_colors.add("murky")
    env._known_potion_colors["murky"] = "restore strength"

    # Since strength is 14(16), the env drinks the murky restore-strength potion and ignores poison/blindness.
    env.step(8)  # '.'
    assert env.client.keys == [".", "q", "e"]
    assert env._status.strength == 16

    # Once strength is back to 16(16), another restore-strength or bad potion is not drunk.
    env.client._screens["."] = restored
    env._pack["g"] = "A murky potion"
    env.client.keys.clear()
    env.step(8)  # '.'
    assert env.client.keys == ["."]


def test_real_rogue_auto_equips_and_quaffs_on_fixed_dungeons() -> None:
    env = RogueEnv(max_steps=10, fixed_dungeons=True)
    try:
        # Dungeon 1030 has splint mail one tile left of the player at start.
        env.reset(seed=1030)
        assert env._status.armor == 4
        _, _, terminated, _, info = env.step(0)  # 'h'
        assert terminated is False
        assert env._status.armor == 5
        assert any("wearing" in m.lower() and "splint mail" in m.lower() for m in info["messages"])

        # Dungeon 1004 has an unidentified potion one tile above the player at start.
        env.reset(seed=1004)
        _, _, terminated, _, info = env.step(2)  # 'k'
        assert terminated is False
        assert any("potion" in m.lower() for m in info["messages"])
        assert "f" not in env._pack
    finally:
        env.close()


def test_stepping_onto_hallucinated_stairs_with_no_pickup_message_enables_descend() -> None:
    # Under hallucination, an unseen staircase is drawn as a random item glyph (e.g. '?'),
    # so stepping onto it produces no pickup message from Rogue.
    before = _with(_room(), 5, 6, "?")
    after = _with(_room(), 5, 6, "@")
    after = _with(after, 5, 5, ".")

    env = _fake_env({"l": after}, before)
    env._last_pos = (5, 5)
    obs, _, _, _, _ = env.step(3)  # 'l'

    assert env._tile_under_player == "%"
    assert obs["action_mask"][10] == 1  # '>' descend stairs is legal








