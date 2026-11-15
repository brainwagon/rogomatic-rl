"""Tests for RogueActorCritic and the PPO training loop."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rogomatic_rl import ACTIONS
from rogomatic_rl.model import RogueActorCritic
from rogomatic_rl.ppo import PPOConfig, train_ppo


def test_actor_critic_shapes_and_grad() -> None:
    model = RogueActorCritic(num_actions=len(ACTIONS))
    screen = torch.randint(32, 127, (4, 24, 80), dtype=torch.uint8)
    local = torch.randint(32, 127, (4, 11, 11), dtype=torch.uint8)
    stats = torch.randn(4, 10)
    hx = torch.zeros((1, 4, model.hidden_dim), dtype=torch.float32)

    action, log_prob, entropy, value, next_hx = model.get_action_and_value(
        screen, local, stats, hx
    )
    assert action.shape == (4,)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4,)
    assert next_hx.shape == (1, 4, model.hidden_dim)

    loss = -(log_prob.mean() + 0.01 * entropy.mean()) + 0.5 * (value ** 2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_ppo_smoke_train(tmp_path: Path) -> None:
    cfg = PPOConfig(
        total_timesteps=32,
        num_envs=2,
        num_steps=16,
        max_episode_steps=50,
        update_epochs=1,
        num_minibatches=2,
        checkpoint_dir=str(tmp_path),
        save_interval=1,
    )
    ckpt_path = train_ppo(cfg)
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert ckpt["global_step"] == 32
    assert ckpt["update"] == 1

    # Resume for another 32 steps (up to 64 total timesteps)
    resume_cfg = PPOConfig(
        total_timesteps=64,
        num_envs=2,
        num_steps=16,
        max_episode_steps=20,
        update_epochs=1,
        num_minibatches=2,
        checkpoint_dir=str(tmp_path),
        save_interval=1,
        resume=True,
    )
    resumed_ckpt_path = train_ppo(resume_cfg)
    resumed_ckpt = torch.load(resumed_ckpt_path, map_location="cpu", weights_only=True)
    assert resumed_ckpt["global_step"] == 64
    assert resumed_ckpt["update"] == 2


def test_interrupted_training_saves_completed_progress(tmp_path: Path, monkeypatch) -> None:
    import rogomatic_rl.ppo as ppo

    cfg = PPOConfig(
        total_timesteps=64,
        num_envs=2,
        num_steps=16,
        max_episode_steps=50,
        update_epochs=1,
        num_minibatches=2,
        checkpoint_dir=str(tmp_path),
        save_interval=100,  # no periodic save would happen before the interrupt
    )

    # Two gradient steps per update: interrupt in the middle of update 2.
    calls = {"n": 0}
    real_clip = ppo.nn.utils.clip_grad_norm_

    def interrupting_clip(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_clip(*args, **kwargs)

    monkeypatch.setattr(ppo.nn.utils, "clip_grad_norm_", interrupting_clip)
    ckpt_path = train_ppo(cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Only the fully completed update is recorded, so a resume repeats update 2.
    assert ckpt["update"] == 1
    assert ckpt["global_step"] == 32
    # The atomic write leaves no temp file behind.
    assert [p.name for p in tmp_path.iterdir()] == [ckpt_path.name]


def test_reward_scaler_brings_large_rewards_to_unit_scale() -> None:
    import numpy as np

    from rogomatic_rl.ppo import RewardScaler

    scaler = RewardScaler(num_envs=4, gamma=0.99)
    rng = np.random.default_rng(0)
    done = np.zeros(4, dtype=bool)
    for _ in range(2000):
        # Mostly tiny rewards with an occasional large one, like a descent.
        reward = np.where(rng.random(4) < 0.01, 10.0, -0.002)
        scaled = scaler(reward, done)

    assert scaler.scale > 1.0
    # Sign and relative size are kept: rewards are divided, not centred.
    assert scaler(np.full(4, -0.002), done)[0] < 0
    assert abs(scaled).max() <= 10.0
    assert scaler(np.full(4, 10.0), done)[0] == pytest.approx(10.0 / scaler.scale, rel=0.05)

    # The statistics survive a save and load.
    restored = RewardScaler(num_envs=4, gamma=0.99)
    restored.load_state_dict(scaler.state_dict())
    assert restored.scale == pytest.approx(scaler.scale)


def test_reward_scaler_forgets_the_return_of_a_finished_game() -> None:
    import numpy as np

    from rogomatic_rl.ppo import RewardScaler

    scaler = RewardScaler(num_envs=2, gamma=0.99)
    scaler(np.array([5.0, 5.0]), np.array([True, False]))

    assert scaler.returns[0] == 0.0
    assert scaler.returns[1] == 5.0


def test_init_from_starts_a_new_run_from_old_weights(tmp_path: Path) -> None:
    common = dict(
        num_envs=2, num_steps=16, max_episode_steps=50, update_epochs=1, num_minibatches=2
    )
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_ckpt_path = train_ppo(
        PPOConfig(total_timesteps=64, checkpoint_dir=str(old_dir), **common)
    )
    old = torch.load(old_ckpt_path, map_location="cpu", weights_only=True)
    assert old["update"] == 2
    assert "reward_scaler" in old

    # No learning (lr 0), so the new run's weights show what was loaded.
    new_ckpt_path = train_ppo(
        PPOConfig(
            total_timesteps=32,
            checkpoint_dir=str(new_dir),
            init_from=str(old_ckpt_path),
            reset_critic=True,
            learning_rate=0.0,
            **common,
        )
    )
    new = torch.load(new_ckpt_path, map_location="cpu", weights_only=True)

    # A new run: the step and update counts start again.
    assert new["update"] == 1
    assert new["global_step"] == 32
    old_w, new_w = old["model_state_dict"], new["model_state_dict"]
    assert torch.equal(old_w["actor.weight"], new_w["actor.weight"])
    assert torch.equal(old_w["gru.weight_hh_l0"], new_w["gru.weight_hh_l0"])
    # Only the value head was re-initialised.
    assert not torch.equal(old_w["critic.weight"], new_w["critic.weight"])


def test_init_from_cannot_be_combined_with_resume(tmp_path: Path) -> None:
    cfg = PPOConfig(
        total_timesteps=32, num_envs=2, num_steps=16, checkpoint_dir=str(tmp_path),
        init_from="anything.pt", resume=True,
    )
    with pytest.raises(ValueError, match="init_from"):
        train_ppo(cfg)
