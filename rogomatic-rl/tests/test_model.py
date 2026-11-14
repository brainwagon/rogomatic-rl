"""Tests for RogueActorCritic and the PPO training loop."""

from __future__ import annotations

from pathlib import Path

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
