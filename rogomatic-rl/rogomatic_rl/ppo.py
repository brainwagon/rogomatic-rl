"""Recurrent PPO training loop for Rogue 5.4."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .env import ACTIONS
from .model import RogueActorCritic


@dataclass
class PPOConfig:
    """Hyperparameters for recurrent PPO training on Rogue54-v0."""

    total_timesteps: int = 100_000
    num_envs: int = 4
    num_steps: int = 128
    max_episode_steps: int = 1000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    seed: int = 42
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 10
    resume: bool = False
    resume_from: Optional[str] = None


def _make_env(max_episode_steps: int) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        import rogomatic_rl  # noqa: F401 - ensure Rogue54-v0 is registered in spawned workers

        env = gym.make("Rogue54-v0", max_steps=max_episode_steps)
        return gym.wrappers.RecordEpisodeStatistics(env)

    return _thunk


def train_ppo(cfg: PPOConfig) -> Path:
    """Train a recurrent RogueActorCritic agent with PPO."""
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    batch_size = cfg.num_envs * cfg.num_steps
    num_updates = max(1, cfg.total_timesteps // batch_size)
    envs_per_batch = max(1, cfg.num_envs // cfg.num_minibatches)

    env_fns = [_make_env(cfg.max_episode_steps) for _ in range(cfg.num_envs)]
    autoreset_mode = gym.vector.AutoresetMode.SAME_STEP
    envs = (
        gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=autoreset_mode)
        if cfg.num_envs > 1
        else gym.vector.SyncVectorEnv(env_fns, autoreset_mode=autoreset_mode)
    )

    agent = RogueActorCritic(num_actions=len(ACTIONS)).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    screens = torch.zeros((cfg.num_steps, cfg.num_envs, 24, 80), dtype=torch.uint8, device=device)
    locals_ = torch.zeros((cfg.num_steps, cfg.num_envs, 11, 11), dtype=torch.uint8, device=device)
    stats = torch.zeros((cfg.num_steps, cfg.num_envs, 10), dtype=torch.float32, device=device)
    action_masks = torch.ones((cfg.num_steps, cfg.num_envs, len(ACTIONS)), dtype=torch.uint8, device=device)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.float32, device=device)

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "rogomatic_ppo_latest.pt"

    global_step = 0
    start_update = 1
    best_level = 1
    best_gold = 0

    if cfg.resume or cfg.resume_from:
        resume_path = Path(cfg.resume_from) if cfg.resume_from else latest_ckpt
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=True)
            agent.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            global_step = int(ckpt.get("global_step", 0))
            start_update = int(ckpt.get("update", global_step // batch_size)) + 1
            best_level = int(ckpt.get("best_level", 1))
            best_gold = int(ckpt.get("best_gold", 0))
            if global_step < cfg.total_timesteps:
                num_updates = max(start_update, cfg.total_timesteps // batch_size)
            else:
                num_updates = (start_update - 1) + max(
                    1, cfg.total_timesteps // batch_size
                )

    initial_global_step = global_step
    start_time = time.time()
    recent_returns: list[float] = []

    try:
        obs, _ = envs.reset(seed=cfg.seed)
        next_screen = torch.as_tensor(obs["screen"], dtype=torch.uint8, device=device)
        next_local = torch.as_tensor(obs["local"], dtype=torch.uint8, device=device)
        next_stats = torch.as_tensor(obs["stats"], dtype=torch.float32, device=device)
        next_mask = torch.as_tensor(obs["action_mask"], dtype=torch.uint8, device=device)
        next_done = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
        next_hx = torch.zeros((1, cfg.num_envs, agent.hidden_dim), dtype=torch.float32, device=device)

        for update in range(start_update, num_updates + 1):
            initial_hx = next_hx.clone()

            if cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

            for step in range(cfg.num_steps):
                global_step += cfg.num_envs
                screens[step] = next_screen
                locals_[step] = next_local
                stats[step] = next_stats
                action_masks[step] = next_mask
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value, next_hx = agent.get_action_and_value(
                        next_screen,
                        next_local,
                        next_stats,
                        next_hx,
                        next_done,
                        action_mask=next_mask,
                    )
                    values[step] = value
                actions[step] = action
                logprobs[step] = logprob

                obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)

                # Bootstrap value on time-limit truncations (where episode did not terminate by death)
                final_obs_arr = infos.get("final_obs")
                if final_obs_arr is not None:
                    for i in range(cfg.num_envs):
                        if truncated[i] and not terminated[i] and final_obs_arr[i] is not None:
                            f_obs = final_obs_arr[i]
                            f_screen = torch.as_tensor(
                                f_obs["screen"], dtype=torch.uint8, device=device
                            ).unsqueeze(0)
                            f_local = torch.as_tensor(
                                f_obs["local"], dtype=torch.uint8, device=device
                            ).unsqueeze(0)
                            f_stats = torch.as_tensor(
                                f_obs["stats"], dtype=torch.float32, device=device
                            ).unsqueeze(0)
                            with torch.no_grad():
                                f_val = agent.get_value(
                                    f_screen, f_local, f_stats, next_hx[:, i : i + 1]
                                )
                            rewards[step, i] += cfg.gamma * f_val.squeeze(0)

                next_screen = torch.as_tensor(obs["screen"], dtype=torch.uint8, device=device)
                next_local = torch.as_tensor(obs["local"], dtype=torch.uint8, device=device)
                next_stats = torch.as_tensor(obs["stats"], dtype=torch.float32, device=device)
                next_mask = torch.as_tensor(obs["action_mask"], dtype=torch.uint8, device=device)
                next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

                for info_src in (infos, infos.get("final_info")):
                    if not isinstance(info_src, dict):
                        continue
                    if "level" in info_src:
                        best_level = max(best_level, int(np.max(info_src["level"])))
                    if "gold" in info_src:
                        best_gold = max(best_gold, int(np.max(info_src["gold"])))
                    if "episode" in info_src and "_episode" in info_src:
                        for i, finished in enumerate(info_src["_episode"]):
                            if finished:
                                recent_returns.append(float(info_src["episode"]["r"][i]))

            with torch.no_grad():
                next_value = agent.get_value(
                    next_screen, next_local, next_stats, next_hx, next_done
                )
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0.0
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            env_inds = np.arange(cfg.num_envs)
            last_pg_loss = 0.0
            last_v_loss = 0.0
            last_ent = 0.0

            for _ in range(cfg.update_epochs):
                np.random.shuffle(env_inds)
                for start in range(0, cfg.num_envs, envs_per_batch):
                    mb_envs = env_inds[start : start + envs_per_batch]
                    mb_screens = screens[:, mb_envs].reshape(-1, 24, 80)
                    mb_locals = locals_[:, mb_envs].reshape(-1, 11, 11)
                    mb_stats = stats[:, mb_envs].reshape(-1, 10)
                    mb_masks = action_masks[:, mb_envs].reshape(-1, len(ACTIONS))
                    mb_actions = actions[:, mb_envs].reshape(-1)
                    mb_logprobs = logprobs[:, mb_envs].reshape(-1)
                    mb_dones = dones[:, mb_envs].reshape(-1)
                    mb_advantages = advantages[:, mb_envs].reshape(-1)
                    mb_returns = returns[:, mb_envs].reshape(-1)
                    mb_hx = initial_hx[:, mb_envs]

                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                        mb_screens,
                        mb_locals,
                        mb_stats,
                        mb_hx,
                        mb_dones,
                        mb_actions,
                        action_mask=mb_masks,
                    )
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()

                    if len(mb_advantages) > 1:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    last_pg_loss = float(pg_loss.item())
                    last_v_loss = float(v_loss.item())
                    last_ent = float(entropy_loss.item())

            sps = int((global_step - initial_global_step) / max(1e-6, time.time() - start_time))
            mean_ret = (
                f"{np.mean(recent_returns[-20:]):.2f}" if recent_returns else "n/a"
            )
            print(
                f"update={update}/{num_updates} step={global_step} sps={sps} "
                f"mean_ret={mean_ret} best_lvl={best_level} best_gold={best_gold} "
                f"pg={last_pg_loss:.4f} vf={last_v_loss:.4f} ent={last_ent:.3f}",
                flush=True,
            )

            if update % cfg.save_interval == 0 or update == num_updates:
                torch.save(
                    {
                        "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "global_step": global_step,
                        "update": update,
                        "best_level": best_level,
                        "best_gold": best_gold,
                    },
                    latest_ckpt,
                )
    finally:
        envs.close()

    return latest_ckpt
