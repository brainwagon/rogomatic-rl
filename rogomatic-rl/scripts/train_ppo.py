"""CLI entrypoint for training Rog-O-Matic RL with PPO."""

from __future__ import annotations

import argparse

from rogomatic_rl.ppo import PPOConfig, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent on Rogue 5.4.")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in checkpoint-dir",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional explicit checkpoint path to resume from",
    )
    args = parser.parse_args()

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        max_episode_steps=args.max_episode_steps,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        resume=args.resume,
        resume_from=args.resume_from,
    )
    ckpt = train_ppo(cfg)
    print(f"Saved checkpoint to {ckpt}")


if __name__ == "__main__":
    main()
