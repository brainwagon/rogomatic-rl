"""CLI entrypoint for training Rog-O-Matic RL with PPO."""

from __future__ import annotations

import argparse

from rogomatic_rl import DEFAULT_SCORE_FILE
from rogomatic_rl.evaluate import DEFAULT_EVAL_SEED
from rogomatic_rl.ppo import PPOConfig, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent on Rogue 5.4.")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
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
    parser.add_argument(
        "--score-file",
        type=str,
        default=str(DEFAULT_SCORE_FILE),
        help="Shared Rogue high score file that finished games are posted to",
    )
    parser.add_argument(
        "--no-score-file",
        action="store_true",
        help="Do not record games in the high score list",
    )
    parser.add_argument(
        "--episode-log",
        type=str,
        default="logs/episodes.jsonl",
        help="JSONL file that gets one row per finished training game",
    )
    parser.add_argument(
        "--no-episode-log", action="store_true", help="Do not write the episode log"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=250,
        help="Run a fixed evaluation every this many updates (0 = never)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=50, help="Games per fixed evaluation"
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help="First dungeon number of the fixed evaluation set",
    )
    parser.add_argument(
        "--eval-deterministic",
        action="store_true",
        help="Evaluate with the most probable action instead of sampling",
    )
    parser.add_argument(
        "--eval-log",
        type=str,
        default="logs/evals.jsonl",
        help="JSONL file that gets one row per fixed evaluation",
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
        score_file=None if args.no_score_file else args.score_file,
        episode_log=None if args.no_episode_log else args.episode_log,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        eval_deterministic=args.eval_deterministic,
        eval_log=args.eval_log,
    )
    ckpt = train_ppo(cfg)
    print(f"Saved checkpoint to {ckpt}")


if __name__ == "__main__":
    main()
