"""Evaluate or watch a trained PPO checkpoint playing Rogue 5.4."""

from __future__ import annotations

import argparse
import time

import torch

from rogomatic_rl import ACTIONS, RogueEnv
from rogomatic_rl.model import RogueActorCritic


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Rogue PPO agent.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/rogomatic_ppo_latest.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    agent = RogueActorCritic(num_actions=len(ACTIONS))
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    render_mode = "human" if args.render else None
    env = RogueEnv(max_steps=args.max_steps, render_mode=render_mode)

    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            hx = torch.zeros((1, 1, agent.hidden_dim), dtype=torch.float32)
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                screen_t = torch.as_tensor(obs["screen"], dtype=torch.uint8).unsqueeze(0)
                local_t = torch.as_tensor(obs["local"], dtype=torch.uint8).unsqueeze(0)
                stats_t = torch.as_tensor(obs["stats"], dtype=torch.float32).unsqueeze(0)
                mask_t = torch.as_tensor(obs["action_mask"], dtype=torch.uint8).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _, hx = agent.get_action_and_value(
                        screen_t,
                        local_t,
                        stats_t,
                        hx,
                        action_mask=mask_t,
                        deterministic=args.deterministic,
                    )
                obs, reward, terminated, truncated, info = env.step(int(action.item()))
                total_reward += reward
                if args.render and args.delay > 0:
                    time.sleep(args.delay)

            print(
                f"Episode {ep}: steps={info['steps']} "
                f"level={info['level']} gold={info['gold']} "
                f"hp={info['hp']}/{info['max_hp']} exp={info['exp']} "
                f"reward={total_reward:.2f} "
                f"terminated={terminated} truncated={truncated}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
