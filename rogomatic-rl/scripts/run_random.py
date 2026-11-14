"""Run a baseline random agent in RogueEnv."""

from __future__ import annotations

import argparse
import time

from rogomatic_rl import RogueEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a random agent on Rogue 5.4.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Render ANSI frames to stdout")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between rendered steps")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = RogueEnv(max_steps=args.max_steps, render_mode=render_mode)

    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
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
