"""Evaluate or watch a trained PPO checkpoint playing Rogue 5.4.

Without ``--render`` this is a fixed evaluation: the policy plays a set of
numbered dungeons (the same ones every time), and the outcome is printed
and appended to a JSONL file so checkpoints can be compared.

With ``--render`` it plays one game at a time on screen.

With ``--record FILE`` it records a game for ``scripts/make_replay.py``,
which turns it into a replay that can be embedded in a web page.
``--record-best N`` plays N games and keeps the best one.
"""

from __future__ import annotations

import argparse
import time

import torch

from rogomatic_rl import ACTIONS, DEFAULT_SCORE_FILE, RogueEnv
from rogomatic_rl.episodes import (
    EpisodeRecord,
    append_jsonl,
    format_summary,
    summarize,
)
from rogomatic_rl.evaluate import DEFAULT_EVAL_SEED, evaluate_policy
from rogomatic_rl.model import RogueActorCritic
from rogomatic_rl.replay import describe_outcome, record_best, save


def _describe(ep: int, record: EpisodeRecord) -> str:
    ending = record.end
    if record.killed_by:
        ending += f" ({record.killed_by})"
    dungeon = f" dungeon={record.seed}" if record.seed is not None else ""
    return (
        f"Episode {ep}:{dungeon} steps={record.steps} level={record.level} "
        f"gold={record.gold} exp={record.exp_level}/{record.exp} "
        f"reward={record.ret:.2f} end={ending}"
    )


def _watch(agent: RogueActorCritic, args: argparse.Namespace) -> list[EpisodeRecord]:
    """Play games one at a time on screen."""
    env = RogueEnv(
        max_steps=args.max_steps,
        render_mode="human",
        score_file=None if args.no_score_file else args.score_file,
        fixed_dungeons=args.seed is not None,
    )
    records = []
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep if args.seed is not None else None
            obs, info = env.reset(seed=seed)
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
                if args.delay > 0:
                    time.sleep(args.delay)

            records.append(EpisodeRecord.from_info(info, ret=total_reward, seed=seed))
            print(_describe(ep + 1, records[-1]))
    finally:
        env.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Rogue PPO agent.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/rogomatic_ppo_latest.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Games to play (default: 50, or 3 with --render)",
    )
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "First dungeon number; game k plays dungeon seed+k "
            f"(default: {DEFAULT_EVAL_SEED}, or random dungeons with --render)"
        ),
    )
    parser.add_argument(
        "--num-envs", type=int, default=8, help="Games played in parallel"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out",
        type=str,
        default="logs/evals.jsonl",
        help="JSONL file that gets one row per evaluation",
    )
    parser.add_argument(
        "--no-out", action="store_true", help="Do not write the evaluation row"
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="FILE",
        help="Record a game to this JSON file for scripts/make_replay.py",
    )
    parser.add_argument(
        "--record-best",
        type=int,
        default=None,
        metavar="N",
        help=(
            "With --record: play N games, in dungeons seed..seed+N-1, and "
            "keep the one that got deepest (ties: most gold)"
        ),
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument(
        "--score-file",
        type=str,
        default=str(DEFAULT_SCORE_FILE),
        help="Rogue high score file for --render games on random dungeons",
    )
    parser.add_argument(
        "--no-score-file",
        action="store_true",
        help="Do not record --render games in the high score list",
    )
    args = parser.parse_args()
    if args.record_best is not None and not args.record:
        parser.error("--record-best needs --record FILE")
    if args.record and args.render:
        parser.error("--record and --render cannot be combined")

    if args.record:
        games = 1 if args.record_best is None else args.record_best
        if games < 1:
            parser.error("--record-best must be at least 1")
        first = DEFAULT_EVAL_SEED if args.seed is None else args.seed
        best, outcomes = record_best(
            args.checkpoint,
            seeds=[first + k for k in range(games)],
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            workers=args.num_envs,
        )
        for ep, outcome in enumerate(outcomes, start=1):
            kept = "  <- kept" if outcome == best.outcome and games > 1 else ""
            print(_describe(ep, outcome) + kept)
        save(best, args.record)
        print(f"Recorded: {describe_outcome(best.outcome)}")
        print(f"Saved {len(best.frames)} screens to {args.record}")
        return

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    agent = RogueActorCritic(num_actions=len(ACTIONS)).to(args.device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    if args.render:
        args.episodes = 3 if args.episodes is None else args.episodes
        agent.to("cpu")
        _watch(agent, args)
        return

    episodes = 50 if args.episodes is None else args.episodes
    seed_base = DEFAULT_EVAL_SEED if args.seed is None else args.seed
    records = evaluate_policy(
        agent,
        device=args.device,
        episodes=episodes,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        seed_base=seed_base,
        deterministic=args.deterministic,
    )
    for ep, record in enumerate(records, start=1):
        print(_describe(ep, record))
    summary = summarize(records)
    print(f"eval {format_summary(summary)}")
    if summary.get("killers"):
        print(f"killers {summary['killers']}")

    if not args.no_out:
        append_jsonl(
            args.out,
            {
                "source": "eval_ppo",
                "checkpoint": args.checkpoint,
                "update": ckpt.get("update"),
                "global_step": ckpt.get("global_step"),
                "seed_base": seed_base,
                "deterministic": args.deterministic,
                "max_steps": args.max_steps,
                **summary,
            },
        )
        print(f"Appended evaluation to {args.out}")


if __name__ == "__main__":
    main()
