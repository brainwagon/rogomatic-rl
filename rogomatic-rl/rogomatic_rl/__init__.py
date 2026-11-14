"""Modern Rog-O-Matic RL package."""

from gymnasium.envs.registration import register

from .env import ACTIONS, DEFAULT_SCORE_FILE, RogueEnv

register(
    id="Rogue54-v0",
    entry_point="rogomatic_rl.env:RogueEnv",
    nondeterministic=True,
)

__all__ = ["ACTIONS", "DEFAULT_SCORE_FILE", "RogueEnv"]
