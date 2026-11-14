"""Recurrent dual-scale CNN Actor-Critic for Rogue 5.4."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _layer_init(
    layer: nn.Module, std: float = np.sqrt(2.0), bias_const: float = 0.0
) -> nn.Module:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    elif isinstance(layer, nn.GRU):
        for name, param in layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
    return layer


class RogueActorCritic(nn.Module):
    """Global + player-centered local CNN + GRU actor-critic for Rogue 5.4."""

    def __init__(
        self,
        num_actions: int,
        vocab_size: int = 128,
        embed_dim: int = 12,
        stats_dim: int = 10,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.char_embed = nn.Embedding(vocab_size, embed_dim)

        # Global 24x80 map encoder -> (B, 32*3*10 = 960)
        self.global_cnn = nn.Sequential(
            _layer_init(nn.Conv2d(embed_dim, 24, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 10)),
            nn.Flatten(),
        )

        # Local 11x11 player-centered encoder -> (B, 32*4*4 = 512)
        self.local_cnn = nn.Sequential(
            _layer_init(nn.Conv2d(embed_dim, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.stats_net = nn.Sequential(
            _layer_init(nn.Linear(stats_dim, 64)),
            nn.ReLU(),
        )

        self.proj = nn.Sequential(
            _layer_init(nn.Linear(960 + 512 + 64, hidden_dim)),
            nn.ReLU(),
        )

        self.gru = _layer_init(nn.GRU(hidden_dim, hidden_dim))
        self.actor = _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)
        self.critic = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        self.stats_scale: torch.Tensor
        self.register_buffer(
            "stats_scale",
            torch.tensor(
                [10.0, 100.0, 20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 100.0, 3.0],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _encode_obs(
        self, screen: torch.Tensor, local: torch.Tensor, stats: torch.Tensor
    ) -> torch.Tensor:
        g = self.char_embed(screen.long().clamp(0, 127)).permute(0, 3, 1, 2)
        l = self.char_embed(local.long().clamp(0, 127)).permute(0, 3, 1, 2)

        g_feat = self.global_cnn(g)
        l_feat = self.local_cnn(l)

        scaled_stats = stats.float() / self.stats_scale
        s_feat = self.stats_net(scaled_stats)
        return self.proj(torch.cat([g_feat, l_feat, s_feat], dim=-1))

    def _forward_gru(
        self,
        screen: torch.Tensor,
        local: torch.Tensor,
        stats: torch.Tensor,
        hx: torch.Tensor,
        done: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder + GRU.

        Accepts either single-step (B, ...) with hx (1, B, H) or
        sequence (T*B, ...) with hx (1, B, H) and done (T*B,).
        """
        batch_size = hx.shape[1]
        total = screen.shape[0]
        seq_len = total // batch_size

        feat = self._encode_obs(screen, local, stats).view(seq_len, batch_size, self.hidden_dim)
        if done is None:
            out, next_hx = self.gru(feat, hx)
            return out.reshape(total, self.hidden_dim), next_hx

        done_seq = done.view(seq_len, batch_size)
        outputs = []
        cur_hx = hx
        for t in range(seq_len):
            cur_hx = cur_hx * (1.0 - done_seq[t]).view(1, batch_size, 1)
            step_out, cur_hx = self.gru(feat[t : t + 1], cur_hx)
            outputs.append(step_out)
        out = torch.cat(outputs, dim=0).reshape(total, self.hidden_dim)
        return out, cur_hx

    def get_value(
        self,
        screen: torch.Tensor,
        local: torch.Tensor,
        stats: torch.Tensor,
        hx: torch.Tensor,
        done: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden, _ = self._forward_gru(screen, local, stats, hx, done)
        return self.critic(hidden).squeeze(-1)

    def get_action_and_value(
        self,
        screen: torch.Tensor,
        local: torch.Tensor,
        stats: torch.Tensor,
        hx: torch.Tensor,
        done: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, next_hx = self._forward_gru(screen, local, stats, hx, done)
        logits = self.actor(hidden)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e8)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(hidden).squeeze(-1)
        return action, log_prob, entropy, value, next_hx
