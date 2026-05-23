from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Agent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(x)).squeeze(-1)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        logits = self.policy_head(features)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        dist = Categorical(logits=masked_logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        # entropy: 0 * log(0) = nan -> treat as 0 (illegal actions contribute nothing)
        entropy = dist.entropy().nan_to_num(0.0)
        value = self.value_head(features).squeeze(-1)
        return action, log_prob, entropy, value
