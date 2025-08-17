from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCategorical(torch.distributions.Categorical):
    def __init__(self, logits:torch.Tensor, mask:torch.Tensor):
        # mask: 1 for legal, 0 for illegal
        very_neg = torch.finfo(logits.dtype).min / 2
        masked_logits = torch.where(mask>0, logits, very_neg)
        super().__init__(logits=masked_logits)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim:int, act_dim:int=213, hidden:int=512, use_gru:bool=False):
        super().__init__()
        self.use_gru = use_gru
        self.fc1 = nn.Linear(obs_dim, hidden)
        if use_gru:
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.pi = nn.Linear(hidden, act_dim)
            self.v = nn.Linear(hidden, 1)
        else:
            self.fc2 = nn.Linear(hidden, hidden)
            self.pi = nn.Linear(hidden, act_dim)
            self.v = nn.Linear(hidden, 1)

    def forward(self, obs, mask, h=None):
        # obs: [B,obs_dim] or [B,T,obs_dim] if GRU
        if self.use_gru and obs.dim()==3:
            x = F.relu(self.fc1(obs))
            x, h = self.gru(x, h)
            logits = self.pi(x)
            value = self.v(x).squeeze(-1)
            return logits, value, h
        else:
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            logits = self.pi(x)
            value = self.v(x).squeeze(-1)
            return logits, value, None

    def act(self, obs, mask, h=None):
        logits, value, h = self.forward(obs, mask, h)
        dist = MaskedCategorical(logits, mask)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value, h

    def evaluate_actions(self, obs, mask, actions):
        logits, value, _ = self.forward(obs, mask, None)
        dist = MaskedCategorical(logits, mask)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, value
