from __future__ import annotations
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from .networks import ActorCritic

@dataclass
class PPOConfig:
    obs_dim:int
    act_dim:int=213
    hidden:int=512
    use_gru:bool=False
    gamma:float=0.99
    lam:float=0.95
    clip_ratio:float=0.2
    vf_coef:float=0.5
    ent_coef:float=0.01
    lr:float=2.5e-4
    epochs:int=4
    minibatch:int=8192
    max_grad_norm:float=0.5

class RolloutBuffer:
    def __init__(self, size:int, obs_dim:int, act_dim:int, device:str="cpu"):
        self.size = size
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.mask = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size,), dtype=torch.long, device=device)
        self.rewards = torch.zeros((size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((size,), dtype=torch.float32, device=device)
        self.logps = torch.zeros((size,), dtype=torch.float32, device=device)
        self.done = torch.zeros((size,), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, obs, mask, action, reward, value, logp, done):
        i = self.ptr
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.obs.device)
        self.mask[i] = torch.as_tensor(mask, dtype=torch.float32, device=self.mask.device)
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.logps[i] = logp
        self.done[i] = done
        self.ptr += 1

    def compute_returns_advantages(self, gamma:float, lam:float):
        # GAE-lambda
        size = self.ptr
        adv = torch.zeros(size, dtype=torch.float32, device=self.obs.device)
        lastgaelam = 0.0
        for t in reversed(range(size)):
            nonterminal = 1.0 - self.done[t]
            next_value = self.values[t+1] if t+1<size else 0.0
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + self.values[:size]
        # normalize adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

def ppo_update(model:ActorCritic, buffer:RolloutBuffer, cfg:PPOConfig, device:str="cpu"):
    adv, ret = buffer.compute_returns_advantages(cfg.gamma, cfg.lam)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    size = buffer.ptr
    idxs = torch.randperm(size)
    for epoch in range(cfg.epochs):
        for start in range(0, size, cfg.minibatch):
            end = min(size, start + cfg.minibatch)
            mb = idxs[start:end]
            obs = buffer.obs[mb]
            mask = buffer.mask[mb]
            acts = buffer.actions[mb]
            old_logp = buffer.logps[mb]
            old_val = buffer.values[mb]
            mb_adv = adv[mb]
            mb_ret = ret[mb]

            logp, ent, value = model.evaluate_actions(obs, mask, acts)

            ratio = torch.exp(logp - old_logp.detach())
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
            pg_loss = -(torch.min(ratio * mb_adv, clipped)).mean()
            v_loss = F.mse_loss(value, mb_ret)
            ent_loss = -ent.mean()
            loss = pg_loss + cfg.vf_coef * v_loss + cfg.ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
    return {"loss": float(loss.item())}
