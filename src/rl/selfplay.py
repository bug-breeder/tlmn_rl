from __future__ import annotations
import numpy as np, torch
from typing import Callable
from .networks import ActorCritic
from .ppo import PPOConfig, RolloutBuffer, ppo_update
from tlmn.env import TLMNEnv

def train_selfplay(total_steps:int=200_000, seed:int=0, device:str="cpu", log_interval:int=2000):
    env = TLMNEnv(seed=seed)
    obs, mask = env.reset()
    obs_dim = obs.shape[0]
    act_dim = mask.shape[0]

    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(device)
    buf = RolloutBuffer(size=log_interval, obs_dim=obs_dim, act_dim=act_dim, device=device)

    step = 0
    while step < total_steps:
        # collect rollouts
        buf.ptr = 0
        for _ in range(log_interval):
            o = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            m = torch.from_numpy(mask.astype(np.float32)).float().to(device).unsqueeze(0)
            with torch.no_grad():
                action, logp, value, _ = model.act(o, m)
            a = int(action.item()); lp = float(logp.item()); v = float(value.item())
            res = env.step(a)
            buf.add(obs, mask.astype(np.float32), a, res.reward, v, lp, float(res.done))
            step += 1
            obs, mask = res.obs, res.mask
            if res.done:
                obs, mask = env.reset()

        # update
        stats = ppo_update(model, buf, cfg, device=device)
        print(f"[PPO] step={step} loss={stats['loss']:.4f}")

    # Save model
    torch.save(model.state_dict(), "tlmn_ppo.pt")
    print("Saved model to tlmn_ppo.pt")
    return model
