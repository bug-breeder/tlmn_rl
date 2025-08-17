from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
import numpy as np
from collections import defaultdict
from .env import TLMNEnv
from .baselines import GreedyAgent
from .encoding import PASS

@dataclass
class EvalResult:
    win_rate: float
    avg_reward: float
    total_episodes: int
    per_position: List[float]

def play_match(env:TLMNEnv, agents:List[Callable]) -> int:
    # agents: list of callables: action = agent(obs, mask)
    obs, mask = env.reset()
    rewards = [0.0,0.0,0.0,0.0]
    cur = env.cur
    while True:
        a = agents[cur](obs, mask)
        step = env.step(a)
        rewards[cur] += step.reward
        if step.done:
            return env.winner
        obs, mask = step.obs, step.mask
        cur = env.cur

def evaluate(env_factory, policy_agent:Callable, episodes:int=1000, opponent="greedy") -> EvalResult:
    if opponent=="greedy":
        opp = GreedyAgent()
        def wrap(a): 
            def f(obs,mask): return a.act(obs,mask)
            return f
        agents = [wrap(opp), wrap(opp), wrap(opp), wrap(opp)]
    else:
        raise NotImplementedError
    wins = 0
    total_r = 0.0
    pos_counts = [0,0,0,0]
    for ep in range(episodes):
        env = env_factory()
        # place our policy at a random seat to test rotation invariance
        seat = np.random.randint(0,4)
        def policy_fn(obs,mask): return policy_agent.act(obs,mask)
        agents[seat] = policy_fn
        winner = play_match(env, agents)
        if winner==seat: wins += 1
        # We don't have per-step rewards per seat in this simple eval; using win rate mainly
    return EvalResult(win_rate=wins/episodes, avg_reward=0.0, total_episodes=episodes, per_position=[])

class StrategyAggregator:
    """Collects contexts -> action stats to produce interpretable heuristics."""
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))

    def observe(self, obs, mask, action):
        # Very simple context: last_type + hand size bucket + have_bomb
        last_type = np.argmax(obs[52+52+4+4+4:52+52+4+4+4+7])
        hand_size = int((1.0 - obs[:52].sum()/13.0)*13)
        bucket = "L" if hand_size<=5 else ("M" if hand_size<=9 else "H")
        ctx = f"last_type={last_type}|hs={bucket}"
        self.counts[ctx][action]+=1

    def dump(self) -> Dict:
        out = {}
        for ctx, d in self.counts.items():
            total = sum(d.values())
            top = sorted(d.items(), key=lambda x:-x[1])[:5]
            out[ctx] = { "total": total, "top_actions": top }
        return out
