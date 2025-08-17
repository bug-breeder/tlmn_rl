import argparse, json, numpy as np, torch
from tlmn.env import TLMNEnv
from tlmn.baselines import GreedyAgent
from tlmn.encoding import PASS
from rl.networks import ActorCritic

def load_model(obs_dim, act_dim, path="tlmn_ppo.pt", device="cpu"):
    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dump_strategy", type=str, default=None)
    args = p.parse_args()

    env = TLMNEnv(reward_shaping=False)
    obs, mask = env.reset()
    obs_dim = obs.shape[0]; act_dim = mask.shape[0]
    model = load_model(obs_dim, act_dim, device=args.device)

    def policy(obs, mask):
        with torch.no_grad():
            import numpy as np
            o = torch.from_numpy(obs).float().unsqueeze(0)
            m = torch.from_numpy(mask.astype(np.float32)).float().unsqueeze(0)
            logits, value, _ = model.forward(o, m, None)
            from rl.networks import MaskedCategorical
            dist = MaskedCategorical(logits, m)
            a = dist.probs.argmax(dim=-1).item()
            return int(a)

    greedy = GreedyAgent()

    wins = 0
    for ep in range(args.episodes):
        env = TLMNEnv(reward_shaping=False)
        obs, mask = env.reset()
        cur = env.cur
        # random seat for our model
        my_seat = np.random.randint(0,4)

        while True:
            if cur == my_seat:
                a = policy(obs, mask)
            else:
                a = greedy.act(obs, mask)
            step = env.step(a)
            if step.done:
                if env.winner == my_seat: wins += 1
                break
            obs, mask = step.obs, step.mask
            cur = env.cur

    print(f"Win rate vs Greedy over {args.episodes} eps: {wins/args.episodes:.3f}")

if __name__ == "__main__":
    main()
