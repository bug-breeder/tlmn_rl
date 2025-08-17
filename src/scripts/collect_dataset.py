import argparse, json, csv, numpy as np, torch
from pathlib import Path
from tlmn.env import TLMNEnv
from rl.networks import ActorCritic, MaskedCategorical
from tlmn.strategy_features import features_from_env, action_family_from_index, action_semantic

def load_model(obs_dim, act_dim, path, device="cpu"):
    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="tlmn_ppo.pt")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--out", type=str, default="dataset.csv")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    env = TLMNEnv(reward_shaping=False)
    obs, mask = env.reset()
    obs_dim = obs.shape[0]; act_dim = mask.shape[0]
    model = load_model(obs_dim, act_dim, args.model, device=args.device)

    path = Path(args.out)
    fout = path.open("w", newline="", encoding="utf-8")
    writer = None

    episodes = 0
    while episodes < args.episodes:
        o = torch.from_numpy(obs).float().unsqueeze(0)
        m = torch.from_numpy(mask.astype(np.float32)).float().unsqueeze(0)
        with torch.no_grad():
            logits, value, _ = model.forward(o, m, None)
            dist = MaskedCategorical(logits, m)
            a = dist.sample().item()

        feats = features_from_env(env)
        feats["action_family"] = action_family_from_index(a)
        sem = action_semantic(a, env)
        for k,v in sem.items():
            feats[f"sem_{k}"] = v

        if writer is None:
            fieldnames = list(feats.keys())
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
        writer.writerow(feats)

        step = env.step(a)
        if step.done:
            episodes += 1
            obs, mask = env.reset()
        else:
            obs, mask = step.obs, step.mask

    fout.close()
    print(f"Wrote dataset with ~{args.episodes} episodes to {path}")

if __name__ == "__main__":
    main()
