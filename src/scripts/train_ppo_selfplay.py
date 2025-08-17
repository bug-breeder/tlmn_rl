import argparse, torch
from rl.selfplay import train_selfplay

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log_interval", type=int, default=2000)
    args = p.parse_args()
    train_selfplay(total_steps=args.total_steps, seed=args.seed, device=args.device, log_interval=args.log_interval)

if __name__ == "__main__":
    main()
