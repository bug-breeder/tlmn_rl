# TLMN-RL (Tiến Lên Miền Nam — Reinforcement Learning Self-Play)

A compact research/starter kit to train an RL agent to **master Tiến lên miền Nam** via **self‑play PPO**.
It includes:

- A **custom game environment** (4 players, imperfect information, configurable house rules).
- A **legality‑masked action space** (213 actions: pass, 52 singles, 13 pairs, 13 triples, 13 quads, 55 straights, 66 double‑runs).
- **Self‑play PPO** (shared weights across seats) with optional LSTM, action masking, reward shaping.
- A **greedy baseline** and evaluation harness.
- **Strategy extraction** utilities (aggregate action stats to produce human‑readable heuristics).

> Default rules: **2 not allowed in straights/double‑runs**. Bomb precedence: **4 đôi thông > tứ quý > 3 đôi thông**.
> Bomb usage: **tứ quý chặt 2 đơn/đôi & 3 đôi thông**; **3 đôi thông chặt 2 đơn**; **4 đôi thông chặt 2 đơn/đôi, 3 đôi thông & tứ quý**.
> First trick must **contain 3♠** (if you start). All of these can be edited in `rulesets.py`.

## Quick start

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train PPO self-play for a small demo run (CPU ok)
python -m scripts.train_ppo_selfplay --total_steps 200000 --log_interval 2000

# Evaluate vs. heuristic baseline
python -m scripts.eval_vs_baseline --episodes 2000
```

## Repo structure

```
src/
  tlmn/
    __init__.py
    cards.py          # card encoding (ranks, suits, utils)
    combos.py         # legal combination enumerators + comparison
    rulesets.py       # house rules & comparator logic
    env.py            # single-policy multi-agent self-play environment
    encoding.py       # observation builder + action mapping/masks (size 213)
    baselines.py      # heuristic agents (greedy / safety-first)
    eval.py           # evaluation harness, Elo, logging
  rl/
    networks.py       # masked Categorical policy, optional GRU
    ppo.py            # PPO (GAE, clipping, entropy, masked loss)
    selfplay.py       # self-play loop and rollout buffers
  scripts/
    train_ppo_selfplay.py
    eval_vs_baseline.py
requirements.txt
README.md
```

## Observation & Action design (succinct)

**Observation** (default, concatenated float vector):
- `hand_52` (0/1) — your current cards.
- `seen_52` (0/1) — cards that have left hands (discard / table history).
- `players_left_4` — cards remaining per player (rotated so you are seat 0).
- `turn_onehot_4`, `lead_onehot_4`, `passes_4` — turn & trick state.
- `last_type_onehot_7` — none, single, pair, triple, straight, pair_run, quad.
- `last_key` — scalar that encodes the minimum you need to beat (rank/suit or highest rank of run).
- `last_len` — length for straight/pair-run.
- `bomb_context` — whether bombs played, #2s seen, etc. (few scalars).
Total dim ≈ **130–160** (see `encoding.py`).

**Action space** (fixed 213 ids) with **masking**:
- 0 = pass
- 1..52 = singles by absolute card (rank+♠♣♦♥)
- 53..65 = pairs by rank
- 66..78 = triples by rank
- 79..91 = quads by rank
- 92..146 = straights by (start rank, length ∈ [3..12]) — canonical suit choice
- 147..212 = pair‑runs (đôi thông) by (start rank, length ∈ [2..12]) — canonical suit choice

> For straights & pair‑runs we use a **canonical extraction** (take lexicographically smallest suits per rank).
> This keeps the action space small/stable while preserving strong play. You can enable “multi‑variant”
> expansion if you want (turn on `allow_multiple_variants=True` in `encoding.py`).

## Training recipe (baseline)

- Algorithm: **PPO** (shared policy/value; one network controls all 4 seats).
- Discount `γ=0.99`, GAE `λ=0.95`, clip `ε=0.2`, value‑loss coef `0.5`, entropy `0.01`.
- LR: **2.5e-4** (Adam), batch size **65536** (collected across many parallel envs), minibatch **8192**, 4–6 epochs.
- **Action masking**: invalid actions get `-inf` logits.
- **Reward**: terminal `+1` for 1st place, `-1` for others; optional **potential-based shaping**
  with potential `Φ(s) = -cards_left + 0.1 * (num_pairs + num_straight_spans)`; shaping `r' = r + γΦ(s')-Φ(s)`.
- Curriculum (optional): start with **2P & only singles/pairs**, add straights, then full rules + bombs.

## Strategy extraction

After training, run:
```bash
python -m scripts.eval_vs_baseline --episodes 20000 --dump_strategy ./strategy.json
```
This aggregates policy choices by contexts (e.g., “last move is single ≥A, I have ≥3 pairs and 1 bomb”) and produces interpretable rules. See `eval.py`.

---

**Disclaimer**: this is a **research starter**, optimized for readability. You can scale up (LSTM, opponent pools, NFSP/Deep‑CFR style average policies) by extending `rl/` modules.
