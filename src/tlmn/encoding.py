from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from .cards import deck_ids, decode_card, card_id, RANK_TO_IDX, SUIT_TO_IDX, THREE_SPADES, HEART_TWO
from .combos import (
    Combo, single, pair, triple, quad, straight, pair_run,
    enumerate_singles, enumerate_pairs, enumerate_triples, enumerate_quads,
    enumerate_straights, enumerate_pair_runs, stronger_same_type
)
from .rulesets import Ruleset

# Action space layout (fixed indices)
# 0                      : PASS
# 1..52                  : singles (rank-major then suit)
# 53..65 (13)            : pairs by rank
# 66..78 (13)            : triples by rank
# 79..91 (13)            : quads by rank
# 92..146 (55)           : straights by (start, length) with 3<=L<=12
# 147..212 (66)          : pair_runs by (start, length) with 2<=L<=12
PASS = 0

def single_index(rank:int, suit:int) -> int:
    return 1 + rank*4 + suit

def pair_index(rank:int) -> int:
    return 53 + rank

def triple_index(rank:int) -> int:
    return 66 + rank

def quad_index(rank:int) -> int:
    return 79 + rank

# Helpers to enumerate straight/pair_run index offsets
STRAIGHT_BASE = 92
PAIRRUN_BASE = 147

# For straights/pair_runs, we index by (start, length) within ranks [3..A] indices [0..11] (no 2)
def straight_index(start:int, length:int) -> int:
    assert 3 <= length <= 12
    idx = 0
    for L in range(3, length):
        idx += (12 - L + 1)  # count segments for shorter lengths
    idx += start  # offset within this length
    return STRAIGHT_BASE + idx

def pairrun_index(start:int, length:int) -> int:
    assert 2 <= length <= 12
    idx = 0
    for L in range(2, length):
        idx += (12 - L + 1)
    idx += start
    return PAIRRUN_BASE + idx

def straight_start_len_from_index(index:int) -> Tuple[int,int]:
    rel = index - STRAIGHT_BASE
    acc = 0
    for L in range(3, 13):
        n = (12 - L + 1)
        if rel < acc + n:
            start = rel - acc
            return start, L
        acc += n
    raise ValueError("bad straight index")

def pairrun_start_len_from_index(index:int) -> Tuple[int,int]:
    rel = index - PAIRRUN_BASE
    acc = 0
    for L in range(2, 13):
        n = (12 - L + 1)
        if rel < acc + n:
            start = rel - acc
            return start, L
        acc += n
    raise ValueError("bad pairrun index")

@dataclass
class ObsConfig:
    include_seen: bool = True
    include_context: bool = True

def build_observation(
    rules:Ruleset,
    hand:Set[int],
    seen:Set[int],
    players_left:List[int],
    cur_seat:int,
    lead_seat:int,
    passes:List[bool],
    last:Optional[Combo],
) -> np.ndarray:
    vec = []

    # hand_52
    h = np.zeros(52, dtype=np.float32)
    for c in hand: h[c] = 1.0
    vec.append(h)

    # seen_52
    s = np.zeros(52, dtype=np.float32)
    for c in seen: s[c] = 1.0
    vec.append(s)

    # players_left (rotate so cur_seat is 0)
    counts = np.array(players_left, dtype=np.float32)
    counts = np.roll(counts, -cur_seat)
    vec.append(counts / 13.0)

    # turn one-hot & lead one-hot & passes
    t = np.zeros(4, dtype=np.float32); t[0] = 1.0
    l = np.zeros(4, dtype=np.float32); l[(lead_seat - cur_seat) % 4] = 1.0
    p = np.array([passes[(cur_seat+i)%4] for i in range(4)], dtype=np.float32)
    vec.append(t); vec.append(l); vec.append(p)

    # last combo summary
    last_type = np.zeros(7, dtype=np.float32)  # none, single, pair, triple, straight, pair_run, quad
    last_key = np.array([ -1.0 ], dtype=np.float32)
    last_len = np.array([ 0.0 ], dtype=np.float32)
    if last is None:
        last_type[0] = 1.0
    else:
        kind_to_idx = {"single":1,"pair":2,"triple":3,"straight":4,"pair_run":5,"quad":6}
        last_type[kind_to_idx[last.kind]] = 1.0
        if last.kind == "single":
            last_key[0] = last.hi_rank + last.hi_suit/10.0
        else:
            last_key[0] = last.hi_rank
        last_len[0] = last.length
    vec.append(last_type); vec.append(last_key); vec.append(last_len)

    # simple bomb context (counts)
    # seen_twos: number of 2s seen; bombs_played: rough scalar
    seen_twos = sum(1 for c in seen if c//4 == RANK_TO_IDX["2"])
    bomb_context = np.array([seen_twos/4.0], dtype=np.float32)
    vec.append(bomb_context)

    return np.concatenate(vec, axis=0)

# -------- Legal action mask (size 213) + decoding back to concrete cards ----------

def legal_action_mask(rules:Ruleset, hand:Set[int], last:Optional[Combo], first_trick_must_contain_three_spades:bool, is_first_trick:bool) -> np.ndarray:
    mask = np.zeros(213, dtype=np.bool_)
    # PASS is always allowed except when you are opening a fresh trick (some tables allow pass; we forbid to ensure progress)
    mask[0] = last is not None

    # Enumerate canonical combos available from hand
    singles = enumerate_singles(hand)
    pairs = enumerate_pairs(hand)
    triples = enumerate_triples(hand)
    quads = enumerate_quads(hand)
    straights = enumerate_straights(hand, rules.allow_two_in_straight, min_len=rules.straight_min_len)
    pairruns = enumerate_pair_runs(hand, rules.allow_two_in_pair_run, min_len=rules.pair_run_min_len)

    # Helper to mark allowed, with first-trick constraint (3♠ must be in chosen cards)
    def allow_if_ok(idx:int, combo:Combo):
        if is_first_trick and first_trick_must_contain_three_spades:
            if  card_id(RANK_TO_IDX["3"], SUIT_TO_IDX["S"]) not in combo.cards:
                return
        if last is None:
            mask[idx] = True
        else:
            if rules.can_play_over(combo, last):
                mask[idx] = True

    # Singles
    for r in range(13):
        for s in range(4):
            idx = single_index(r,s)
            # Do we own that exact card?
            cid = r*4 + s
            if cid not in hand: continue
            comb = singles[[decode_card(c)[0]==r and decode_card(c)[1]==s for c in [cid]].index(True)] if cid in hand else None
            # Build explicit combo
            comb = single(cid)
            allow_if_ok(idx, comb)

    # Pairs
    for comb in pairs:
        idx = pair_index(comb.hi_rank)
        allow_if_ok(idx, comb)

    # Triples
    for comb in triples:
        idx = triple_index(comb.hi_rank)
        allow_if_ok(idx, comb)

    # Quads
    for comb in quads:
        idx = quad_index(comb.hi_rank)
        allow_if_ok(idx, comb)

    # Straights
    for comb in straights:
        # derive start and length
        rks = sorted({decode_card(c)[0] for c in comb.cards})
        start = rks[0]; L = len(rks)
        idx = straight_index(start, L)
        allow_if_ok(idx, comb)

    # Pair runs
    for comb in pairruns:
        rks = sorted({decode_card(c)[0] for c in comb.cards})
        start = rks[0]; L = len(rks)
        idx = pairrun_index(start, L)
        allow_if_ok(idx, comb)

    return mask

def decode_action_to_combo(index:int, hand:Set[int], rules:Ruleset) -> Combo:
    from .cards import decode_card
    if index == PASS:
        return Combo(kind="pass", cards=tuple())
    if 1 <= index <= 52:
        r = (index-1)//4; s = (index-1)%4
        c = r*4 + s
        return single(c)
    if 53 <= index <= 65:
        r = index - 53
        # take two lowest suits you own for that rank
        cs = sorted([c for c in hand if decode_card(c)[0]==r], key=lambda c: decode_card(c)[1])
        return pair((cs[0], cs[1]))
    if 66 <= index <= 78:
        r = index - 66
        cs = sorted([c for c in hand if decode_card(c)[0]==r], key=lambda c: decode_card(c)[1])
        return triple((cs[0], cs[1], cs[2]))
    if 79 <= index <= 91:
        r = index - 79
        cs = sorted([c for c in hand if decode_card(c)[0]==r], key=lambda c: decode_card(c)[1])
        return quad((cs[0], cs[1], cs[2], cs[3]))
    if 92 <= index <= 146:
        start, L = straight_start_len_from_index(index)
        # pick canonical straight
        seq = []
        for rr in range(start, start+L):
            cs = sorted([c for c in hand if decode_card(c)[0]==rr], key=lambda c: decode_card(c)[1])
            seq.append(cs[0])
        return straight(tuple(seq))
    if 147 <= index <= 212:
        start, L = pairrun_start_len_from_index(index)
        seq = []
        for rr in range(start, start+L):
            cs = sorted([c for c in hand if decode_card(c)[0]==rr], key=lambda c: decode_card(c)[1])
            seq.extend([cs[0], cs[1]])
        return pair_run(tuple(seq))
    raise ValueError("Bad action index")

def remove_cards_from_hand(hand:Set[int], combo:Combo):
    for c in combo.cards:
        hand.remove(c)
