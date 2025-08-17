from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Set
import numpy as np

from .cards import RANK_TO_IDX
from .combos import (
    enumerate_pairs, enumerate_triples, enumerate_quads,
    enumerate_straights, enumerate_pair_runs, Combo
)
from .encoding import PASS
from .rulesets import Ruleset

@dataclass
class FeatureConfig:
    short_hand:int = 5
    mid_hand:int = 9

def summarize_last(last:Combo|None) -> Dict[str, Any]:
    if last is None:
        return dict(last_kind="none", last_len=0, last_rank=-1, last_is_2=False)
    last_kind = last.kind
    last_len = last.length
    last_rank = int(last.hi_rank) if last.hi_rank is not None else -1
    last_is_2 = (last_kind in ("single","pair")) and (last.hi_rank == RANK_TO_IDX["2"])
    return dict(last_kind=last_kind, last_len=last_len, last_rank=last_rank, last_is_2=bool(last_is_2))

def hand_shape_features(rules:Ruleset, hand:Set[int]) -> Dict[str, Any]:
    pairs = enumerate_pairs(hand)
    triples = enumerate_triples(hand)
    quads = enumerate_quads(hand)
    straights = enumerate_straights(hand, rules.allow_two_in_straight)
    pruns = enumerate_pair_runs(hand, rules.allow_two_in_pair_run)

    twos_in_hand = sum(1 for c in hand if c//4 == RANK_TO_IDX["2"])

    max_s_len = max([c.length for c in straights], default=0)
    max_p_len = max([c.length for c in pruns], default=0)
    has_3pr = any(c.length>=3 for c in pruns)
    has_4pr = any(c.length>=4 for c in pruns)

    return dict(
        n_pairs=len(pairs),
        n_triples=len(triples),
        n_quads=len(quads),
        n_straights=len(straights),
        n_pruns=len(pruns),
        max_s_len=max_s_len,
        max_p_len=max_p_len,
        has_quad = len(quads)>0,
        has_3pr = has_3pr,
        has_4pr = has_4pr,
        twos_in_hand=twos_in_hand,
    )

def global_context_features(seen:Set[int], players_left:List[int], cur:int, lead:int, passed:List[bool]) -> Dict[str, Any]:
    seen_twos = sum(1 for c in seen if c//4 == RANK_TO_IDX["2"])
    others_left = [players_left[(cur+i)%4] for i in range(1,4)]
    opp_min_left = int(min(others_left))
    opp_max_left = int(max(others_left))
    opp_sum_left = int(sum(others_left))
    opp_passed = int(sum(passed[(cur+i)%4] for i in range(1,4)))
    return dict(
        seen_twos=seen_twos,
        opp_min_left=opp_min_left,
        opp_max_left=opp_max_left,
        opp_sum_left=opp_sum_left,
        opp_passed=opp_passed,
        is_leader = (cur==lead)
    )

def bin_hand_size(n:int, cfg:FeatureConfig) -> str:
    if n <= cfg.short_hand: return "S"
    if n <= cfg.mid_hand: return "M"
    return "L"

def features_from_env(env, cfg:FeatureConfig|None=None) -> Dict[str, Any]:
    if cfg is None: cfg = FeatureConfig()

    hand = env.hands[env.cur]
    f = {}
    f["hand_size"] = len(hand)
    f["hand_bucket"] = bin_hand_size(f["hand_size"], cfg)
    f.update(hand_shape_features(env.rules, hand))
    f.update(global_context_features(env.seen, [len(h) for h in env.hands], env.cur, env.lead, env.passed))
    f.update(summarize_last(env.last_combo))
    f["first_trick"] = bool(env.first_trick and (env.cur==env.lead))
    return f

def action_family_from_index(a:int) -> str:
    if a == 0: return "pass"
    if 1 <= a <= 52: return "single"
    if 53 <= a <= 65: return "pair"
    if 66 <= a <= 78: return "triple"
    if 79 <= a <= 91: return "quad"
    if 92 <= a <= 146: return "straight"
    if 147 <= a <= 212: return "pair_run"
    return "unknown"

def action_semantic(a:int, env) -> Dict[str, Any]:
    fam = action_family_from_index(a)
    d = {"action_family": fam}
    last = env.last_combo
    if last is None:
        return d
    if fam in ("single","pair","triple","quad"):
        if last.kind==fam:
            d["beats_by"] = 1
        else:
            d["beats_by"] = 0
    elif fam in ("straight","pair_run"):
        d["length_match"] = int(last.kind==fam and getattr(last,"length",0)>0)
    return d
