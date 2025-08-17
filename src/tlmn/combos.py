from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from .cards import RANKS, SUITS, RANK_TO_IDX, SUIT_TO_IDX, card_id, decode_card, sort_cards_by_rank_suit

# Combination semantics
# type: "pass", "single", "pair", "triple", "quad", "straight", "pair_run"
@dataclass(frozen=True)
class Combo:
    kind: str
    cards: Tuple[int, ...]           # specific card ids (canonical selection for runs)
    length: int = 1                  # for runs
    hi_rank: Optional[int] = None    # highest rank (index), used for ordering
    hi_suit: Optional[int] = None    # only used for single tie-break

    def __str__(self) -> str:
        if self.kind == "pass":
            return "PASS"
        return f"{self.kind.upper()}[{','.join(map(lambda c: str(c), self.cards))}]"

def single(c:int) -> Combo:
    r,s = decode_card(c)
    return Combo(kind="single", cards=(c,), length=1, hi_rank=r, hi_suit=s)

def pair(cards:Tuple[int,int]) -> Combo:
    r1,s1 = decode_card(cards[0])
    r2,s2 = decode_card(cards[1])
    assert r1==r2
    hi = max(s1,s2)
    return Combo(kind="pair", cards=tuple(sorted(cards)), length=1, hi_rank=r1, hi_suit=hi)

def triple(cards:Tuple[int,int,int]) -> Combo:
    r = decode_card(cards[0])[0]
    assert all(decode_card(c)[0]==r for c in cards)
    return Combo(kind="triple", cards=tuple(sorted(cards)), length=1, hi_rank=r)

def quad(cards:Tuple[int,int,int,int]) -> Combo:
    r = decode_card(cards[0])[0]
    assert all(decode_card(c)[0]==r for c in cards)
    return Combo(kind="quad", cards=tuple(sorted(cards)), length=1, hi_rank=r)

def straight(cards:Tuple[int,...]) -> Combo:
    # cards are canonical: strictly increasing ranks (no 2), one card per rank
    rs = [decode_card(c)[0] for c in cards]
    assert all(rs[i]+1==rs[i+1] for i in range(len(rs)-1))
    return Combo(kind="straight", cards=tuple(cards), length=len(cards), hi_rank=rs[-1])

def pair_run(cards:Tuple[int,...]) -> Combo:
    # cards length even, consist of pairs for consecutive ranks (no 2)
    assert len(cards)%2==0
    L = len(cards)//2
    ranks = sorted(list({decode_card(c)[0] for c in cards}))
    assert all(ranks[i]+1==ranks[i+1] for i in range(len(ranks)-1))
    return Combo(kind="pair_run", cards=tuple(cards), length=L, hi_rank=ranks[-1])

# Enumerators from a given hand (set of card ids)
def enumerate_singles(hand:Set[int]) -> List[Combo]:
    return [single(c) for c in sorted(hand)]

def enumerate_pairs(hand:Set[int]) -> List[Combo]:
    buf = []
    ranks: Dict[int, List[int]] = {}
    for c in hand:
        r,s = decode_card(c)
        ranks.setdefault(r, []).append(c)
    for r, cs in ranks.items():
        if len(cs) >= 2:
            cs_sorted = sorted(cs, key=lambda c: decode_card(c)[1])
            buf.append(pair((cs_sorted[0], cs_sorted[1])))
    return buf

def enumerate_triples(hand:Set[int]) -> List[Combo]:
    buf = []
    ranks: Dict[int, List[int]] = {}
    for c in hand:
        r,_ = decode_card(c)
        ranks.setdefault(r, []).append(c)
    for r, cs in ranks.items():
        if len(cs) >= 3:
            cs_sorted = sorted(cs, key=lambda c: decode_card(c)[1])
            buf.append(triple((cs_sorted[0], cs_sorted[1], cs_sorted[2])))
    return buf

def enumerate_quads(hand:Set[int]) -> List[Combo]:
    buf = []
    ranks: Dict[int, List[int]] = {}
    for c in hand:
        r,_ = decode_card(c)
        ranks.setdefault(r, []).append(c)
    for r, cs in ranks.items():
        if len(cs) == 4:
            buf.append(quad(tuple(sorted(cs, key=lambda c: decode_card(c)[1]))))
    return buf

def _ranks_to_canonical_straight(hand:Set[int], start:int, L:int) -> Optional[Combo]:
    # choose the lexicographically smallest suit per rank
    seq = []
    for r in range(start, start+L):
        cs = [c for c in hand if decode_card(c)[0]==r]
        if not cs:
            return None
        cmin = sorted(cs, key=lambda c: decode_card(c)[1])[0]
        seq.append(cmin)
    return straight(tuple(seq))

def enumerate_straights(hand:Set[int], allow_two:bool=False, min_len:int=3, max_len:int=12) -> List[Combo]:
    buf = []
    lo = 0
    hi = 12 if allow_two else 11  # hi is last rank index allowed in run (A=11 if no 2)
    Lmax = min(max_len, hi - lo + 1)
    for L in range(min_len, Lmax+1):
        for start in range(lo, hi-L+2):
            c = _ranks_to_canonical_straight(hand, start, L)
            if c: buf.append(c)
    return buf

def _ranks_to_canonical_pair_run(hand:Set[int], start:int, L:int) -> Optional[Combo]:
    # need at least two suits in each rank
    seq = []
    for r in range(start, start+L):
        cs = [c for c in hand if decode_card(c)[0]==r]
        if len(cs) < 2:
            return None
        cs_sorted = sorted(cs, key=lambda c: decode_card(c)[1])
        seq.extend([cs_sorted[0], cs_sorted[1]])
    return pair_run(tuple(seq))

def enumerate_pair_runs(hand:Set[int], allow_two:bool=False, min_len:int=2, max_len:int=12) -> List[Combo]:
    buf = []
    lo = 0
    hi = 12 if allow_two else 11
    Lmax = min(max_len, hi - lo + 1)
    for L in range(min_len, Lmax+1):
        for start in range(lo, hi-L+2):
            c = _ranks_to_canonical_pair_run(hand, start, L)
            if c: buf.append(c)
    return buf

def combo_key_for_compare(c:Combo) -> Tuple:
    """Key used to compare combos of same kind/length; larger tuple means stronger."""
    if c.kind == "single":
        return (c.hi_rank, c.hi_suit)
    elif c.kind in ("pair","triple","quad"):
        return (c.hi_rank, )
    elif c.kind in ("straight","pair_run"):
        return (c.length, c.hi_rank)
    else:
        return (-1, )

def stronger_same_type(a:Combo, b:Combo) -> bool:
    assert a.kind == b.kind
    if a.kind in ("straight","pair_run"):
        if a.length != b.length:
            return False
    return combo_key_for_compare(a) > combo_key_for_compare(b)
