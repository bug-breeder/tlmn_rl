from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Rank order in TLMN: 3 < 4 < ... < K < A < 2
RANKS = ["3","4","5","6","7","8","9","T","J","Q","K","A","2"]
RANK_TO_IDX = {r:i for i,r in enumerate(RANKS)}
IDX_TO_RANK = {i:r for r,i in RANK_TO_IDX.items()}

# Suit order used for tie-breaking: ♠ < ♣ < ♦ < ♥  (hearts highest)
SUITS = ["S","C","D","H"]
SUIT_TO_IDX = {s:i for i,s in enumerate(SUITS)}
IDX_TO_SUIT = {i:s for s,i in SUIT_TO_IDX.items()}

def card_id(rank_idx:int, suit_idx:int) -> int:
    return rank_idx*4 + suit_idx

def decode_card(c:int) -> Tuple[int,int]:
    return c//4, c%4

def card_str(c:int) -> str:
    r,s = decode_card(c)
    suit_map = {"S":"♠","C":"♣","D":"♦","H":"♥"}
    return f"{RANKS[r]}{suit_map[SUITS[s]]}"

def deck_ids() -> List[int]:
    return [i for i in range(52)]

def sort_cards_by_rank_suit(cards:List[int]) -> List[int]:
    return sorted(cards, key=lambda c: (c//4, c%4))

def has_card(cards:set[int], target:int) -> bool:
    return target in cards

THREE_SPADES = card_id(RANK_TO_IDX["3"], SUIT_TO_IDX["S"])
HEART_TWO = card_id(RANK_TO_IDX["2"], SUIT_TO_IDX["H"])
