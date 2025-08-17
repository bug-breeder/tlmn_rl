from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .combos import Combo, stronger_same_type

@dataclass
class Ruleset:
    allow_two_in_straight: bool = False
    allow_two_in_pair_run: bool = False
    pair_run_min_len: int = 2
    straight_min_len: int = 3
    # Bomb precedence (higher number = stronger tier)
    bomb_tier_3pair_run: int = 1
    bomb_tier_quad: int = 2
    bomb_tier_4pair_run: int = 3
    # First trick must contain 3♠ when you own it
    first_trick_must_contain_three_spades: bool = True

    def bomb_tier(self, c:Combo) -> int:
        if c.kind == "pair_run" and c.length >= 4:
            return self.bomb_tier_4pair_run
        if c.kind == "quad":
            return self.bomb_tier_quad
        if c.kind == "pair_run" and c.length >= 3:
            return self.bomb_tier_3pair_run
        return 0

    def can_play_over(self, new:Combo, prev:Optional[Combo]) -> bool:
        """Whether 'new' can be legally played over 'prev' under this ruleset."""
        if prev is None:
            # New trick (free lead): any non-pass combo is okay
            return new.kind != "pass"

        # Same-kind chase (non-bomb logic)
        if prev.kind == new.kind:
            if prev.kind in ("single","pair","triple","quad"):
                return stronger_same_type(new, prev)
            if prev.kind in ("straight","pair_run"):
                if new.length != prev.length:
                    return False
                return stronger_same_type(new, prev)
            return False

        # Bomb logic:
        # - If prev is SINGLE of rank 2 : can cut by quad or 3+/4+ pair_run per precedence
        # - If prev is PAIR of rank 2   : can cut by quad or 4+ pair_run
        # - If prev is 3-pair-run       : can cut by quad or >=4-pair-run
        # - Bomb vs bomb: higher tier wins; if equal tier & same kind, compare strength (length then hi_rank)
        # (Note: some tables allow 4-pair-run to beat ANYTHING. Here we keep it to "cut heo / lower bombs".)
        from .cards import RANK_TO_IDX
        if prev.kind == "single":
            # detect heo single
            if prev.hi_rank == RANK_TO_IDX["2"]:
                return self.bomb_tier(new) > 0
            return False
        if prev.kind == "pair":
            # detect đôi heo
            if prev.hi_rank == RANK_TO_IDX["2"]:
                # only quad or >=4 pair_run
                t = self.bomb_tier(new)
                return (new.kind == "quad") or (new.kind=="pair_run" and new.length>=4 and t>0)
            return False
        if prev.kind in ("straight", "triple"):
            return False  # bombs don't cut these in common rules
        if prev.kind == "pair_run":
            if prev.length >= 3:
                # Can cut by quad or longer pair_run
                if new.kind == "quad":
                    return True
                if new.kind == "pair_run":
                    if new.length > prev.length:
                        return True
                    if new.length == prev.length:
                        # same length compare strength
                        return stronger_same_type(new, prev)
                    return False
                return False
            else:
                return False

        # prev is quad: only higher-tier 4-pair-run can cut (optional)
        if prev.kind == "quad":
            if new.kind == "pair_run" and new.length >= 4:
                return True
            return False

        return False

DefaultRuleset = Ruleset()
