from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import random
import numpy as np

from .cards import deck_ids, decode_card, card_id, RANK_TO_IDX, SUIT_TO_IDX, THREE_SPADES
from .combos import Combo, single, enumerate_singles
from .rulesets import Ruleset, DefaultRuleset
from .encoding import (
    build_observation, legal_action_mask, decode_action_to_combo, remove_cards_from_hand, PASS
)

@dataclass
class StepResult:
    obs: np.ndarray
    mask: np.ndarray
    reward: float
    done: bool
    info: dict

class TLMNEnv:
    """
    Single-policy multi-agent self-play environment.
    At each step, the "active seat" (0..3) takes an action; we rotate observations so that
    the learner always sees themselves as seat 0.
    """
    def __init__(self, rules:Ruleset=DefaultRuleset, seed:int=0, reward_shaping:bool=True):
        self.rules = rules
        self.rng = random.Random(seed)
        self.n_players = 4

        # dynamic state
        self.hands: List[Set[int]] = []
        self.seen: Set[int] = set()
        self.cur: int = 0
        self.lead: int = 0
        self.passed: List[bool] = [False]*4
        self.last_combo: Optional[Combo] = None
        self.first_trick: bool = True
        self.winner: Optional[int] = None

        self.reward_shaping = reward_shaping
        self.prev_potential = 0.0

    def reset(self, seed:Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None: self.rng.seed(seed)

        # deal 13 cards to 4 players
        deck = deck_ids()
        self.rng.shuffle(deck)
        self.hands = [set(deck[i*13:(i+1)*13]) for i in range(4)]
        self.seen = set()
        self.last_combo = None
        self.passed = [False]*4
        self.winner = None
        self.first_trick = True

        # find who holds 3♠
        three_spades_holder = None
        for i in range(4):
            if (RANK_TO_IDX["3"]*4 + SUIT_TO_IDX["S"]) in self.hands[i]:
                three_spades_holder = i; break
        self.cur = self.lead = three_spades_holder if three_spades_holder is not None else 0

        self.prev_potential = self._potential(self.cur)

        obs = self._obs_for_current()
        mask = self._mask_for_current()
        return obs, mask

    def _players_left(self) -> List[int]:
        return [len(h) for h in self.hands]

    def _obs_for_current(self) -> np.ndarray:
        return build_observation(
            self.rules,
            self.hands[self.cur],
            self.seen,
            self._players_left(),
            cur_seat=self.cur,
            lead_seat=self.lead,
            passes=self.passed,
            last=self.last_combo
        )

    def _mask_for_current(self) -> np.ndarray:
        return legal_action_mask(
            self.rules,
            self.hands[self.cur],
            self.last_combo,
            first_trick_must_contain_three_spades=self.rules.first_trick_must_contain_three_spades,
            is_first_trick=self.first_trick and (self.cur==self.lead)
        )

    def step(self, action:int) -> StepResult:
        assert self.winner is None, "Episode already done"

        # decode action to combo; if illegal (mask False), override to PASS
        mask = self._mask_for_current()
        if action < 0 or action >= len(mask) or not mask[action]:
            action = 0  # PASS

        combo = decode_action_to_combo(action, self.hands[self.cur], self.rules)
        reward = 0.0

        # apply action
        if action == PASS:
            self.passed[self.cur] = True
        else:
            # play cards
            for c in combo.cards:
                self.hands[self.cur].remove(c)
                self.seen.add(c)
            self.last_combo = combo
            self.lead = self.cur
            self.passed = [False]*4

            # check win
            if len(self.hands[self.cur]) == 0:
                self.winner = self.cur

        # advance turn
        done = False
        if self.winner is not None:
            done = True
            # terminal rewards: +1 for winner, -1 for others (symmetric zero-sum if we sum to 0)
            reward = +1.0
        else:
            # If everyone else passed and we return to the lead player => new trick
            nxt = (self.cur + 1) % 4
            # find next player who hasn't finished; in normal rules nobody finishes mid-trick except by playing out
            # Here just proceed cyclically
            self.cur = nxt

            # if the next player is the previous lead and all other 3 have passed, reset trick
            if self.cur == self.lead and all(self.passed[i] or i==self.lead for i in range(4)):
                self.last_combo = None
                self.passed = [False]*4
                self.first_trick = False

        # reward shaping (potential-based): Phi = -cards_left + 0.1 * (#pairs + #straight_spans)
        shaped = 0.0
        if self.reward_shaping and not done:
            new_pot = self._potential(self.lead)  # potential anchored to current leader
            shaped = 0.99 * new_pot - self.prev_potential
            self.prev_potential = new_pot
            reward += shaped

        obs = self._obs_for_current()
        mask = self._mask_for_current()
        info = {"winner": self.winner, "shaping": shaped}
        return StepResult(obs=obs, mask=mask, reward=reward, done=done, info=info)

    def _potential(self, seat:int) -> float:
        # potential tied to the active player's hand
        h = self.hands[self.cur]
        # pairs count (rough)
        rank_counts = {}
        for c in h:
            r,_ = decode_card(c)
            rank_counts[r] = rank_counts.get(r,0)+1
        pairs = sum(1 for r,c in rank_counts.items() if c>=2)
        # straight spans (rough proxy): count consecutive rank spans present
        ranks_present = sorted(list(rank_counts.keys()))
        spans = 0
        i=0
        while i < len(ranks_present):
            j=i
            while j+1<len(ranks_present) and ranks_present[j+1]==ranks_present[j]+1: j+=1
            span_len = j-i+1
            spans += max(0, span_len-2)  # only reward spans length >=3
            i=j+1
        return -len(h) + 0.1*(pairs + spans)

    def clone_for_eval(self) -> "TLMNEnv":
        return TLMNEnv(rules=self.rules, seed=self.rng.randint(0, 1<<30), reward_shaping=False)
