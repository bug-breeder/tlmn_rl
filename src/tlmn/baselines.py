from __future__ import annotations
from typing import Tuple
import numpy as np
from .encoding import PASS

class GreedyAgent:
    """Greedy: play the **weakest legal** action; on lead, prefer straights > pair_runs > pairs > triples > singles; keep 2s if possible."""
    def __init__(self):
        pass

    def act(self, obs, mask) -> int:
        legal = np.where(mask)[0]
        if len(legal)==0: return PASS
        # Heuristic: prefer structured combos when on lead (we can't see "on lead" bit directly here;
        # but caller can pass richer obs; for simplicity we just prefer non-single if many cards left)
        # Basic ranking of action ids by type buckets:
        def bucket(a:int) -> int:
            if a==0: return 10
            if 92<=a<=146: return 1  # straights
            if 147<=a<=212: return 2  # pair-runs
            if 53<=a<=65: return 3    # pairs
            if 66<=a<=78: return 4    # triples
            if 79<=a<=91: return 5    # quads (save them usually; but greedy may play if forced)
            if 1<=a<=52: return 6     # singles
            return 9
        legal_sorted = sorted(legal, key=lambda a: (bucket(a), a))
        return legal_sorted[0]
