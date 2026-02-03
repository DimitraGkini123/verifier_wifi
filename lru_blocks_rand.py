# lru_blocks.py
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional
import random


@dataclass
class DeviceLRUBlocks:
    n_blocks: int
    # Order: LRU -> MRU (least recent at beginning)
    _od: OrderedDict[int, None]

    # --- NEW: RNG + sampling knobs (optional but useful) ---
    _rng: random.Random = random.Random()

    @classmethod
    def fresh(cls, n_blocks: int, seed: Optional[int] = None) -> "DeviceLRUBlocks":
        rng = random.Random(seed)
        return cls(
            n_blocks=n_blocks,
            _od=OrderedDict((i, None) for i in range(n_blocks)),
            _rng=rng,
        )

    def ensure_size(self, n_blocks: int, seed: Optional[int] = None) -> None:
        """If block count changes (reprovision), reset to fresh."""
        if n_blocks != self.n_blocks:
            self.n_blocks = n_blocks
            self._od = OrderedDict((i, None) for i in range(n_blocks))
            # reset RNG too (optional)
            if seed is not None:
                self._rng = random.Random(seed)

    def pick(
        self,
        k: int,
        *,
        pool_frac: float = 0.25,
        pool_min: int = 32,
        shuffle_output: bool = True,
    ) -> List[int]:
        """
        LRU-biased random sampling.

        Instead of always taking the first k LRU blocks (deterministic),
        we form a candidate pool from the LRU side and sample k uniformly
        from that pool.

        Args:
          k: number of blocks to pick
          pool_frac: fraction of the address space to consider as LRU pool (0..1]
          pool_min: minimum pool size (useful when n_blocks is small)
          shuffle_output: return chosen indices in random order

        Returns:
          list[int] of chosen indices (unique, without replacement)
        """
        n = self.n_blocks
        k = max(1, min(int(k), n))

        # Candidate pool size
        pool_frac = float(pool_frac)
        if pool_frac <= 0.0:
            pool_frac = 0.25
        if pool_frac > 1.0:
            pool_frac = 1.0

        pool_size = max(int(pool_min), int(round(pool_frac * n)))
        pool_size = min(n, max(k, pool_size))  # must be >= k and <= n

        # LRU side pool = first pool_size keys
        keys = list(self._od.keys())
        pool = keys[:pool_size]

        # Sample k from the pool (uniform, no replacement)
        chosen = self._rng.sample(pool, k)

        # Optional: shuffle again (sample already random, but this ensures "order randomness" explicitly)
        if shuffle_output:
            self._rng.shuffle(chosen)

        return chosen

    def touch(self, indices: List[int]) -> None:
        """Move touched indices to MRU end."""
        for i in indices:
            if i in self._od:
                self._od.move_to_end(i, last=True)

    def export_state(self) -> Dict:
        """Serialize order for persistence."""
        return {"n_blocks": self.n_blocks, "order": list(self._od.keys())}

    @classmethod
    def from_state(cls, state: Dict, seed: Optional[int] = None) -> "DeviceLRUBlocks":
        n = int(state.get("n_blocks", 0) or 0)
        order = state.get("order", [])

        # Basic validation
        if n <= 0 or not isinstance(order, list) or len(order) != n:
            return cls.fresh(max(n, 1) if n > 0 else 1, seed=seed)

        # Coerce to ints
        try:
            order_int = [int(x) for x in order]
        except Exception:
            return cls.fresh(n, seed=seed)

        # Must be a permutation of 0..n-1
        if set(order_int) != set(range(n)):
            return cls.fresh(n, seed=seed)

        # Build OrderedDict in that exact order (LRU -> MRU)
        od = OrderedDict((i, None) for i in order_int)
        rng = random.Random(seed)
        return cls(n_blocks=n, _od=od, _rng=rng)
