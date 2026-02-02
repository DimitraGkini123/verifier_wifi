# lru_blocks.py
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DeviceLRUBlocks:
    n_blocks: int
    # Order: LRU -> MRU (least recent at beginning)
    _od: OrderedDict[int, None]

    @classmethod
    def fresh(cls, n_blocks: int) -> "DeviceLRUBlocks":
        return cls(n_blocks=n_blocks, _od=OrderedDict((i, None) for i in range(n_blocks)))

    def ensure_size(self, n_blocks: int) -> None:
        """If block count changes (reprovision), reset to fresh."""
        if n_blocks != self.n_blocks:
            self.n_blocks = n_blocks
            self._od = OrderedDict((i, None) for i in range(n_blocks))

    def pick(self, k: int) -> List[int]:
        k = max(1, min(int(k), self.n_blocks))
        # First k keys are least-recently-used
        return list(self._od.keys())[:k]

    def touch(self, indices: List[int]) -> None:
        """Move touched indices to MRU end."""
        for i in indices:
            if i in self._od:
                self._od.move_to_end(i, last=True)

    def export_state(self) -> Dict:
        """Serialize order for persistence."""
        return {"n_blocks": self.n_blocks, "order": list(self._od.keys())}

    @classmethod
    def from_state(cls, state: Dict) -> "DeviceLRUBlocks":
        n = int(state.get("n_blocks", 0) or 0)
        order = state.get("order", [])

        # Basic validation
        if n <= 0 or not isinstance(order, list) or len(order) != n:
            return cls.fresh(max(n, 1) if n > 0 else 1)

        # Coerce to ints
        try:
            order_int = [int(x) for x in order]
        except Exception:
            return cls.fresh(n)

        # Must be a permutation of 0..n-1
        if set(order_int) != set(range(n)):
            return cls.fresh(n)

        # Build OrderedDict in that exact order (LRU -> MRU)
        od = OrderedDict((i, None) for i in order_int)
        return cls(n_blocks=n, _od=od)
