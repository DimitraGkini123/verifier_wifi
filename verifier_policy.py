# verifier_policy.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List


class Label(str, Enum):
    LIGHT = "LIGHT"
    MEDIUM = "MEDIUM"
    HEAVY = "HEAVY"
    SUSPICIOUS = "SUSPICIOUS"


class AttestKind(str, Enum):
    NONE = "NONE"
    PARTIAL = "PARTIAL"
    FULL = "FULL"


@dataclass
class PolicyDecision:
    do_get_windows: bool = True
    get_windows_max: int = 6
    attest_kind: AttestKind = AttestKind.NONE
    k: int = 0
    reason: str = ""


@dataclass
class InferenceSummary:
    majority: Label
    confidence: float
    n: int


@dataclass
class DevicePolicyState:
    stable_label: Label = Label.MEDIUM
    last_majority: Label = Label.MEDIUM
    last_confidence: float = 0.0
    last_reason: str = "init"

    # hysteresis
    pending_label: Optional[Label] = None
    pending_count: int = 0

    # timers
    next_get_windows_ts: float = 0.0
    next_attest_ts: float = 0.0

    # optional cooldown against spamming attest
    attest_cooldown_s: float = 0.0


@dataclass
class PolicyConfig:
    # GET_WINDOWS behavior by stable label
    get_windows_period_s: Dict[Label, float] = field(default_factory=lambda: {
        Label.LIGHT: 0.5,       # συχνά
        Label.MEDIUM: 1.0,
        Label.HEAVY: 2.0,       # πιο αραιά
        Label.SUSPICIOUS: 0.5,  # συχνά για evidence
    })
    get_windows_max: Dict[Label, int] = field(default_factory=lambda: {
        Label.LIGHT: 12,
        Label.MEDIUM: 8,
        Label.HEAVY: 6,
        Label.SUSPICIOUS: 12,
    })

    # ATTEST behavior by stable label
    attest_period_s: Dict[Label, float] = field(default_factory=lambda: {
        Label.LIGHT: 10.0,         # partial συχνά (πχ κάθε 10s)
        Label.MEDIUM: 20.0,
        Label.HEAVY: 30.0,         # πιο αραιό attest όταν είναι heavy (για να μην βαραίνει)
        Label.SUSPICIOUS: 5.0,     # συχνά
    })
    attest_kind: Dict[Label, AttestKind] = field(default_factory=lambda: {
        Label.LIGHT: AttestKind.PARTIAL,
        Label.MEDIUM: AttestKind.PARTIAL,
        Label.HEAVY: AttestKind.PARTIAL,      # keep partial even on heavy (your earlier preference)
        Label.SUSPICIOUS: AttestKind.FULL,    # escalate
    })
    partial_k: Dict[Label, int] = field(default_factory=lambda: {
        Label.LIGHT: 8,       # μεγάλο k
        Label.MEDIUM: 4,
        Label.HEAVY: 2,       # μικρό k για “lightweight” attest
        Label.SUSPICIOUS: 0,
    })

    # safety: minimum time between attest triggers
    min_attest_cooldown_s: float = 1.0


class PolicyEngine:
    """
    Holds per-device state:
      - on_inference_batch(dev, labels[]) updates stable_label with hysteresis
      - tick(dev, now) returns what to do now (GET_WINDOWS? ATTEST?)
    """

    def __init__(self, hysteresis_n: int = 2, enable_get_windows: bool = True, config: Optional[PolicyConfig] = None):
        self.hysteresis_n = max(1, int(hysteresis_n))
        self.enable_get_windows = bool(enable_get_windows)
        self.cfg = config or PolicyConfig()
        self.devices: Dict[str, DevicePolicyState] = {}

    def _st(self, dev: str) -> DevicePolicyState:
        st = self.devices.get(dev)
        if st is None:
            st = DevicePolicyState()
            self.devices[dev] = st
        return st

    @staticmethod
    def _majority(labels: List[Label]) -> InferenceSummary:
        if not labels:
            return InferenceSummary(majority=Label.MEDIUM, confidence=0.0, n=0)
        counts: Dict[Label, int] = {}
        for lb in labels:
            counts[lb] = counts.get(lb, 0) + 1
        maj = max(counts.items(), key=lambda kv: kv[1])[0]
        conf = counts[maj] / float(len(labels))
        return InferenceSummary(majority=maj, confidence=conf, n=len(labels))

    def on_inference_batch(self, dev: str, labels: List[Label], now: float) -> InferenceSummary:
        st = self._st(dev)
        summ = self._majority(labels)

        st.last_majority = summ.majority
        st.last_confidence = summ.confidence

        # hysteresis: change stable only after N consecutive majorities
        if summ.n == 0:
            st.last_reason = "no_valid_labels"
            return summ

        if summ.majority == st.stable_label:
            st.pending_label = None
            st.pending_count = 0
            st.last_reason = "majority_matches_stable"
            return summ

        # majority differs from stable
        if st.pending_label != summ.majority:
            st.pending_label = summ.majority
            st.pending_count = 1
            st.last_reason = f"pending_start:{summ.majority.value}"
        else:
            st.pending_count += 1
            st.last_reason = f"pending_count:{st.pending_count}/{self.hysteresis_n}"

        if st.pending_count >= self.hysteresis_n:
            old = st.stable_label
            st.stable_label = st.pending_label or summ.majority
            st.pending_label = None
            st.pending_count = 0
            st.last_reason = f"stable_changed:{old.value}->{st.stable_label.value}"

            # when label changes, allow immediate scheduling (reset timers)
            st.next_get_windows_ts = 0.0
            st.next_attest_ts = 0.0

        return summ

    def tick(self, dev: str, now: float) -> PolicyDecision:
        st = self._st(dev)
        lb = st.stable_label

        # schedule GET_WINDOWS
        do_get = False
        gw_max = self.cfg.get_windows_max[lb]
        if self.enable_get_windows:
            if now >= st.next_get_windows_ts:
                do_get = True
                st.next_get_windows_ts = now + float(self.cfg.get_windows_period_s[lb])

        # schedule ATTEST
        kind = AttestKind.NONE
        k = 0
        if now >= st.next_attest_ts and now >= st.attest_cooldown_s:
            kind = self.cfg.attest_kind[lb]
            if kind == AttestKind.PARTIAL:
                k = int(self.cfg.partial_k[lb])
            st.next_attest_ts = now + float(self.cfg.attest_period_s[lb])
            st.attest_cooldown_s = now + float(self.cfg.min_attest_cooldown_s)

        return PolicyDecision(
            do_get_windows=do_get,
            get_windows_max=gw_max,
            attest_kind=kind,
            k=k,
            reason=st.last_reason
        )
