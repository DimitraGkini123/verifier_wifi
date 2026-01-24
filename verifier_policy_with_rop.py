# verifier_policy_with_rop.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List


class Label(str, Enum):
    # model-driven labels
    LIGHT_SAFE = "light_safe"
    MEDIUM_SAFE = "medium_safe"
    HEAVY_SAFE = "heavy_safe"
    LIGHT_ROP = "light_rop"
    HEAVY_ROP = "heavy_rop"

    # internal policy label for uncertainty / not enough confidence
    SUSPICIOUS = "suspicious"


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
    confidence: float         # vote confidence (fraction or weighted fraction)
    n: int

    # extras (optional, for logging/decision)
    weighted_majority: Optional[Label] = None
    weighted_confidence: Optional[float] = None
    model_conf_avg: Optional[float] = None
    model_conf_min: Optional[float] = None
    model_conf_max: Optional[float] = None
    rop_score_avg: Optional[float] = None   # e.g. avg P(rop) across windows


@dataclass
class DevicePolicyState:
    stable_label: Label = Label.MEDIUM_SAFE

    last_majority: Label = Label.MEDIUM_SAFE
    last_confidence: float = 0.0
    last_reason: str = "init"

    # keep last model/proba summaries for logs
    last_model_conf_avg: Optional[float] = None
    last_model_conf_min: Optional[float] = None
    last_model_conf_max: Optional[float] = None
    last_weighted_majority: Optional[Label] = None
    last_weighted_conf: Optional[float] = None
    last_rop_score_avg: Optional[float] = None

    # hysteresis
    pending_label: Optional[Label] = None
    pending_count: int = 0

    # timers
    next_get_windows_ts: float = 0.0
    next_attest_ts: float = 0.0

    # cooldown
    attest_cooldown_ts: float = 0.0


@dataclass
class PolicyConfig:
    # GET_WINDOWS behavior by stable label
    get_windows_period_s: Dict[Label, float] = field(default_factory=lambda: {
        Label.LIGHT_SAFE: 0.5,
        Label.MEDIUM_SAFE: 1.0,
        Label.HEAVY_SAFE: 2.0,

        # for ROP: gather lots of evidence frequently
        Label.LIGHT_ROP: 0.3,
        Label.HEAVY_ROP: 0.3,

        # uncertain: sample frequently to resolve
        Label.SUSPICIOUS: 0.4,
    })
    get_windows_max: Dict[Label, int] = field(default_factory=lambda: {
        Label.LIGHT_SAFE: 12,
        Label.MEDIUM_SAFE: 8,
        Label.HEAVY_SAFE: 6,
        Label.LIGHT_ROP: 12,
        Label.HEAVY_ROP: 12,
        Label.SUSPICIOUS: 12,
    })

    # ATTEST behavior by stable label
    # NOTE: for ROP we choose NONE (hash won't detect ROP)
    attest_period_s: Dict[Label, float] = field(default_factory=lambda: {
        Label.LIGHT_SAFE: 10.0,
        Label.MEDIUM_SAFE: 20.0,
        Label.HEAVY_SAFE: 30.0,
        Label.LIGHT_ROP: 999999.0,
        Label.HEAVY_ROP: 999999.0,
        Label.SUSPICIOUS: 15.0,  # optional: sometimes do FULL/partial when unsure
    })
    attest_kind: Dict[Label, AttestKind] = field(default_factory=lambda: {
        Label.LIGHT_SAFE: AttestKind.PARTIAL,
        Label.MEDIUM_SAFE: AttestKind.PARTIAL,
        Label.HEAVY_SAFE: AttestKind.PARTIAL,
        Label.LIGHT_ROP: AttestKind.NONE,
        Label.HEAVY_ROP: AttestKind.NONE,

        # if you're uncertain, you *can* do FULL occasionally to catch non-ROP tampering
        Label.SUSPICIOUS: AttestKind.FULL,
    })
    partial_k: Dict[Label, int] = field(default_factory=lambda: {
        Label.LIGHT_SAFE: 8,
        Label.MEDIUM_SAFE: 4,
        Label.HEAVY_SAFE: 2,
        Label.LIGHT_ROP: 0,
        Label.HEAVY_ROP: 0,
        Label.SUSPICIOUS: 0,
    })

    # safety: minimum time between attest triggers
    min_attest_cooldown_s: float = 1.0

    # confidence gating for "accepting" ROP
    rop_accept_vote_conf: float = 0.60        # weighted_confidence
    rop_accept_model_conf_avg: float = 0.60   # avg max-proba
    rop_accept_rop_score_avg: float = 0.60    # avg P(rop)


class PolicyEngine:
    """
    Per-device:
      - on_inference_batch(...) updates stable_label with hysteresis + confidence gating
      - tick(...) returns GET_WINDOWS + ATTEST scheduling based on stable_label
    """

    def __init__(
        self,
        hysteresis_n: int = 2,
        enable_get_windows: bool = True,
        config: Optional[PolicyConfig] = None
    ):
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
            return InferenceSummary(majority=Label.MEDIUM_SAFE, confidence=0.0, n=0)
        counts: Dict[Label, int] = {}
        for lb in labels:
            counts[lb] = counts.get(lb, 0) + 1
        maj = max(counts.items(), key=lambda kv: kv[1])[0]
        conf = counts[maj] / float(len(labels))
        return InferenceSummary(majority=maj, confidence=conf, n=len(labels))

    def on_inference_batch(
        self,
        dev: str,
        labels: List[Label],
        now: float,
        *,
        weighted_majority: Optional[Label] = None,
        weighted_confidence: Optional[float] = None,
        model_conf_avg: Optional[float] = None,
        model_conf_min: Optional[float] = None,
        model_conf_max: Optional[float] = None,
        rop_score_avg: Optional[float] = None
    ) -> InferenceSummary:
        st = self._st(dev)
        summ = self._majority(labels)

        st.last_majority = summ.majority
        st.last_confidence = summ.confidence
        st.last_weighted_majority = weighted_majority
        st.last_weighted_conf = weighted_confidence
        st.last_model_conf_avg = model_conf_avg
        st.last_model_conf_min = model_conf_min
        st.last_model_conf_max = model_conf_max
        st.last_rop_score_avg = rop_score_avg

        # Also return in summary object (for logging)
        summ.weighted_majority = weighted_majority
        summ.weighted_confidence = weighted_confidence
        summ.model_conf_avg = model_conf_avg
        summ.model_conf_min = model_conf_min
        summ.model_conf_max = model_conf_max
        summ.rop_score_avg = rop_score_avg

        if summ.n == 0:
            st.last_reason = "no_valid_labels"
            return summ

        # Decide the "candidate" label we want to feed into hysteresis:
        # Prefer weighted_majority if provided, else plain majority.
        candidate = weighted_majority or summ.majority

        # Confidence gating for ROP: if candidate is ROP but confidence is weak -> treat as SUSPICIOUS
        if candidate in (Label.LIGHT_ROP, Label.HEAVY_ROP):
            wconf = float(weighted_confidence) if weighted_confidence is not None else summ.confidence
            mcavg = float(model_conf_avg) if model_conf_avg is not None else 0.0
            rsc = float(rop_score_avg) if rop_score_avg is not None else 0.0

            if not (wconf >= self.cfg.rop_accept_vote_conf and mcavg >= self.cfg.rop_accept_model_conf_avg and rsc >= self.cfg.rop_accept_rop_score_avg):
                candidate = Label.SUSPICIOUS
                st.last_reason = "rop_low_confidence->suspicious"

        # Standard hysteresis on candidate vs stable
        if candidate == st.stable_label:
            st.pending_label = None
            st.pending_count = 0
            if st.last_reason == "init":
                st.last_reason = "majority_matches_stable"
            elif "rop_low_confidence" not in st.last_reason:
                st.last_reason = "majority_matches_stable"
            return summ

        if st.pending_label != candidate:
            st.pending_label = candidate
            st.pending_count = 1
            st.last_reason = f"pending_start:{candidate.value}"
        else:
            st.pending_count += 1
            st.last_reason = f"pending_count:{st.pending_count}/{self.hysteresis_n}"

        if st.pending_count >= self.hysteresis_n:
            old = st.stable_label
            st.stable_label = st.pending_label or candidate
            st.pending_label = None
            st.pending_count = 0
            st.last_reason = f"stable_changed:{old.value}->{st.stable_label.value}"

            # reset timers so changes take effect quickly
            st.next_get_windows_ts = 0.0
            st.next_attest_ts = 0.0

        return summ

    def tick(self, dev: str, now: float) -> PolicyDecision:
        st = self._st(dev)
        lb = st.stable_label

        # GET_WINDOWS schedule
        do_get = False
        gw_max = int(self.cfg.get_windows_max[lb])
        if self.enable_get_windows and now >= st.next_get_windows_ts:
            do_get = True
            st.next_get_windows_ts = now + float(self.cfg.get_windows_period_s[lb])

        # ATTEST schedule
        kind = AttestKind.NONE
        k = 0
        if now >= st.next_attest_ts and now >= st.attest_cooldown_ts:
            kind = self.cfg.attest_kind[lb]
            if kind == AttestKind.PARTIAL:
                k = int(self.cfg.partial_k[lb])

            st.next_attest_ts = now + float(self.cfg.attest_period_s[lb])
            st.attest_cooldown_ts = now + float(self.cfg.min_attest_cooldown_s)

        return PolicyDecision(
            do_get_windows=do_get,
            get_windows_max=gw_max,
            attest_kind=kind,
            k=k,
            reason=st.last_reason
        )
