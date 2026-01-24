# verifier_policy_server.py

import asyncio
import json
import secrets
import time
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from utils import save_json_atomic, jdump, sha256, ts_ms, now_s, unhex
from lru_blocks import DeviceLRUBlocks
from policy_device_lr_rop import DeviceLRPolicy

# NEW: policy engine that schedules GET_WINDOWS + ATTEST based on stable label
from verifier_policy_with_rop import PolicyEngine, Label as PLLabel, AttestKind as PLAttestKind

LRU_STATE_PATH = "lru_state.json"

HOST = "0.0.0.0"
PORT = 4242
GOLDEN_PATH = "golden.json"

TRUST_UNKNOWN = "UNKNOWN"
TRUST_TRUSTED = "TRUSTED"
TRUST_UNTRUSTED = "UNTRUSTED"

# ---------------- ML / Policy config ----------------
MODEL_PATH = "models/device_1_safe_rop.joblib"
ML_ENABLE = True

# Policy hysteresis: πόσες φορές πρέπει να δεις majority διαφορετικό για να αλλάξεις stable label
POLICY_HYSTERESIS_N = 2

# Safety jitter (μικρό) για να μη συγχρονίζονται πολλά devices (αν έχεις πολλά)
LOOP_TICK_S = 0.20
JITTER_S = 0.05

# Initial full attestation on connect?
DO_INITIAL_FULL_ATTEST = True


@dataclass
class PendingReq:
    fut: asyncio.Future
    sent_msg: dict


@dataclass
class DeviceConn:
    device_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    pending: Dict[str, PendingReq] = field(default_factory=dict)
    last_seen_ts: float = field(default_factory=lambda: time.time())

    trust_state: str = TRUST_UNKNOWN
    attest_fail_streak: int = 0
    last_attest_ok_ts: float = 0.0
    last_attest_fail_ts: float = 0.0

    def is_alive(self) -> bool:
        return not self.writer.is_closing()


class VerifierPolicyServer:
    def __init__(self, golden_db: dict): #φορτώνει golden hashes 
        # Golden DB
        self.golden = golden_db

        # Devices
        self.devices: Dict[str, DeviceConn] = {}
        self.selected_device: Optional[str] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Per-device window cursor ("since")
        self.last_seen: Dict[str, int] = {}

        # Files
        self.windows_fp: Dict[str, Any] = {}
        self.events_fp: Dict[str, Any] = {}
        self.attest_fp: Dict[str, Any] = {}

        # Attestation locks (per device)
        self.attest_locks: Dict[str, asyncio.Lock] = {}

        # LRU blocks for partial
        self.block_lru: Dict[str, DeviceLRUBlocks] = {}
        self._load_lru_state()

        # ML model
        self.lr_policy = DeviceLRPolicy(MODEL_PATH) if ML_ENABLE else None

        # NEW: Policy engine per server (keeps per-device state inside)
        self.policy = PolicyEngine(hysteresis_n=POLICY_HYSTERESIS_N, enable_get_windows=True)

        # Per-device policy task
        self.policy_tasks: Dict[str, asyncio.Task] = {}

    # ----------------- file helpers -----------------
    def _open_files_for(self, dev: str):
        if dev in self.windows_fp:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        wpath = f"windows_{dev}_{stamp}.jsonl"
        epath = f"events_{dev}_{stamp}.jsonl"
        apath = f"attest_{dev}_{stamp}.jsonl"
        self.windows_fp[dev] = open(wpath, "a", encoding="utf-8", buffering=1)
        self.events_fp[dev] = open(epath, "a", encoding="utf-8", buffering=1)
        self.attest_fp[dev] = open(apath, "a", encoding="utf-8", buffering=1)
        print(f"[{now_s()}] files for {dev}: windows={wpath} events={epath} attest={apath}")

    @staticmethod
    def _jwrite(fp, obj: dict):
        fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        fp.flush()

    def _close_files_for(self, dev: str):
        for dct in (self.windows_fp, self.events_fp, self.attest_fp):
            fp = dct.pop(dev, None)
            try:
                if fp:
                    fp.close()
            except Exception:
                pass

    # ----------------- LRU persistence -----------------
    def _load_lru_state(self):
        try:
            with open(LRU_STATE_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for dev, st in raw.items():
                    if isinstance(st, dict):
                        self.block_lru[dev] = DeviceLRUBlocks.from_state(st)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _save_lru_state(self):
        try:
            blob = {dev: lru.export_state() for dev, lru in self.block_lru.items()}
            save_json_atomic(LRU_STATE_PATH, blob)
        except Exception:
            pass

    def _get_block_lru(self, dev: str) -> Optional[DeviceLRUBlocks]:
        bc = self.get_block_count(dev)
        if bc <= 0:
            return None
        lru = self.block_lru.get(dev)
        if lru is None:
            lru = DeviceLRUBlocks.fresh(bc)
            self.block_lru[dev] = lru
            self._save_lru_state()
        else:
            lru.ensure_size(bc)
        return lru

    # ----------------- locks -----------------
    def _attest_lock(self, dev: str) -> asyncio.Lock:  #επιστρέφει per device lock ώστε να μην τρέχουν 2 attest ταυτόχρονα στο ίδιο device.
        lk = self.attest_locks.get(dev)
        if lk is None:
            lk = asyncio.Lock()
            self.attest_locks[dev] = lk
        return lk

    # ----------------- golden access -----------------
    def golden_full_hash(self, device_id: str, region: str = "fw") -> Optional[bytes]:
        try:
            return unhex(self.golden[device_id][region]["sha256"])
        except Exception:
            return None

    def has_golden_blocks(self, device_id: str) -> bool:
        try:
            _ = self.golden[device_id]["blocks"]["hashes"]
            return True
        except Exception:
            return False

    def get_block_count(self, device_id: str) -> int:
        try:
            return int(self.golden[device_id]["blocks"]["block_count"])
        except Exception:
            return 0

    def golden_block_hash(self, device_id: str, index: int) -> Optional[bytes]:
        try:
            return unhex(self.golden[device_id]["blocks"]["hashes"][index])
        except Exception:
            return None

    # ----------------- logging attest -----------------
    def log_attest_event(
        self,
        dev: str,
        kind: str,
        k: Optional[int],
        indices: Optional[list[int]],
        resp: dict,
        trust_before: str,
        trust_after: str,
        trigger: str = "POLICY",
        ml: Optional[dict] = None,
    ):
        fp = self.attest_fp.get(dev)
        if not fp:
            return
        self._jwrite(fp, {
            "ts_ms": ts_ms(),
            "device": dev,
            "event": "attest",
            "attest_kind": kind,
            "trigger": trigger,
            "ml": ml,
            "k": k,
            "indices": indices,
            "trust_before": trust_before,
            "trust_after": trust_after,
            "verify_ok": resp.get("verify_ok"),
            "verify_reason": resp.get("verify_reason", resp.get("reason")),
            "rtt_ms": resp.get("_rtt_ms"),
            "req_bytes": resp.get("_req_bytes"),
            "resp_bytes": resp.get("_resp_bytes"),
        })

    # ----------------- RX loop -----------------
    async def rx_loop(self, dc: DeviceConn):
        while True:
            line = await dc.reader.readline()
            if not line:
                return
            dc.last_seen_ts = time.time()
            try:
                msg = json.loads(line.decode("utf-8"))
            except Exception:
                print(f"[{now_s()}] [RX raw] {dc.device_id}: {line!r}")
                continue

            req_id = msg.get("req_id")
            if req_id and req_id in dc.pending:
                pending = dc.pending.pop(req_id)
                verified_msg = self.verify_if_needed(dc.device_id, pending.sent_msg, msg)
                if not pending.fut.done():
                    pending.fut.set_result(verified_msg)
            else:
                print(f"[{now_s()}] [RX] {dc.device_id}: {msg}")

    # ----------------- request send -----------------
    async def send_request(self, device_id: str, msg: dict, timeout: float = 5.0) -> dict:
        dc = self.devices.get(device_id)
        if not dc or not dc.is_alive():  # αν δεν ειναι συνδεδεμένη η συσκευή
            return {"type": "ERROR", "reason": "device_not_connected"}

        req_id = secrets.token_hex(8)
        msg = dict(msg)
        msg["req_id"] = req_id

        fut = self.loop.create_future()
        dc.pending[req_id] = PendingReq(fut=fut, sent_msg=msg)

        dc.writer.write(jdump(msg))
        await dc.writer.drain()

        try:
            resp = await asyncio.wait_for(fut, timeout=timeout)
            return resp
        except asyncio.TimeoutError:
            dc.pending.pop(req_id, None)
            return {"type": "ERROR", "reason": "timeout_waiting_response", "req_id": req_id}

    async def send_request_timed(self, device_id: str, msg: dict, timeout: float = 5.0) -> dict:
        req_line = jdump({**msg, "req_id": "0000000000000000"})
        req_bytes = len(req_line)
        t0 = time.perf_counter()
        resp = await self.send_request(device_id, msg, timeout=timeout)
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        resp_bytes = len(jdump(resp)) if isinstance(resp, dict) else 0
        if isinstance(resp, dict):
            resp["_rtt_ms"] = round(rtt_ms, 2)
            resp["_req_bytes"] = int(req_bytes)
            resp["_resp_bytes"] = int(resp_bytes)
        return resp

    # ----------------- verification -----------------
    def verify_if_needed(self, device_id: str, sent: dict, received: dict) -> dict:
        mode = sent.get("mode")
        rtype = received.get("type")

        if rtype not in ("ATTEST_RESPONSE", "PONG", "WINDOWS"):
            return received

        # FULL
        if mode == "FULL_HASH_PROVER" and rtype == "ATTEST_RESPONSE":
            golden = self.golden_full_hash(device_id, region=sent.get("region", "fw"))
            if golden is None:
                received["verify_ok"] = False
                received["verify_reason"] = "missing_golden_full_hash"
                return received

            nonce_hex = sent.get("nonce")
            if "response_hex" in received and nonce_hex:
                nonce = unhex(nonce_hex)
                expected = sha256(nonce + golden)
                got = unhex(received["response_hex"])
                ok = (got == expected)
                received["verify_ok"] = ok
                received["verify_reason"] = "nonce_bound_match" if ok else "nonce_bound_mismatch"
                return received

            received["verify_ok"] = False
            received["verify_reason"] = "missing_hash_fields"
            return received

        # PARTIAL
        if mode == "PARTIAL_BLOCKS" and rtype == "ATTEST_RESPONSE":
            if not self.has_golden_blocks(device_id):
                received["verify_ok"] = False
                received["verify_reason"] = "missing_golden_blocks"
                return received

            nonce_hex = sent.get("nonce")
            nonce = unhex(nonce_hex) if nonce_hex else None

            blocks = received.get("blocks", []) or []
            all_ok = True
            reasons = []

            for b in blocks:
                idx = b.get("index")
                if idx is None:
                    all_ok = False
                    reasons.append("block_missing_index")
                    continue

                golden_b = self.golden_block_hash(device_id, int(idx))
                if golden_b is None:
                    all_ok = False
                    reasons.append(f"missing_golden_block_{idx}")
                    continue

                ok = False
                if nonce is not None and "response_hex" in b:
                    expected = sha256(nonce + golden_b)
                    got = unhex(b["response_hex"])
                    ok = (got == expected)
                elif "hash_hex" in b:
                    ok = (unhex(b["hash_hex"]) == golden_b)

                if not ok:
                    all_ok = False
                    reasons.append(f"block_{idx}_mismatch")

            received["verify_ok"] = all_ok
            received["verify_reason"] = "ok" if all_ok else ",".join(reasons)
            return received

        return received

    # ----------------- trust update -----------------
    def _update_trust_from_attest(self, dev: str, resp: dict, attempt: int):
        dc = self.devices.get(dev)
        if not dc:
            return

        reason = resp.get("verify_reason", resp.get("reason", "unknown"))

        if reason in ("missing_golden_full_hash", "missing_golden_blocks", "no_golden_blocks"):
            dc.trust_state = TRUST_UNKNOWN
            dc.attest_fail_streak = 0
            return

        ok = bool(resp.get("verify_ok", False))
        if ok:
            dc.trust_state = TRUST_TRUSTED
            dc.attest_fail_streak = 0
            dc.last_attest_ok_ts = time.time()
        else:
            dc.attest_fail_streak += 1
            dc.last_attest_fail_ts = time.time()
            if dc.attest_fail_streak >= 2:
                dc.trust_state = TRUST_UNTRUSTED

    # ----------------- attest actions -----------------
    async def attest_full_once(self, dev: str, timeout: float = 8.0) -> dict:
        nonce = secrets.token_hex(8)
        resp = await self.send_request_timed(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "FULL_HASH_PROVER",
            "region": "fw",
            "nonce": nonce
        }, timeout=timeout)
        return resp

    async def attest_full_with_retry(self, dev: str) -> dict:
        resp1 = await self.attest_full_once(dev, timeout=8.0)
        self._update_trust_from_attest(dev, resp1 if isinstance(resp1, dict) else {}, attempt=1)
        if resp1.get("verify_ok", False):
            return resp1
        await asyncio.sleep(1.0)
        resp2 = await self.attest_full_once(dev, timeout=8.0)
        self._update_trust_from_attest(dev, resp2 if isinstance(resp2, dict) else {}, attempt=2)
        return resp2

    async def attest_full_and_log(self, dev: str, trigger: str = "POLICY", ml: Optional[dict] = None) -> dict:
        async with self._attest_lock(dev):
            dc = self.devices.get(dev)
            if not dc:
                return {"type": "ERROR", "reason": "device_not_connected"}

            trust_before = dc.trust_state
            resp = await self.attest_full_with_retry(dev)
            trust_after = dc.trust_state if dev in self.devices else TRUST_UNKNOWN

            self.log_attest_event(
                dev=dev, kind="FULL", k=None, indices=None,
                resp=resp if isinstance(resp, dict) else {"type": "ERROR", "reason": "bad_resp"},
                trust_before=trust_before, trust_after=trust_after,
                trigger=trigger, ml=ml
            )
            return resp

    async def attest_partial_once(self, dev: str, k: int, timeout: float = 12.0) -> dict:
        bc = self.get_block_count(dev)
        if bc <= 0:
            return {"type": "ERROR", "reason": "no_golden_blocks"}

        k = max(1, min(int(k), bc))
        lru = self._get_block_lru(dev)
        if lru is None:
            return {"type": "ERROR", "reason": "no_golden_blocks"}

        indices = sorted(lru.pick(k))
        nonce = secrets.token_hex(8)

        resp = await self.send_request_timed(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "PARTIAL_BLOCKS",
            "region": "fw",
            "nonce": nonce,
            "indices": indices
        }, timeout=timeout)

        if isinstance(resp, dict):
            resp["_k"] = k
            resp["_indices"] = indices
        return resp

    async def attest_partial_and_log(self, dev: str, k: int, trigger: str = "POLICY", ml: Optional[dict] = None) -> dict:
        async with self._attest_lock(dev):
            dc = self.devices.get(dev)
            if not dc:
                return {"type": "ERROR", "reason": "device_not_connected"}

            trust_before = dc.trust_state
            resp = await self.attest_partial_once(dev, k=k, timeout=12.0)
            self._update_trust_from_attest(dev, resp if isinstance(resp, dict) else {}, attempt=1)

            # touch LRU only if ok
            if isinstance(resp, dict) and resp.get("verify_ok", False):
                idxs = resp.get("_indices") or []
                lru = self._get_block_lru(dev)
                if lru is not None and idxs:
                    lru.touch(idxs)
                    self._save_lru_state()

            trust_after = dc.trust_state if dev in self.devices else TRUST_UNKNOWN
            indices = resp.get("_indices") if isinstance(resp, dict) else None
            kk = resp.get("_k") if isinstance(resp, dict) else k

            self.log_attest_event(
                dev=dev, kind="PARTIAL", k=kk, indices=indices,
                resp=resp if isinstance(resp, dict) else {"type": "ERROR", "reason": "bad_resp"},
                trust_before=trust_before, trust_after=trust_after,
                trigger=trigger, ml=ml
            )
            return resp

    # ----------------- policy loop (NEW) -----------------
    @staticmethod
    def _map_model_label_to_policy_label(x) -> PLLabel:
        # model returns strings: light_safe/medium_safe/heavy_safe/light_rop/heavy_rop
        s = str(x).strip()
        try:
            return {
                "light_safe": PLLabel.LIGHT_SAFE,
                "medium_safe": PLLabel.MEDIUM_SAFE,
                "heavy_safe": PLLabel.HEAVY_SAFE,
                "light_rop": PLLabel.LIGHT_ROP,
                "heavy_rop": PLLabel.HEAVY_ROP,
            }.get(s, PLLabel.SUSPICIOUS)
        except Exception:
            return PLLabel.SUSPICIOUS


    async def policy_loop(self, dev: str):
        self._open_files_for(dev)
        fp_evt = self.events_fp.get(dev)

        # initial full attest (optional)
        if DO_INITIAL_FULL_ATTEST:
            await asyncio.sleep(0.5)
            if dev in self.devices:
                if fp_evt:
                    self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "initial_full_attest_start"})
                await self.attest_full_and_log(dev, trigger="INITIAL")

        while True:
            if dev not in self.devices:
                return

            # tick policy
            decision = self.policy.tick(dev, now=time.time())

            # 1) GET_WINDOWS if due
            if decision.do_get_windows:
                # don't overlap GET_WINDOWS during attestation
                lk = self._attest_lock(dev)
                if lk.locked():
                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(), "device": dev,
                            "event": "policy_skip_get_windows_attest_inflight",
                            "reason": decision.reason
                        })
                else:
                    since = int(self.last_seen.get(dev, 0))
                    t0 = time.perf_counter()
                    resp = await self.send_request(dev, {
                        "type": "GET_WINDOWS",
                        "since": since,
                        "max": int(decision.get_windows_max),
                    }, timeout=8.0)
                    rtt_ms = int((time.perf_counter() - t0) * 1000)

                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "policy_get_windows_done",
                            "since": since,
                            "max": int(decision.get_windows_max),
                            "rtt_ms": rtt_ms,
                            "resp_type": resp.get("type"),
                            "reason": decision.reason
                        })

                    if resp.get("type") == "WINDOWS":
                        windows = resp.get("windows", []) or []

                        # cursor update
                        to_id = resp.get("to", None)
                        if to_id is None and windows:
                            try:
                                to_id = int(windows[-1].get("window_id"))
                            except Exception:
                                to_id = None
                        if to_id is not None:
                            try:
                                self.last_seen[dev] = int(to_id) + 1
                            except Exception:
                                pass

                        # write windows
                        fp_win = self.windows_fp.get(dev)
                        dc = self.devices.get(dev)
                        trust = dc.trust_state if dc else TRUST_UNKNOWN
                        tnow = time.time()
                        if fp_win:
                            for w in windows:
                                self._jwrite(fp_win, {
                                    "ts": tnow,
                                    "device_id_str": dev,
                                    "trust_state": trust,
                                    "trusted_for_decision": (trust == TRUST_TRUSTED),
                                    **w
                                })

                        # 2) ML inference on ALL windows -> majority vote -> update policy
                                                # 2) ML inference on ALL windows -> (weighted) vote -> update policy
                        if self.lr_policy is not None and windows:
                            labels_for_policy: list[PLLabel] = []
                            label_counts: Dict[str, int] = {}
                            ok_cnt = 0

                            weighted_scores: Dict[str, float] = {}   # label -> sum(weight)
                            conf_values: list[float] = []
                            rop_scores: list[float] = []

                            # OPTIONAL: labels that come from prover windows (if your windows include it)
                            prover_counts: Dict[str, int] = {}

                            for w in windows:
                                # if prover already includes a label field in the window, log it too
                                pl_from_prover = w.get("label")
                                if pl_from_prover is not None:
                                    ps = str(pl_from_prover)
                                    prover_counts[ps] = prover_counts.get(ps, 0) + 1

                                pr = self.lr_policy.predict(dev, w)
                                if pr.get("ok") and pr.get("label") is not None:
                                    ok_cnt += 1
                                    pl = self._map_model_label_to_policy_label(pr["label"])
                                    labels_for_policy.append(pl)

                                    label_counts[pl.value] = label_counts.get(pl.value, 0) + 1

                                    # weight by model confidence if available else 1.0
                                    wc = pr.get("model_conf")
                                    wgt = float(wc) if wc is not None else 1.0
                                    weighted_scores[pl.value] = weighted_scores.get(pl.value, 0.0) + wgt
                                    if wc is not None:
                                        conf_values.append(float(wc))

                                    # rop score (P(light_rop)+P(heavy_rop))
                                    rs = pr.get("rop_score")
                                    if rs is not None:
                                        rop_scores.append(float(rs))

                            # weighted majority
                            weighted_majority = None
                            weighted_conf = None
                            if weighted_scores:
                                total_weight = sum(weighted_scores.values())
                                weighted_majority, best_w = max(weighted_scores.items(), key=lambda kv: kv[1])
                                weighted_conf = (best_w / total_weight) if total_weight > 0 else 0.0
                                weighted_majority = self._map_model_label_to_policy_label(weighted_majority)

                            model_conf_avg = (sum(conf_values) / len(conf_values)) if conf_values else None
                            model_conf_min = min(conf_values) if conf_values else None
                            model_conf_max = max(conf_values) if conf_values else None
                            rop_score_avg = (sum(rop_scores) / len(rop_scores)) if rop_scores else None

                            # update policy with extra confidence info
                            summ = self.policy.on_inference_batch(
                                dev,
                                labels_for_policy,
                                now=time.time(),
                                weighted_majority=weighted_majority,
                                weighted_confidence=weighted_conf,
                                model_conf_avg=model_conf_avg,
                                model_conf_min=model_conf_min,
                                model_conf_max=model_conf_max,
                                rop_score_avg=rop_score_avg,
                            )

                            majority = summ.majority.value
                            conf = float(summ.confidence)

                            # IMPORTANT: tick AFTER on_inference_batch (so new stable label affects scheduling)
                            decision = self.policy.tick(dev, now=time.time())

                            if fp_evt:
                                self._jwrite(fp_evt, {
                                    "ts_ms": ts_ms(),
                                    "device": dev,
                                    "event": "ml_inference_batch",
                                    "n_windows": len(windows),
                                    "n_ok": ok_cnt,
                                    "label_counts": label_counts,

                                    "majority_label": majority,
                                    "majority_frac": round(conf, 3),

                                    "weighted_majority_label": (summ.weighted_majority.value if summ.weighted_majority else None),
                                    "weighted_majority_frac": (round(float(summ.weighted_confidence), 3) if summ.weighted_confidence is not None else None),

                                    "model_conf_avg": (round(float(model_conf_avg), 3) if model_conf_avg is not None else None),
                                    "model_conf_min": (round(float(model_conf_min), 3) if model_conf_min is not None else None),
                                    "model_conf_max": (round(float(model_conf_max), 3) if model_conf_max is not None else None),

                                    "rop_score_avg": (round(float(rop_score_avg), 3) if rop_score_avg is not None else None),

                                    "policy_stable_label": self.policy.devices[dev].stable_label.value,
                                    "policy_reason": self.policy.devices[dev].last_reason,

                                    "prover_label_counts": prover_counts if prover_counts else None,

                                    "window_id_range": {
                                        "from": windows[0].get("window_id"),
                                        "to": windows[-1].get("window_id"),
                                    },
                                })


            # 3) ATTEST if due (policy-driven)
            if decision.attest_kind != PLAttestKind.NONE:
                # build ml meta snapshot for logs
                st = self.policy.devices.get(dev)
                ml_meta = {
                    "policy_stable_label": st.stable_label.value,
                    "policy_last_majority": st.last_majority.value,
                    "policy_conf": round(float(st.last_confidence), 3),
                    "policy_reason": st.last_reason,

                    "weighted_majority": (st.last_weighted_majority.value if st.last_weighted_majority else None),
                    "weighted_conf": (round(float(st.last_weighted_conf), 3) if st.last_weighted_conf is not None else None),

                    "model_conf_avg": (round(float(st.last_model_conf_avg), 3) if st.last_model_conf_avg is not None else None),
                    "model_conf_min": (round(float(st.last_model_conf_min), 3) if st.last_model_conf_min is not None else None),
                    "model_conf_max": (round(float(st.last_model_conf_max), 3) if st.last_model_conf_max is not None else None),

                    "rop_score_avg": (round(float(st.last_rop_score_avg), 3) if st.last_rop_score_avg is not None else None),
                }


                if decision.attest_kind == PLAttestKind.FULL:
                    asyncio.create_task(self.attest_full_and_log(dev, trigger="POLICY", ml=ml_meta))
                elif decision.attest_kind == PLAttestKind.PARTIAL:
                    asyncio.create_task(self.attest_partial_and_log(dev, k=int(decision.k), trigger="POLICY", ml=ml_meta))

            await asyncio.sleep(LOOP_TICK_S + random.random() * JITTER_S)

    # ----------------- client handler -----------------
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        print(f"[{now_s()}] [+] Connection from {peer}")

        # Expect HELLO
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        except asyncio.TimeoutError:
            print(f"[{now_s()}] [-] No HELLO from {peer}, closing")
            writer.close()
            await writer.wait_closed()
            return

        if not line:
            writer.close()
            await writer.wait_closed()
            return

        try:
            hello = json.loads(line.decode("utf-8"))
        except Exception:
            print(f"[{now_s()}] [-] Bad HELLO JSON from {peer}: {line!r}")
            writer.close()
            await writer.wait_closed()
            return

        if hello.get("type") != "HELLO" or "device_id" not in hello:
            print(f"[{now_s()}] [-] Expected HELLO with device_id, got: {hello}")
            writer.close()
            await writer.wait_closed()
            return

        device_id = hello["device_id"]
        dc = DeviceConn(device_id=device_id, reader=reader, writer=writer)
        self.devices[device_id] = dc
        if self.selected_device is None:
            self.selected_device = device_id

        self._open_files_for(device_id)
        fp_evt = self.events_fp.get(device_id)
        if fp_evt:
            self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": device_id, "event": "device_registered"})

        # Start policy loop
        if device_id not in self.policy_tasks or self.policy_tasks[device_id].done():
            self.policy_tasks[device_id] = asyncio.create_task(self.policy_loop(device_id))
            print(f"[{now_s()}] [POLICY] started for {device_id}")

        # Start RX loop
        try:
            await self.rx_loop(dc)
        finally:
            # stop policy loop
            task = self.policy_tasks.get(device_id)
            if task and not task.done():
                task.cancel()
            self.policy_tasks.pop(device_id, None)

            # close files
            self._close_files_for(device_id)

            # cleanup registry
            if self.devices.get(device_id) is dc:
                del self.devices[device_id]
                if self.selected_device == device_id:
                    self.selected_device = next(iter(self.devices), None)

            writer.close()
            await writer.wait_closed()
            print(f"[{now_s()}] [x] Disconnected device_id={device_id}")

    # ----------------- optional CLI (same idea as before) -----------------
    def cli_thread(self):
        print("\nCLI commands:")
        print("  list")
        print("  use <device_id>")
        print("  ping")
        print("  quit\n")

        while True:
            try:
                cmd = input("verifier_policy> ").strip()
            except EOFError:
                cmd = "quit"

            if cmd == "":
                continue

            if cmd == "quit":
                print("bye.")
                self.loop.call_soon_threadsafe(self.loop.stop)
                return

            if cmd == "list":
                devs = list(self.devices.keys())
                print(f"connected: {devs} | selected={self.selected_device}")
                continue

            if cmd.startswith("use "):
                _, dev = cmd.split(" ", 1)
                dev = dev.strip()
                if dev in self.devices:
                    self.selected_device = dev
                    print(f"selected={dev}")
                else:
                    print("no such device connected")
                continue

            dev = self.selected_device
            if not dev:
                print("no device selected/connected")
                continue

            if cmd == "ping":
                coro = self.send_request(dev, {"type": "PING"})
            else:
                print("unknown command")
                continue

            fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
            try:
                resp = fut.result(timeout=8.0)
                print(f"[{now_s()}] [RESP] {resp}")
            except Exception as e:
                print("error waiting:", e)


async def main():
    try:
        with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
            golden = json.load(f)
    except FileNotFoundError:
        golden = {}

    srv = VerifierPolicyServer(golden)
    srv.loop = asyncio.get_running_loop()

    # CLI thread (optional)
    t = threading.Thread(target=srv.cli_thread, daemon=True)
    t.start()

    server = await asyncio.start_server(srv.handle_client, HOST, PORT)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"[{now_s()}] Verifier POLICY listening on {addrs}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
