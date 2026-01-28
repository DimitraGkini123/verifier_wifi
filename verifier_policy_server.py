# verifier_policy_server.py
import os
from pathlib import Path
import asyncio
import json
import secrets
import time
import random
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple

from utils import save_json_atomic, jdump, sha256, ts_ms, now_s, unhex
from lru_blocks import DeviceLRUBlocks
from policy_device_lr import DeviceLRPolicy

# Policy engine that schedules GET_WINDOWS + ATTEST based on stable label
from verifier_policy import PolicyEngine, Label as PLLabel, AttestKind as PLAttestKind

LRU_STATE_PATH = "lru_state.json"

HOST = "0.0.0.0"
PORT = 4242
GOLDEN_PATH = "golden.json"

# NEW: budget config file
BUDGET_CFG_PATH = "budget_config.json"
LOG_DIR = "logs"
TRUST_UNKNOWN = "UNKNOWN"
TRUST_TRUSTED = "TRUSTED"
TRUST_UNTRUSTED = "UNTRUSTED"

#quarantine configs 
QUARANTINE_ON_FULL_FAILS = 2          # trigger after 2 consecutive FULL failures
QUARANTINE_RECHECK_S = 10.0          # how often to attempt recovery FULL while quarantined


# ---------------- ML / Policy config ----------------
MODEL_PATH = "models/verifier_models_per_device.joblib"
ML_ENABLE = True

# Policy hysteresis: πόσες φορές πρέπει να δεις majority διαφορετικό για να αλλάξεις stable label
POLICY_HYSTERESIS_N = 2

# Safety jitter (μικρό) για να μη συγχρονίζονται πολλά devices (αν έχεις πολλά)
LOOP_TICK_S = 0.20
JITTER_S = 0.05

# Initial full attestation on connect?
DO_INITIAL_FULL_ATTEST = True


# =========================
# Budgeting (token bucket)
# =========================

FULL_COST_UNITS = 100  # 100 budget = full hash

@dataclass
class BudgetConfig:
    per_min_units: int = 50
    cap_units: int = 50
    min_k: int = 1  # minimum partial k allowed (degrade not below this)

@dataclass
class DeviceBudgetState:
    tokens: float = 0.0
    last_ts: float = 0.0

class BudgetManager:
    """
    Token bucket per device:
      - refill continuously at per_min_units / 60 per sec
      - cap at cap_units
      - spend immediately when scheduling to avoid races
    """

    def __init__(self, default_cfg: BudgetConfig, per_device_cfg: Optional[Dict[str, BudgetConfig]] = None):
        self.default_cfg = default_cfg
        self.per_device_cfg = per_device_cfg or {}
        self.st: Dict[str, DeviceBudgetState] = {}

    def _cfg(self, dev: str) -> BudgetConfig:
        return self.per_device_cfg.get(dev, self.default_cfg)

    def _state(self, dev: str, now: float) -> DeviceBudgetState:
        st = self.st.get(dev)
        if st is None:
            cfg = self._cfg(dev)
            st = DeviceBudgetState(tokens=float(cfg.cap_units), last_ts=now)
            self.st[dev] = st
        return st

    def _refill(self, dev: str, now: float) -> None:
        cfg = self._cfg(dev)
        st = self._state(dev, now)
        dt = max(0.0, now - st.last_ts)
        rate = float(cfg.per_min_units) / 60.0  # units per second
        st.tokens = min(float(cfg.cap_units), st.tokens + dt * rate)
        st.last_ts = now

    def tokens_now(self, dev: str, now: float) -> float:
        self._refill(dev, now)
        return self.st[dev].tokens

    def spend(self, dev: str, cost: float, now: float) -> bool:
        self._refill(dev, now)
        st = self.st[dev]
        if st.tokens + 1e-9 >= cost:
            st.tokens -= float(cost)
            return True
        return False

    def refund(self, dev: str, cost: float, now: float) -> None:
        self._refill(dev, now)
        cfg = self._cfg(dev)
        st = self.st[dev]
        st.tokens = min(float(cfg.cap_units), st.tokens + float(cost))

    # ----- cost model -----
    @staticmethod
    def cost_full() -> int:
        return FULL_COST_UNITS

    @staticmethod
    def cost_partial(k: int, block_count: int) -> int:
        """
        Proportional cost:
          FULL = 100 units
          PARTIAL(k) ≈ 100 * k / block_count
        Minimum 1 unit to avoid "free" attest.
        """
        bc = max(1, int(block_count))
        kk = max(1, min(int(k), bc))
        return max(1, int(round(FULL_COST_UNITS * (kk / float(bc)))))

    def fit_plan(
        self,
        dev: str,
        now: float,
        ideal_kind: str,
        ideal_k: int,
        block_count: int,
        min_k: int = 1
    ) -> Tuple[str, int, int, str]:
        """
        Decide what can be executed given current tokens.
        Returns: (kind, k, cost_units, reason)
          kind in {"NONE","PARTIAL","FULL"}
        """
        toks = self.tokens_now(dev, now)

        # Normalize limits
        bc = max(1, int(block_count))  # used only for costing; caller should still check real bc>0 for partial
        mk = max(1, min(int(min_k), bc))

        if ideal_kind == "FULL":
            c = self.cost_full()
            if toks >= c:
                return ("FULL", 0, c, "budget_ok_full")

            # Degrade FULL -> PARTIAL (largest k that fits, but not below min_k)
            for kk in range(bc, mk - 1, -1):
                cc = self.cost_partial(kk, bc)
                if toks >= cc:
                    return ("PARTIAL", kk, cc, f"degrade_full_to_partial:k={kk}")
            return ("NONE", 0, 0, "budget_insufficient_for_any_attest")

        if ideal_kind == "PARTIAL":
            kk0 = max(mk, min(int(ideal_k), bc))
            for kk in range(kk0, mk - 1, -1):
                cc = self.cost_partial(kk, bc)
                if toks >= cc:
                    if kk == kk0:
                        return ("PARTIAL", kk, cc, "budget_ok_partial")
                    return ("PARTIAL", kk, cc, f"degrade_partial:k={kk0}->{kk}")
            return ("NONE", 0, 0, "budget_insufficient_for_partial")

        return ("NONE", 0, 0, "policy_none")


# =========================
# Server structures
# =========================

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
    
    #quarantine!!
    full_hash_fail_streak: int = 0          # counts consecutive FULL failures (final result)
    quarantined: bool = False
    quarantine_since_ts: float = 0.0
    last_quarantine_check_ts: float = 0.0   # throttle full retries while quarantined

    def is_alive(self) -> bool:
        return not self.writer.is_closing()


class VerifierPolicyServer:
    def __init__(self, golden_db: dict):
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

        # Policy engine per server
        self.policy = PolicyEngine(hysteresis_n=POLICY_HYSTERESIS_N, enable_get_windows=True)

        # Per-device policy task
        self.policy_tasks: Dict[str, asyncio.Task] = {}

        # Budget (from budget_config.json)
        per_dev_budget, per_dev_min_k = self._load_budget_config(BUDGET_CFG_PATH)
        self.device_min_k: Dict[str, int] = per_dev_min_k

        self.budget = BudgetManager(
            default_cfg=BudgetConfig(per_min_units=50, cap_units=50, min_k=1),
            per_device_cfg=per_dev_budget
        )
    # ----------------- quarantine -----------------
    def _enter_quarantine(self, dev: str, reason: str):
        dc = self.devices.get(dev)
        if not dc:
            return
        if dc.quarantined:
            return

        dc.quarantined = True
        dc.quarantine_since_ts = time.time()
        dc.last_quarantine_check_ts = 0.0
        dc.trust_state = TRUST_UNTRUSTED
        dc.full_hash_fail_streak = max(dc.full_hash_fail_streak, QUARANTINE_ON_FULL_FAILS)

        fp_evt = self.events_fp.get(dev)
        if fp_evt:
            self._jwrite(fp_evt, {
                "ts_ms": ts_ms(),
                "device": dev,
                "event": "quarantine_entered",
                "reason": reason,
                "full_hash_fail_streak": dc.full_hash_fail_streak,
            })

    def _exit_quarantine(self, dev: str, reason: str = "full_hash_recovered"):
        dc = self.devices.get(dev)
        if not dc:
            return
        if not dc.quarantined:
            return

        dc.quarantined = False
        dc.quarantine_since_ts = 0.0
        dc.last_quarantine_check_ts = 0.0
        dc.full_hash_fail_streak = 0  # reset

        fp_evt = self.events_fp.get(dev)
        if fp_evt:
            self._jwrite(fp_evt, {
                "ts_ms": ts_ms(),
                "device": dev,
                "event": "quarantine_exited",
                "reason": reason,
            })

    def _is_quarantined(self, dev: str) -> bool:
        dc = self.devices.get(dev)
        return bool(dc and dc.quarantined)

    async def _quarantine_periodic_recheck(self, dev: str):
        """
        Called while device is quarantined.
        Every QUARANTINE_RECHECK_S seconds, run a FULL attestation.
        If FULL is OK -> exit quarantine and resume normal operation.
        """
        dc = self.devices.get(dev)
        if not dc or not dc.quarantined:
            return

        now = time.time()

        # throttle: run at most once per QUARANTINE_RECHECK_S
        if (now - float(dc.last_quarantine_check_ts or 0.0)) < QUARANTINE_RECHECK_S:
            return

        dc.last_quarantine_check_ts = now

        fp_evt = self.events_fp.get(dev)
        if fp_evt:
            self._jwrite(fp_evt, {
                "ts_ms": ts_ms(),
                "device": dev,
                "event": "quarantine_recheck_start",
                "interval_s": QUARANTINE_RECHECK_S,
            })

        # FULL recheck (uses your existing retry + logging + trust update)
        resp = await self.attest_full_and_log(dev, trigger="QUARANTINE_RECHECK", ml={
            "policy_reason": "quarantine_recheck",
        })

        ok = bool(isinstance(resp, dict) and resp.get("verify_ok", False))

        if ok:
            self._exit_quarantine(dev, reason="recheck_full_ok")

            # optional but recommended: reset policy state so it doesn't keep stale history
            try:
                if hasattr(self.policy, "reset_device"):
                    self.policy.reset_device(dev)  # if you implemented it
                else:
                    if hasattr(self.policy, "devices") and isinstance(self.policy.devices, dict):
                        self.policy.devices.pop(dev, None)
            except Exception:
                pass

            dc2 = self.devices.get(dev)
            if dc2:
                dc2.trust_state = TRUST_TRUSTED
                dc2.full_hash_fail_streak = 0

            if fp_evt:
                self._jwrite(fp_evt, {
                    "ts_ms": ts_ms(),
                    "device": dev,
                    "event": "quarantine_recheck_ok_exit",
                })
        else:
            if fp_evt:
                self._jwrite(fp_evt, {
                    "ts_ms": ts_ms(),
                    "device": dev,
                    "event": "quarantine_recheck_failed_stay",
                    "verify_reason": (resp.get("verify_reason") if isinstance(resp, dict) else "bad_resp"),
                })



    # ----------------- budget config loader -----------------
    def _load_budget_config(self, path: str) -> Tuple[Dict[str, BudgetConfig], Dict[str, int]]:
        per_device_budget: Dict[str, BudgetConfig] = {}
        per_device_min_k: Dict[str, int] = {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except FileNotFoundError:
            print(f"[{now_s()}] [BUDGET] No {path} found. Using defaults.")
            return per_device_budget, per_device_min_k
        except Exception as e:
            print(f"[{now_s()}] [BUDGET] Failed to load {path}: {e}. Using defaults.")
            return per_device_budget, per_device_min_k

        levels = cfg.get("levels", {}) or {}
        devices = cfg.get("devices", {}) or {}

        for dev, level_name in devices.items():
            lvl = levels.get(level_name, {}) or {}
            per_min = int(lvl.get("per_min_units", 50))
            cap = int(lvl.get("cap_units", per_min))
            min_k = int(lvl.get("min_k", 1))

            per_device_budget[dev] = BudgetConfig(per_min_units=per_min, cap_units=cap, min_k=max(1, min_k))
            per_device_min_k[dev] = max(1, min_k)

        print(f"[{now_s()}] [BUDGET] loaded {len(per_device_budget)} device budgets from {path}")
        return per_device_budget, per_device_min_k

    # ----------------- file helpers -----------------
    def _open_files_for(self, dev: str):
        if dev in self.windows_fp:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        wpath = str(Path(LOG_DIR) / f"windows_{dev}_{stamp}.jsonl")
        epath = str(Path(LOG_DIR) / f"events_{dev}_{stamp}.jsonl")
        apath = str(Path(LOG_DIR) / f"attest_{dev}_{stamp}.jsonl")
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
    async def force_provision_golden_full(self, dev: str, region: str = "fw") -> dict:
        nonce = secrets.token_hex(8)
        resp = await self.send_request(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "FULL_HASH_PROVER",
            "region": region,
            "nonce": nonce
        }, timeout=12.0)

        fw_hex = resp.get("fw_hash_hex")
        if not fw_hex:
            return {"type": "ERROR", "reason": "missing_fw_hash_hex_in_response", "resp": resp}

        self.set_golden_full_hash(dev, region, fw_hex)
        
        return {"type": "OK", "event": "golden_overwritten", "device": dev, "region": region, "fw_hash_hex": fw_hex}
    
    def set_golden_full_hash(self, device_id: str, region: str, fw_hash_hex: str):
        if device_id not in self.golden:
            self.golden[device_id] = {}
        if region not in self.golden[device_id]:
            self.golden[device_id][region] = {}
        self.golden[device_id][region]["sha256"] = fw_hash_hex.lower()
        save_json_atomic(GOLDEN_PATH, self.golden)

    def set_golden_blocks(self, device_id: str, block_size: int, hashes_hex: list[str], force: bool = False):
        if device_id not in self.golden:
            self.golden[device_id] = {}

        # guard: μην overwrite κατά λάθος
        if (not force) and self.has_golden_blocks(device_id):
            raise RuntimeError("golden_blocks_already_exist_refusing_overwrite")

        self.golden[device_id]["blocks"] = {
            "block_size": int(block_size),
            "block_count": int(len(hashes_hex)),
            "hashes": [h.lower() for h in hashes_hex],
        }
        save_json_atomic(GOLDEN_PATH, self.golden)

    async def provision_golden_blocks(self, dev: str, force: bool = False) -> dict:
        # guard: μην overwrite αν υπάρχει ήδη
        if self.has_golden_blocks(dev) and not force:
            return {"type": "ERROR", "reason": "golden_blocks_already_exists_refusing_overwrite", "device": dev}

        # 1) Probe (ζήτα 1 block) για να πάρεις metadata
        nonce = secrets.token_hex(8)
        probe = await self.send_request(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "PARTIAL_BLOCKS",
            "region": "fw",
            "nonce": nonce,
            "indices": [0]
        }, timeout=12.0)

        if probe.get("type") != "ATTEST_RESPONSE" or probe.get("mode") != "PARTIAL_BLOCKS":
            return {"type": "ERROR", "reason": "bad_probe_response", "resp": probe}

        block_count = int(probe.get("block_count", 0) or 0)

        # accept either "block_size" or infer from first block's "len"
        block_size = int(probe.get("block_size", 0) or 0)
        if block_size <= 0:
            blocks0 = probe.get("blocks", [])
            if blocks0 and isinstance(blocks0, list):
                block_size = int(blocks0[0].get("len", 0) or 0)

        if block_size <= 0 or block_count <= 0:
            return {"type": "ERROR", "reason": "missing_block_meta", "resp": probe}


        # 2) Ζήτα ΟΛΑ τα blocks
        nonce = secrets.token_hex(8)
        indices = list(range(block_count))
        resp = await self.send_request(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "PARTIAL_BLOCKS",
            "region": "fw",
            "nonce": nonce,
            "indices": indices
        }, timeout=30.0)

        if resp.get("type") != "ATTEST_RESPONSE" or resp.get("mode") != "PARTIAL_BLOCKS":
            return {"type": "ERROR", "reason": "bad_blocks_response", "resp": resp}

        blocks = resp.get("blocks", [])
        got = {}
        for b in blocks:
            if "index" in b and "hash_hex" in b:
                got[int(b["index"])] = b["hash_hex"]

        if len(got) < block_count:
            return {"type": "ERROR", "reason": "missing_some_blocks", "got": len(got), "need": block_count}

        hashes = [got[i] for i in range(block_count)]

        try:
            self.set_golden_blocks(dev, block_size, hashes, force=force)
        except RuntimeError as e:
            return {"type": "ERROR", "reason": str(e)}

        return {
            "type": "OK",
            "event": "golden_blocks_provisioned" if not force else "golden_blocks_overwritten",
            "device": dev,
            "block_size": block_size,
            "block_count": block_count
        }

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
    def _attest_lock(self, dev: str) -> asyncio.Lock:
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
        indices: Optional[List[int]],
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
            "budget_tokens_after": resp.get("_budget_tokens_after"),
            "budget_cost_units": resp.get("_budget_cost_units"),
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
        if not dc or not dc.is_alive():
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

            # 1) Do FULL (with retry)
            resp = await self.attest_full_with_retry(dev)

            # 2) Update trust (attest_full_with_retry already calls _update_trust_from_attest twice,
            #    but dc.trust_state is now updated. We'll use resp.verify_ok for the FULL-streak logic.)
            trust_after = dc.trust_state if dev in self.devices else TRUST_UNKNOWN

            # 3) FULL consecutive-fail streak + quarantine
            #    (counts only FULL attest results; ignores partial)
            if isinstance(resp, dict):
                ok = bool(resp.get("verify_ok", False))

                if ok:
                    dc.full_hash_fail_streak = 0

                    # If we were quarantined and FULL is now OK -> exit quarantine
                    if getattr(dc, "quarantined", False):
                        self._exit_quarantine(dev, reason="full_hash_ok")

                else:
                    dc.full_hash_fail_streak = int(getattr(dc, "full_hash_fail_streak", 0)) + 1

                    if dc.full_hash_fail_streak >= QUARANTINE_ON_FULL_FAILS:
                        self._enter_quarantine(
                            dev,
                            reason=f"full_hash_failed_{dc.full_hash_fail_streak}x:{resp.get('verify_reason')}"
                        )

            # budget meta -> resp for logging
            if isinstance(resp, dict) and ml:
                resp["_budget_cost_units"] = ml.get("budget_cost_units")
                resp["_budget_tokens_after"] = ml.get("budget_tokens_after")

            self.log_attest_event(
                dev=dev, kind="FULL", k=None, indices=None,
                resp=resp if isinstance(resp, dict) else {"type": "ERROR", "reason": "bad_resp"},
                trust_before=trust_before, trust_after=trust_after,
                trigger=trigger, ml=ml
            )
            return resp
    async def _attest_full_inner_and_log(self, dev: str, trigger: str = "POLICY", ml: Optional[dict] = None, force_quarantine_on_fail: bool = False,) -> dict:
        """
        SAME as attest_full_and_log but assumes caller ALREADY holds self._attest_lock(dev).
        This avoids deadlocks when we escalate FULL from inside partial_and_log.
        """
        dc = self.devices.get(dev)
        if not dc:
            return {"type": "ERROR", "reason": "device_not_connected"}

        trust_before = dc.trust_state

        # 1) Do FULL (with retry)
        resp = await self.attest_full_with_retry(dev)

        trust_after = dc.trust_state if dev in self.devices else TRUST_UNKNOWN

        # 2) FULL failure streak + quarantine logic
        if isinstance(resp, dict):
            ok = bool(resp.get("verify_ok", False))

            if ok:
                dc.full_hash_fail_streak = 0
                if getattr(dc, "quarantined", False):
                    self._exit_quarantine(dev, reason="full_hash_ok")

            else:
                # increment streak
                dc.full_hash_fail_streak = int(getattr(dc, "full_hash_fail_streak", 0)) + 1

                # If this FULL came from escalation, quarantine immediately on 1 fail
                if force_quarantine_on_fail:
                    self._enter_quarantine(dev, reason=f"escalated_full_failed:{resp.get('verify_reason')}")
                else:
                    if dc.full_hash_fail_streak >= QUARANTINE_ON_FULL_FAILS:
                        self._enter_quarantine(
                            dev,
                            reason=f"full_hash_failed_{dc.full_hash_fail_streak}x:{resp.get('verify_reason')}"
                        )

        # budget meta -> resp for logging
        if isinstance(resp, dict) and ml:
            resp["_budget_cost_units"] = ml.get("budget_cost_units")
            resp["_budget_tokens_after"] = ml.get("budget_tokens_after")

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

            # 1) PARTIAL
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

            # budget meta -> resp for logging
            if isinstance(resp, dict) and ml:
                resp["_budget_cost_units"] = ml.get("budget_cost_units")
                resp["_budget_tokens_after"] = ml.get("budget_tokens_after")

            # log PARTIAL
            self.log_attest_event(
                dev=dev, kind="PARTIAL", k=kk, indices=indices,
                resp=resp if isinstance(resp, dict) else {"type": "ERROR", "reason": "bad_resp"},
                trust_before=trust_before, trust_after=trust_after,
                trigger=trigger, ml=ml
            )

            # 2) If PARTIAL failed -> FULL immediately -> if FULL fails -> quarantine immediately
            if isinstance(resp, dict) and not bool(resp.get("verify_ok", False)):
                fp_evt = self.events_fp.get(dev)
                if fp_evt:
                    self._jwrite(fp_evt, {
                        "ts_ms": ts_ms(),
                        "device": dev,
                        "event": "partial_failed_escalate_full",
                        "partial_reason": resp.get("verify_reason", resp.get("reason")),
                    })

                ml2 = dict(ml or {})
                ml2["escalation_from"] = "PARTIAL_FAIL"

                full_resp = await self._attest_full_inner_and_log(
                    dev=dev,
                    trigger="ESCALATE_FROM_PARTIAL",
                    ml=ml2,
                    force_quarantine_on_fail=True,   # <-- ΕΔΩ κλειδώνεις quarantine στο 1ο FULL fail
                )
                return full_resp

            return resp


    # ----------------- policy loop helpers -----------------
    @staticmethod
    def _map_model_label_to_policy_label(x) -> PLLabel:
        """
        Accept ints or strings. Common mapping:
          0=LIGHT, 1=MEDIUM, 2=HEAVY, 3=SUSPICIOUS
        """
        try:
            if isinstance(x, int):
                return {0: PLLabel.LIGHT, 1: PLLabel.MEDIUM, 2: PLLabel.HEAVY, 3: PLLabel.SUSPICIOUS}.get(x, PLLabel.MEDIUM)
            s = str(x).strip().upper()
            if s.isdigit():
                return {0: PLLabel.LIGHT, 1: PLLabel.MEDIUM, 2: PLLabel.HEAVY, 3: PLLabel.SUSPICIOUS}.get(int(s), PLLabel.MEDIUM)
            if "SUSP" in s:
                return PLLabel.SUSPICIOUS
            if "HEAV" in s:
                return PLLabel.HEAVY
            if "LIGHT" in s:
                return PLLabel.LIGHT
            if "MED" in s:
                return PLLabel.MEDIUM
        except Exception:
            pass
        return PLLabel.MEDIUM

    async def policy_loop(self, dev: str):
        self._open_files_for(dev)
        fp_evt = self.events_fp.get(dev)

        # initial full attest (optional) - bypass budget (it's provisioning / baseline)
        if DO_INITIAL_FULL_ATTEST:
            await asyncio.sleep(0.5)
            if dev in self.devices:
                if fp_evt:
                    self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "initial_full_attest_start"})
                await self.attest_full_and_log(dev, trigger="INITIAL")

        while True:
            if dev not in self.devices:
                return

            now = time.time()
            dc = self.devices.get(dev)   # <-- ΠΡΕΠΕΙ να υπάρχει, αλλιώς dc is undefined


            # tick policy
            decision = self.policy.tick(dev, now=now)

            if dc and getattr(dc, "quarantined", False):
                # In quarantine: do NOT send windows, do NOT do policy attest.
                # Only do periodic FULL recheck.
                if fp_evt:
                    if decision.do_get_windows:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "quarantine_skip_get_windows",
                            "policy_reason": decision.reason
                        })
                    if decision.attest_kind != PLAttestKind.NONE:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "quarantine_skip_policy_attest",
                            "policy_attest_kind": str(decision.attest_kind),
                            "policy_reason": decision.reason
                        })

                # throttle recheck
                tnow = time.time()
                if (tnow - float(dc.last_quarantine_check_ts or 0.0)) >= QUARANTINE_RECHECK_S:
                    dc.last_quarantine_check_ts = tnow

                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "quarantine_recheck_start",
                            "interval_s": QUARANTINE_RECHECK_S
                        })

                    # FULL recheck (already updates trust + can exit quarantine in your attest_full_and_log)
                    resp = await self.attest_full_and_log(dev, trigger="QUARANTINE_RECHECK", ml={
                        "policy_reason": "quarantine_recheck"
                    })

                    ok = bool(isinstance(resp, dict) and resp.get("verify_ok", False))
                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "quarantine_recheck_done",
                            "verify_ok": ok,
                            "verify_reason": (resp.get("verify_reason") if isinstance(resp, dict) else "bad_resp")
                        })

                await asyncio.sleep(LOOP_TICK_S + random.random() * JITTER_S)
                continue
            # =========================
            # END QUARANTINE GATE
            # =========================

            # 1) GET_WINDOWS if due
            if decision.do_get_windows:
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
                        if self.lr_policy is not None and windows:
                            labels_for_policy: List[PLLabel] = []
                            label_counts: Dict[str, int] = {}
                            ok_cnt = 0

                            weighted_scores: Dict[str, float] = {}  # label -> sum(weight)
                            conf_values: List[float] = []

                            for w in windows:
                                pr = self.lr_policy.predict(dev, w)
                                if pr.get("ok") and pr.get("label") is not None:
                                    ok_cnt += 1
                                    pl = self._map_model_label_to_policy_label(pr["label"])
                                    labels_for_policy.append(pl)
                                    label_counts[pl.value] = label_counts.get(pl.value, 0) + 1

                                    wc = pr.get("model_conf")
                                    wgt = float(wc) if wc is not None else 1.0
                                    weighted_scores[pl.value] = weighted_scores.get(pl.value, 0.0) + wgt
                                    if wc is not None:
                                        conf_values.append(float(wc))

                            # Update policy with (simple) majority of labels_for_policy
                            summ = self.policy.on_inference_batch(dev, labels_for_policy, now=time.time())
                            # refresh decision after inference
                            decision = self.policy.tick(dev, now=time.time())

                            # Weighted majority (debug info only)
                            weighted_majority = None
                            weighted_conf = 0.0
                            total_weight = sum(weighted_scores.values()) if weighted_scores else 0.0
                            if weighted_scores and total_weight > 0:
                                weighted_majority, best_w = max(weighted_scores.items(), key=lambda kv: kv[1])
                                weighted_conf = best_w / total_weight

                            model_conf_avg = (sum(conf_values) / len(conf_values)) if conf_values else None
                            model_conf_min = (min(conf_values)) if conf_values else None
                            model_conf_max = (max(conf_values)) if conf_values else None

                            if fp_evt:
                                self._jwrite(fp_evt, {
                                    "ts_ms": ts_ms(),
                                    "device": dev,
                                    "event": "ml_inference_batch",
                                    "n_windows": len(windows),
                                    "n_ok": ok_cnt,
                                    "label_counts": label_counts,
                                    "majority_label": summ.majority.value,
                                    "majority_frac": round(float(summ.confidence), 3),
                                    "policy_stable_label": self.policy.devices[dev].stable_label.value,
                                    "policy_reason": self.policy.devices[dev].last_reason,
                                    "weighted_majority_label": weighted_majority,
                                    "weighted_majority_frac": round(float(weighted_conf), 3) if weighted_majority else None,
                                    "model_conf_avg": round(float(model_conf_avg), 3) if model_conf_avg is not None else None,
                                    "model_conf_min": round(float(model_conf_min), 3) if model_conf_min is not None else None,
                                    "model_conf_max": round(float(model_conf_max), 3) if model_conf_max is not None else None,
                                    "window_id_range": {
                                        "from": windows[0].get("window_id") if windows else None,
                                        "to": windows[-1].get("window_id") if windows else None,
                                    },
                                })

            # 3) ATTEST if due (policy-driven) + BUDGET GATE
            if decision.attest_kind != PLAttestKind.NONE:
                st = self.policy.devices.get(dev)
                ml_meta = {
                    "policy_stable_label": st.stable_label.value if st else None,
                    "policy_last_majority": st.last_majority.value if st else None,
                    "policy_conf": round(float(st.last_confidence), 3) if st else None,
                    "policy_reason": st.last_reason if st else None,
                }

                # Ideal plan from policy
                if decision.attest_kind == PLAttestKind.FULL:
                    ideal_kind = "FULL"
                    ideal_k = 0
                else:
                    ideal_kind = "PARTIAL"
                    ideal_k = int(decision.k)

                real_bc = self.get_block_count(dev)
                min_k = int(self.device_min_k.get(dev, 1))
                now2 = time.time()

                tokens_before = self.budget.tokens_now(dev, now2)

                fit_kind, fit_k, cost_units, budget_reason = self.budget.fit_plan(
                    dev=dev,
                    now=now2,
                    ideal_kind=ideal_kind,
                    ideal_k=ideal_k,
                    block_count=max(1, real_bc if real_bc > 0 else 1),  # for costing only
                    min_k=min_k
                )

                # If partial selected but no blocks exist, force NONE (avoid pointless scheduling)
                if fit_kind == "PARTIAL" and real_bc <= 0:
                    fit_kind, fit_k, cost_units, budget_reason = ("NONE", 0, 0, "no_golden_blocks_for_partial")

                fp_evt = self.events_fp.get(dev)
                if fp_evt:
                    self._jwrite(fp_evt, {
                        "ts_ms": ts_ms(),
                        "device": dev,
                        "event": "budget_gate",
                        "ideal": {"kind": ideal_kind, "k": ideal_k},
                        "fit": {"kind": fit_kind, "k": fit_k, "cost_units": cost_units},
                        "min_k": min_k,
                        "block_count": real_bc,
                        "tokens_before": round(tokens_before, 3),
                        "reason": budget_reason,
                    })

                if fit_kind != "NONE":
                    # Spend budget immediately (reserve) to avoid races
                    ok_spend = self.budget.spend(dev, cost_units, now2)
                    if not ok_spend:
                        if fp_evt:
                            self._jwrite(fp_evt, {
                                "ts_ms": ts_ms(),
                                "device": dev,
                                "event": "budget_spend_failed_after_fit",
                                "cost_units": cost_units
                            })
                    else:
                        tokens_after = self.budget.tokens_now(dev, now2)

                        ml_meta = dict(ml_meta)
                        ml_meta["budget_reason"] = budget_reason
                        ml_meta["budget_cost_units"] = int(cost_units)
                        ml_meta["budget_tokens_after"] = round(tokens_after, 3)
                        ml_meta["budget_tokens_before"] = round(tokens_before, 3)
                        ml_meta["budget_fit_kind"] = fit_kind
                        ml_meta["budget_fit_k"] = int(fit_k)

                        if fit_kind == "FULL":
                            asyncio.create_task(self.attest_full_and_log(dev, trigger="POLICY", ml=ml_meta))
                        else:
                            asyncio.create_task(self.attest_partial_and_log(dev, k=int(fit_k), trigger="POLICY", ml=ml_meta))

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

    # ----------------- optional CLI -----------------
    def cli_thread(self):
        print("\nCLI commands:")
        print("  list")
        print("  use <device_id>")
        print("  ping")
        print("  budget")
        print("  force_provision_golden  (overwrite golden!)")
        print("  force_provision_blocks (overwrite blocks!)")
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

            if cmd == "budget":
                devs = list(self.devices.keys())
                now = time.time()
                for d in devs:
                    t = self.budget.tokens_now(d, now)
                    mk = self.device_min_k.get(d, 1)
                    bc = self.get_block_count(d)
                    print(f"{d}: tokens={t:.2f}, min_k={mk}, block_count={bc}")
                continue

            dev = self.selected_device
            if not dev:
                print("no device selected/connected")
                continue

            if cmd == "ping":
                coro = self.send_request(dev, {"type": "PING"})
            elif cmd == "force_provision_golden":
                coro = self.force_provision_golden_full(dev, region="fw")
            elif cmd == "force_provision_blocks":
                coro = self.provision_golden_blocks(dev, force=True)

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

    # CLI thread
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
