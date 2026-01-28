# verifier_policy_server_rop.py

import asyncio
import json
import secrets
import time
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from utils import save_json_atomic, jdump, sha256, ts_ms, now_s, unhex
from lru_blocks import DeviceLRUBlocks

# ROP-aware LR policy
from policy_device_lr_rop import DeviceLRPolicy

# ROP-aware policy engine
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

POLICY_HYSTERESIS_N = 2

LOOP_TICK_S = 0.20
JITTER_S = 0.05

DO_INITIAL_FULL_ATTEST = True

# ---------------- AUTO GOLDEN PROVISION ----------------
AUTO_PROVISION_GOLDEN_FULL = True
AUTO_PROVISION_GOLDEN_BLOCKS = True
AUTO_PROVISION_REGION = "fw"


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
    def __init__(self, golden_db: dict):
        self.golden = golden_db

        self.devices: Dict[str, DeviceConn] = {}
        self.selected_device: Optional[str] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self.last_seen: Dict[str, int] = {}

        self.windows_fp: Dict[str, Any] = {}
        self.events_fp: Dict[str, Any] = {}
        self.attest_fp: Dict[str, Any] = {}

        self.attest_locks: Dict[str, asyncio.Lock] = {}

        self.block_lru: Dict[str, DeviceLRUBlocks] = {}
        self._load_lru_state()

        self.lr_policy = DeviceLRPolicy(MODEL_PATH) if ML_ENABLE else None

        self.policy = PolicyEngine(hysteresis_n=POLICY_HYSTERESIS_N, enable_get_windows=True)
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
        # ----------------- golden write/provision -----------------
    def has_golden_full(self, device_id: str, region: str = "fw") -> bool:
        try:
            _ = self.golden[device_id][region]["sha256"]
            return True
        except Exception:
            return False

    def set_golden_full_hash(self, device_id: str, region: str, fw_hash_hex: str):
        if device_id not in self.golden:
            self.golden[device_id] = {}
        if region not in self.golden[device_id]:
            self.golden[device_id][region] = {}
        self.golden[device_id][region]["sha256"] = str(fw_hash_hex).lower()
        save_json_atomic(GOLDEN_PATH, self.golden)

    async def provision_golden_full(self, dev: str, region: str = "fw") -> dict:
        # guard: don't overwrite
        if self.has_golden_full(dev, region):
            return {
                "type": "ERROR",
                "reason": "golden_already_exists_refusing_overwrite",
                "device": dev,
                "region": region,
                "existing_sha256": self.golden.get(dev, {}).get(region, {}).get("sha256")
            }

        nonce = secrets.token_hex(8)
        resp = await self.send_request_timed(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "FULL_HASH_PROVER",
            "region": region,
            "nonce": nonce
        }, timeout=12.0)

        fw_hex = resp.get("fw_hash_hex") or resp.get("hash_hex")
        if not fw_hex:
            return {"type": "ERROR", "reason": "missing_fw_hash_hex_in_response", "resp": resp}

        self.set_golden_full_hash(dev, region, fw_hex)
        return {"type": "OK", "event": "golden_provisioned", "device": dev, "region": region, "fw_hash_hex": fw_hex}

    async def force_provision_golden_full(self, dev: str, region: str = "fw") -> dict:
        nonce = secrets.token_hex(8)
        resp = await self.send_request_timed(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "FULL_HASH_PROVER",
            "region": region,
            "nonce": nonce
        }, timeout=12.0)

        fw_hex = resp.get("fw_hash_hex") or resp.get("hash_hex")
        if not fw_hex:
            return {"type": "ERROR", "reason": "missing_fw_hash_hex_in_response", "resp": resp}

        self.set_golden_full_hash(dev, region, fw_hex)
        return {"type": "OK", "event": "golden_overwritten", "device": dev, "region": region, "fw_hash_hex": fw_hex}

    def set_golden_blocks(self, device_id: str, block_size: int, hashes_hex: list[str], force: bool = False):
        if device_id not in self.golden:
            self.golden[device_id] = {}

        if (not force) and self.has_golden_blocks(device_id):
            raise RuntimeError("golden_blocks_already_exist_refusing_overwrite")

        self.golden[device_id]["blocks"] = {
            "block_size": int(block_size),
            "block_count": int(len(hashes_hex)),
            "hashes": [str(h).lower() for h in hashes_hex],
        }
        save_json_atomic(GOLDEN_PATH, self.golden)

    async def provision_golden_blocks(self, dev: str, force: bool = False) -> dict:
        if self.has_golden_blocks(dev) and not force:
            return {"type": "ERROR", "reason": "golden_blocks_already_exists_refusing_overwrite", "device": dev}

        # 1) probe for metadata
        nonce = secrets.token_hex(8)
        probe = await self.send_request_timed(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "PARTIAL_BLOCKS",
            "region": "fw",
            "nonce": nonce,
            "indices": [0]
        }, timeout=12.0)

        if probe.get("type") != "ATTEST_RESPONSE" or probe.get("mode") != "PARTIAL_BLOCKS":
            return {"type": "ERROR", "reason": "bad_probe_response", "resp": probe}

        block_count = int(probe.get("block_count", 0) or 0)
        block_size = int(probe.get("block_size", 0) or 0)

        if block_size <= 0:
            blocks0 = probe.get("blocks", []) or []
            if blocks0 and isinstance(blocks0, list):
                block_size = int(blocks0[0].get("len", 0) or 0)

        if block_size <= 0 or block_count <= 0:
            return {"type": "ERROR", "reason": "missing_block_meta", "resp": probe}

        # 2) fetch all blocks
        nonce = secrets.token_hex(8)
        indices = list(range(block_count))
        resp = await self.send_request_timed(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "PARTIAL_BLOCKS",
            "region": "fw",
            "nonce": nonce,
            "indices": indices
        }, timeout=30.0)

        if resp.get("type") != "ATTEST_RESPONSE" or resp.get("mode") != "PARTIAL_BLOCKS":
            return {"type": "ERROR", "reason": "bad_blocks_response", "resp": resp}

        blocks = resp.get("blocks", []) or []
        got = {}
        for b in blocks:
            if "index" in b and ("hash_hex" in b):
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

    async def auto_provision_if_needed(self, dev: str):
        """
        Called at startup per-device: if golden missing, fetch it once (no overwrite).
        Uses attest lock so it doesn't collide with initial attestation / windows fetch.
        """
        fp_evt = self.events_fp.get(dev)

        async with self._attest_lock(dev):
            if dev not in self.devices:
                return

            if AUTO_PROVISION_GOLDEN_FULL and (not self.has_golden_full(dev, AUTO_PROVISION_REGION)):
                if fp_evt:
                    self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision_full_start"})
                r = await self.provision_golden_full(dev, region=AUTO_PROVISION_REGION)
                if fp_evt:
                    self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision_full_done", "resp": r})

            if AUTO_PROVISION_GOLDEN_BLOCKS and (not self.has_golden_blocks(dev)):
                if fp_evt:
                    self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision_blocks_start"})
                r = await self.provision_golden_blocks(dev, force=False)
                if fp_evt:
                    self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision_blocks_done", "resp": r})

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
        # FULL
        if mode == "FULL_HASH_PROVER" and rtype == "ATTEST_RESPONSE":
            golden = self.golden_full_hash(device_id, region=sent.get("region", "fw"))
            if golden is None:
                received["verify_ok"] = False
                received["verify_reason"] = "missing_golden_full_hash"
                return received

            nonce_hex = sent.get("nonce")

            # 1) direct hash field (useful for provisioning/debug)
            fw_hex = received.get("fw_hash_hex") or received.get("hash_hex")
            if fw_hex:
                try:
                    got = unhex(fw_hex)
                    ok = (got == golden)
                    received["verify_ok"] = ok
                    received["verify_reason"] = "direct_hash_match" if ok else "direct_hash_mismatch"
                    return received
                except Exception:
                    pass

            # 2) nonce-bound response
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

    # ----------------- policy loop (ROP-aware) -----------------
    @staticmethod
    def _map_model_label_to_policy_label(x) -> PLLabel:
        """
        Accepts strings like:
          light_safe, medium_safe, heavy_safe, light_rop, heavy_rop, suspicious
        Also tolerates LIGHT/MEDIUM/HEAVY (mapped to *_SAFE).
        """
        try:
            if isinstance(x, PLLabel):
                return x

            # allow legacy int mapping (best-effort)
            if isinstance(x, int):
                return {
                    0: PLLabel.LIGHT_SAFE,
                    1: PLLabel.MEDIUM_SAFE,
                    2: PLLabel.HEAVY_SAFE,
                    3: PLLabel.SUSPICIOUS,
                }.get(x, PLLabel.MEDIUM_SAFE)

            s = str(x).strip().lower()

            if "light_rop" in s or ("rop" in s and "light" in s):
                return PLLabel.LIGHT_ROP
            if "heavy_rop" in s or ("rop" in s and "heavy" in s):
                return PLLabel.HEAVY_ROP

            if "susp" in s:
                return PLLabel.SUSPICIOUS

            # safe labels
            if "light_safe" in s or s == "light":
                return PLLabel.LIGHT_SAFE
            if "medium_safe" in s or s == "medium":
                return PLLabel.MEDIUM_SAFE
            if "heavy_safe" in s or s == "heavy":
                return PLLabel.HEAVY_SAFE

            # fallback heuristics
            if "light" in s:
                return PLLabel.LIGHT_SAFE
            if "heavy" in s:
                return PLLabel.HEAVY_SAFE
            if "med" in s:
                return PLLabel.MEDIUM_SAFE

        except Exception:
            pass
        return PLLabel.MEDIUM_SAFE

    async def policy_loop(self, dev: str):
        self._open_files_for(dev)
        fp_evt = self.events_fp.get(dev)

        if DO_INITIAL_FULL_ATTEST:
            await asyncio.sleep(0.5)

            if dev in self.devices:
                # 1) auto-provision goldens (NO overwrite) αν λείπουν
                await self.auto_provision_if_needed(dev)

                # 2) αν ακόμη δεν έχει golden full, μην κάνεις INITIAL attest (θα βγει UNKNOWN)
                if not self.has_golden_full(dev, region="fw"):
                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "initial_full_attest_skipped_no_golden",
                        })
                else:
                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "initial_full_attest_start"
                        })
                    await self.attest_full_and_log(dev, trigger="INITIAL")

        while True:
            if dev not in self.devices:
                return

            decision = self.policy.tick(dev, now=time.time())

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

                        # 2) ML inference (ROP-aware) -> update policy with rich stats
                        if self.lr_policy is not None and windows:
                            labels_for_policy: List[PLLabel] = []
                            label_counts: Dict[str, int] = {}
                            ok_cnt = 0

                            weighted_scores: Dict[PLLabel, float] = {}
                            conf_values: List[float] = []
                            rop_values: List[float] = []

                            for w in windows:
                                pr = self.lr_policy.predict(dev, w)
                                if pr.get("ok") and pr.get("label") is not None:
                                    ok_cnt += 1
                                    pl = self._map_model_label_to_policy_label(pr["label"])
                                    labels_for_policy.append(pl)
                                    label_counts[pl.value] = label_counts.get(pl.value, 0) + 1

                                    wc = pr.get("model_conf")
                                    wgt = float(wc) if wc is not None else 1.0
                                    weighted_scores[pl] = weighted_scores.get(pl, 0.0) + wgt
                                    if wc is not None:
                                        conf_values.append(float(wc))

                                    rs = pr.get("rop_score")
                                    if rs is not None:
                                        rop_values.append(float(rs))

                            # weighted majority + weighted confidence
                            weighted_majority: Optional[PLLabel] = None
                            weighted_confidence: Optional[float] = None
                            if weighted_scores:
                                total_w = sum(weighted_scores.values())
                                wmaj, best_w = max(weighted_scores.items(), key=lambda kv: kv[1])
                                weighted_majority = wmaj
                                weighted_confidence = (best_w / total_w) if total_w > 0 else 0.0

                            model_conf_avg = (sum(conf_values) / len(conf_values)) if conf_values else None
                            model_conf_min = (min(conf_values)) if conf_values else None
                            model_conf_max = (max(conf_values)) if conf_values else None

                            rop_score_avg = (sum(rop_values) / len(rop_values)) if rop_values else None

                            # update policy with rich stats (ROP gating happens inside PolicyEngine)
                            summ = self.policy.on_inference_batch(
                                dev,
                                labels_for_policy,
                                now=time.time(),
                                weighted_majority=weighted_majority,
                                weighted_confidence=weighted_confidence,
                                model_conf_avg=model_conf_avg,
                                model_conf_min=model_conf_min,
                                model_conf_max=model_conf_max,
                                rop_score_avg=rop_score_avg
                            )

                            # refresh decision right away after label update
                            decision = self.policy.tick(dev, now=time.time())

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

                                    "weighted_majority": (weighted_majority.value if weighted_majority else None),
                                    "weighted_confidence": (round(float(weighted_confidence), 3) if weighted_confidence is not None else None),

                                    "model_conf_avg": (round(float(model_conf_avg), 3) if model_conf_avg is not None else None),
                                    "model_conf_min": (round(float(model_conf_min), 3) if model_conf_min is not None else None),
                                    "model_conf_max": (round(float(model_conf_max), 3) if model_conf_max is not None else None),

                                    "rop_score_avg": (round(float(rop_score_avg), 3) if rop_score_avg is not None else None),

                                    "policy_stable_label": self.policy.devices[dev].stable_label.value,
                                    "policy_reason": self.policy.devices[dev].last_reason,

                                    "window_id_range": {
                                        "from": windows[0].get("window_id"),
                                        "to": windows[-1].get("window_id"),
                                    },
                                })

            # 3) ATTEST if due (policy-driven)
            if decision.attest_kind != PLAttestKind.NONE:
                lk = self._attest_lock(dev)
                if lk.locked():
                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "policy_skip_attest_inflight",
                            "attest_kind": str(decision.attest_kind),
                            "reason": decision.reason
                        })
                else:
                    st = self.policy.devices.get(dev)
                    ml_meta = None
                    if st is not None:
                        ml_meta = {
                            "policy_stable_label": st.stable_label.value,
                            "policy_last_majority": st.last_majority.value,
                            "policy_vote_conf": round(float(st.last_confidence), 3),

                            "policy_weighted_majority": (st.last_weighted_majority.value if st.last_weighted_majority else None),
                            "policy_weighted_conf": (round(float(st.last_weighted_conf), 3) if st.last_weighted_conf is not None else None),

                            "policy_model_conf_avg": (round(float(st.last_model_conf_avg), 3) if st.last_model_conf_avg is not None else None),
                            "policy_model_conf_min": (round(float(st.last_model_conf_min), 3) if st.last_model_conf_min is not None else None),
                            "policy_model_conf_max": (round(float(st.last_model_conf_max), 3) if st.last_model_conf_max is not None else None),

                            "policy_rop_score_avg": (round(float(st.last_rop_score_avg), 3) if st.last_rop_score_avg is not None else None),

                            "policy_reason": st.last_reason,
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

        if device_id not in self.policy_tasks or self.policy_tasks[device_id].done():
            self.policy_tasks[device_id] = asyncio.create_task(self.policy_loop(device_id))
            print(f"[{now_s()}] [POLICY] started for {device_id}")

        try:
            await self.rx_loop(dc)
        finally:
            task = self.policy_tasks.get(device_id)
            if task and not task.done():
                task.cancel()
            self.policy_tasks.pop(device_id, None)

            self._close_files_for(device_id)

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
        print("  provision_golden")
        print("  force_provision_golden")
        print("  provision_blocks")
        print("  force_provision_blocks")
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
            elif cmd == "provision_golden":
                coro = self.provision_golden_full(dev, region="fw")
            elif cmd == "force_provision_golden":
                coro = self.force_provision_golden_full(dev, region="fw")
            elif cmd == "provision_blocks":
                coro = self.provision_golden_blocks(dev, force=False)
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
