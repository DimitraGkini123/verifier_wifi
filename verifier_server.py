
import asyncio
import json
import secrets
import hashlib
import threading
import time
import random
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from utils import save_json_atomic, jdump, sha256, ts_ms, now_s, unhex
from utils import now_s, ts_ms
from lru_blocks import DeviceLRUBlocks

from policy_device_lr import DeviceLRPolicy

LRU_STATE_PATH = "lru_state.json"


HOST = "0.0.0.0"
PORT = 4242
GOLDEN_PATH = "golden.json"
TRUST_UNKNOWN = "UNKNOWN"
TRUST_TRUSTED = "TRUSTED"
TRUST_UNTRUSTED = "UNTRUSTED"

# ---------------- Auto collection config ----------------
# ---------------- Periodic attestation config ----------------
AUTO_ATTEST = True
FULL_ATTEST_PERIOD_S = 60
PARTIAL_ATTEST_PERIOD_S = 10
PARTIAL_K = [1,2,3,4,5,6,7,8,9]
ATTEST_JITTER_S = 1.0

AUTO_COLLECT = True
COLLECT_PERIOD_MS = 600        # poll every 500ms
COLLECT_MAX_WINDOWS = 6     # 500ms -> 5 windows (100ms each)
MODEL_PATH = "models/verifier_models_per_device.joblib"
ML_ENABLE = True
ML_TRIGGER_ATTEST = True   # αν θες να στέλνει attestation βάσει inference
ML_COOLDOWN_S = 10.0       # για να μη σπαμάρει

# ----------------- device session -----------------
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


class VerifierServer:
        #creates log for attestations 
    def log_attest_event(self, dev: str, kind: str, k: int | None, indices: list[int] | None,
                     resp: dict, trust_before: str, trust_after: str,
                     trigger: str = "UNKNOWN", ml: dict | None = None):

        fp = self.attest_fp.get(dev)
        if not fp:
            return
        self._jwrite(fp, {
            "ts_ms": ts_ms(),
            "device": dev,
            "event": "attest",
            "attest_kind": kind,
            "trigger": trigger,          # <-- NEW
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
            "compute_us_total": resp.get("compute_us_total"),   # από prover (θα το βάλεις)
        })


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
        self.golden[device_id][region]["sha256"] = fw_hash_hex.lower()
        save_json_atomic(GOLDEN_PATH, self.golden)

    async def provision_golden_full(self, dev: str, region: str = "fw") -> dict:
        # guard: don't overwrite existing golden
        if self.has_golden_full(dev, region):
            return {
                "type": "ERROR",
                "reason": "golden_already_exists_refusing_overwrite",
                "device": dev,
                "region": region,
                "existing_sha256": self.golden.get(dev, {}).get(region, {}).get("sha256")
            }

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
        return {"type": "OK", "event": "golden_provisioned", "device": dev, "region": region, "fw_hash_hex": fw_hex}
    
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

    #for partial hash 
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

    async def send_request_timed(self, device_id: str, msg: dict, timeout: float = 5.0) -> dict:
        # bytes of request line (approx exact, because jdump is exact line we send)
        req_line = jdump({**msg, "req_id": "0000000000000000"})  # placeholder
        req_bytes = len(req_line)

        t0 = time.perf_counter()
        resp = await self.send_request(device_id, msg, timeout=timeout)
        rtt_ms = (time.perf_counter() - t0) * 1000.0

        # response bytes: we don't have raw line here; approximate with json dump
        resp_bytes = len(jdump(resp)) if isinstance(resp, dict) else 0

        if isinstance(resp, dict):
            resp["_rtt_ms"] = round(rtt_ms, 2)
            resp["_req_bytes"] = int(req_bytes)
            resp["_resp_bytes"] = int(resp_bytes)
        return resp
    
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
            # if block count changed, reset
            lru.ensure_size(bc)
        return lru


    
    def __init__(self, golden_db: dict):
        #lru for partial
                # LRU per device for partial blocks
        self.block_lru: Dict[str, DeviceLRUBlocks] = {}

        # load persisted LRU state if exists
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
            
        self.golden = golden_db
        self.devices: Dict[str, DeviceConn] = {}
        self.selected_device: Optional[str] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # per-device window cursor
        self.last_seen: Dict[str, int] = {}

        # auto-collector
        self.collect_tasks: Dict[str, asyncio.Task] = {}
        self.windows_fp: Dict[str, Any] = {}
        self.events_fp: Dict[str, Any] = {}

        
     # auto-attestation
        self.attest_tasks: Dict[str, asyncio.Task] = {}
        self.attest_fp: Dict[str, Any] = {}
        self.partial_k_cursor: Dict[str, int] = {}
    ##model 
        self.policy = DeviceLRPolicy(MODEL_PATH) if ML_ENABLE else None
        self.last_ml_trigger_ts: Dict[str, float] = {}

        self.attest_locks: Dict[str, asyncio.Lock] = {}
#helper 
    def _attest_lock(self, dev: str) -> asyncio.Lock:
        lk = self.attest_locks.get(dev)
        if lk is None:
            lk = asyncio.Lock()
            self.attest_locks[dev] = lk
        return lk


    # -------- golden access --------
    def golden_full_hash(self, device_id: str, region: str = "fw") -> Optional[bytes]:
        try:
            return unhex(self.golden[device_id][region]["sha256"])
        except Exception:
            return None

    def golden_block_hash(self, device_id: str, index: int) -> Optional[bytes]:
        try:
            return unhex(self.golden[device_id]["blocks"]["hashes"][index])
        except Exception:
            return None

    # -------- JSONL file helpers --------
    def _open_files_for(self, dev: str):
        if dev in self.windows_fp:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        wpath = f"windows_{dev}_{stamp}.jsonl"
        epath = f"events_{dev}_{stamp}.jsonl"
        apath = apath = f"attest_{dev}_{stamp}.jsonl"
        self.windows_fp[dev] = open(wpath, "a", encoding="utf-8", buffering=1)
        self.events_fp[dev] = open(epath, "a", encoding="utf-8", buffering=1)
        self.attest_fp[dev] = open(apath, "a", encoding="utf-8", buffering=1)
        print(f"[{now_s()}] [COLLECT] files for {dev}:")
        print(f"  windows -> {wpath}")
        print(f"  events  -> {epath}")
        print(f"  attest  -> {apath}")

    @staticmethod
    def _jwrite(fp, obj: dict):
        fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        fp.flush()

    def _close_files_for(self, dev: str):
        fpw = self.windows_fp.pop(dev, None)
        fpe = self.events_fp.pop(dev, None)
        fpa = self.attest_fp.pop(dev, None)
        try:
            if fpw:
                fpw.close()
            if fpe:
                fpe.close()
            if fpa:
                fpa.close()
        except Exception:
            pass

    def _save_lru_state(self):
        try:
            blob = {dev: lru.export_state() for dev, lru in self.block_lru.items()}
            save_json_atomic(LRU_STATE_PATH, blob)
        except Exception:
            pass


    # -------- networking --------
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        print(f"[{now_s()}] [+] Connection from {peer}")

        # Expect HELLO first
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

        print(f"[{now_s()}] [=] Registered device_id={device_id}. Selected={self.selected_device}")
        # Μετά το print "Registered ..."
        asyncio.create_task(self.initial_attest(device_id))

        # start auto-collector
        if AUTO_COLLECT:
            if device_id not in self.collect_tasks or self.collect_tasks[device_id].done():
                self._open_files_for(device_id)
                self.collect_tasks[device_id] = asyncio.create_task(
                    self.collector_loop(device_id, COLLECT_PERIOD_MS, COLLECT_MAX_WINDOWS)
                )
                print(f"[{now_s()}] [COLLECT] started for {device_id} every {COLLECT_PERIOD_MS}ms, max={COLLECT_MAX_WINDOWS}")
        
        if AUTO_ATTEST:
            # σιγουρέψου ότι έχουν ανοιχτεί files
            self._open_files_for(device_id)

            if device_id not in self.attest_tasks or self.attest_tasks[device_id].done():
                self.attest_tasks[device_id] = asyncio.create_task(self.periodic_attest_loop(device_id))
                print(
                    f"[{now_s()}] [ATTEST] periodic started for {device_id} "
                    f"(partial={PARTIAL_ATTEST_PERIOD_S}s k={PARTIAL_K}, full={FULL_ATTEST_PERIOD_S}s)"
                )

        # Start RX loop
        try:
            await self.rx_loop(dc)
        finally:
            # stop collector on disconnect
            task = self.collect_tasks.get(device_id)
            if task and not task.done():
                task.cancel()
            self.collect_tasks.pop(device_id, None)

            # close files
            self._close_files_for(device_id)

            # cleanup registry on disconnect
            if self.devices.get(device_id) is dc:
                del self.devices[device_id]
                if self.selected_device == device_id:
                    self.selected_device = next(iter(self.devices), None)
                        # stop attestation task on disconnect
            at = self.attest_tasks.get(device_id)
            if at and not at.done():
                at.cancel()
            self.attest_tasks.pop(device_id, None)
            self.partial_k_cursor.pop(device_id, None)


            writer.close()
            await writer.wait_closed()
            print(f"[{now_s()}] [x] Disconnected device_id={device_id}")

    async def initial_attest(self, dev: str):
    # μικρό delay για να έχει ξεκινήσει καλά το RX loop / δικτυακή στοίβα
        await asyncio.sleep(0.5)
        if dev not in self.devices:
            return

        print(f"[{now_s()}] [ATTEST] initial full attestation for {dev} ...")
        resp = await self.attest_full_with_retry(dev)

        # log σε events file (αν είναι ανοιχτό)
        fp = self.events_fp.get(dev)
        if fp:
            self._jwrite(fp, {
                "ts_ms": ts_ms(),
                "device": dev,
                "event": "initial_attest",
                "trust_state": self.devices[dev].trust_state if dev in self.devices else TRUST_UNKNOWN,
                "resp": resp
            })

    
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

            # Match to pending request
            req_id = msg.get("req_id")
            if req_id and req_id in dc.pending:
                pending = dc.pending.pop(req_id)
                verified_msg = self.verify_if_needed(dc.device_id, pending.sent_msg, msg)
                if not pending.fut.done():
                    pending.fut.set_result(verified_msg)
            else:
                print(f"[{now_s()}] [RX] {dc.device_id}: {msg}")

    def verify_if_needed(self, device_id: str, sent: dict, received: dict) -> dict:
        mode = sent.get("mode")
        rtype = received.get("type")

        if rtype not in ("ATTEST_RESPONSE", "METRICS", "PONG", "WINDOWS"):
            return received

        if mode == "FULL_HASH_PROVER" and rtype == "ATTEST_RESPONSE":
            golden = self.golden_full_hash(device_id, region=sent.get("region", "fw"))
            if golden is None:
                received["verify_ok"] = False
                received["verify_reason"] = "missing_golden_full_hash"
                return received

            nonce_hex = sent.get("nonce")
            if "hash_hex" in received:
                got = unhex(received["hash_hex"])
                ok = (got == golden)
                received["verify_ok"] = ok
                received["verify_reason"] = "direct_hash_match" if ok else "direct_hash_mismatch"
                return received

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

        if mode == "PARTIAL_BLOCKS" and rtype == "ATTEST_RESPONSE":
            if not self.has_golden_blocks(device_id):
                received["verify_ok"] = False
                received["verify_reason"] = "missing_golden_blocks"
                return received

            nonce_hex = sent.get("nonce")  # <- NEW
            nonce = unhex(nonce_hex) if nonce_hex else None

            blocks = received.get("blocks", [])
            all_ok = True
            reasons = []

            for b in blocks:
                idx = b.get("index")
                if idx is None:
                    all_ok = False
                    reasons.append("block_missing_index")
                    continue

                golden_b = self.golden_block_hash(device_id, idx)
                if golden_b is None:
                    all_ok = False
                    reasons.append(f"missing_golden_block_{idx}")
                    continue

                ok = False

                # NEW: nonce-bound verification
                if nonce is not None and "response_hex" in b:
                    expected = sha256(nonce + golden_b)
                    got = unhex(b["response_hex"])
                    ok = (got == expected)

                # fallback: direct hash compare (useful for provisioning/debug)
                elif "hash_hex" in b:
                    ok = (unhex(b["hash_hex"]) == golden_b)

                elif "data_hex" in b:
                    ok = (sha256(unhex(b["data_hex"])) == golden_b)

                if not ok:
                    all_ok = False
                    reasons.append(f"block_{idx}_mismatch")

            received["verify_ok"] = all_ok
            received["verify_reason"] = "ok" if all_ok else ",".join(reasons)
            return received


        return received

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

    # -------- auto collector --------
    async def collector_loop(self, dev: str, period_ms: int, max_windows: int):
        self._open_files_for(dev)

        while True:
            try:
                # --- NEW: μην κάνεις GET_WINDOWS όταν τρέχει attestation ---
                lk = self._attest_lock(dev)
                if lk.locked():
                    fp_evt = self.events_fp.get(dev)
                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "collector_skip_attest_inflight",
                        })
                    await asyncio.sleep(period_ms / 1000.0)
                    continue

                since = int(self.last_seen.get(dev, 0))

                t0 = time.perf_counter()
                resp = await self.send_request(dev, {
                    "type": "GET_WINDOWS",
                    "since": since,
                    "max": int(max_windows),
                }, timeout=8.0)
                rtt_ms = int((time.perf_counter() - t0) * 1000)

                # πάντα γράψε event για να ξέρεις ότι ο loop ζει
                fp_evt = self.events_fp.get(dev)
                if fp_evt:
                    self._jwrite(fp_evt, {
                        "ts_ms": ts_ms(),
                        "device": dev,
                        "event": "collector_tick",
                        "since": since,
                        "rtt_ms": rtt_ms,
                        "resp_type": resp.get("type"),
                    })

                if resp.get("type") != "WINDOWS":
                    # log error response
                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "collector_error",
                            "since": since,
                            "rtt_ms": rtt_ms,
                            "resp": resp,
                        })
                    await asyncio.sleep(period_ms / 1000.0)
                    continue

                windows = resp.get("windows", []) or []
                cnt = int(resp.get("count", len(windows)) or 0)

                # --- update cursor robustly ---
                # 1) prefer resp["to"]
                to_id = resp.get("to", None)
                if to_id is not None:
                    try:
                        to_id = int(to_id)
                    except Exception:
                        to_id = None

                # 2) if "to" missing/invalid, compute from last window
                if (to_id is None) and windows:
                    # assumes each window dict has "window_id"
                    try:
                        to_id = int(windows[-1].get("window_id"))
                    except Exception:
                        to_id = None

                # IMPORTANT: make it "next since" (avoid refetching same window)
                if to_id is not None:
                    self.last_seen[dev] = to_id + 1

                # log summary
                if fp_evt:
                    self._jwrite(fp_evt, {
                        "ts_ms": ts_ms(),
                        "device": dev,
                        "event": "windows_ok",
                        "since": since,
                        "from": resp.get("from"),
                        "to": resp.get("to"),
                        "cursor_set_to": self.last_seen.get(dev),
                        "count": cnt,
                        "dropped_old": resp.get("dropped_old"),
                        "dropped_overflow": resp.get("dropped_overflow"),
                        "rtt_ms": rtt_ms,
                    })

                # write window lines
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
# ---------------- ML inference (ALL windows) + majority vote + optional attestation trigger ----------------
                if self.policy is not None and windows:
                    labels = []
                    per_window = []  # για debug/logging
                    ok_cnt = 0

                    for w in windows:
                        pr = self.policy.predict(dev, w)
                        per_window.append({
                            "window_id": w.get("window_id"),
                            "ok": pr.get("ok"),
                            "reason": pr.get("reason"),
                            "label": None if pr.get("label") is None else str(pr.get("label")),
                        })
                        if pr.get("ok") and pr.get("label") is not None:
                            ok_cnt += 1
                            labels.append(str(pr.get("label")))

                    majority_label = None
                    label_counts = {}
                    if labels:
                        for lb in labels:
                            label_counts[lb] = label_counts.get(lb, 0) + 1
                        # majority
                        majority_label = max(label_counts.items(), key=lambda kv: kv[1])[0]

                    decision = self.policy.decide(majority_label)

                    # optional: compute a simple "confidence" = majority fraction
                    total = sum(label_counts.values()) if label_counts else 0
                    majority_frac = (label_counts.get(majority_label, 0) / total) if total > 0 else 0.0

                    # log inference event (batch)
                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "ml_inference_batch",
                            "n_windows": len(windows),
                            "n_ok": ok_cnt,
                            "label_counts": label_counts,
                            "majority_label": majority_label,
                            "majority_frac": round(majority_frac, 3),
                            "decision": decision,
                            "window_id_range": {
                                "from": windows[0].get("window_id"),
                                "to": windows[-1].get("window_id"),
                            },
                            # Αν το θες, κράτα και per-window λεπτομέρεια (μπορεί να φουσκώσει το αρχείο)
                            # "per_window": per_window,
                        })

                    # optional: trigger attestation (cooldown + lock + majority)
                    if ML_TRIGGER_ATTEST:
                        # Μην ξεκινάς νέο attestation αν τρέχει ήδη (lock)
                        lk = self._attest_lock(dev)
                        if lk.locked():
                            if fp_evt:
                                self._jwrite(fp_evt, {
                                    "ts_ms": ts_ms(),
                                    "device": dev,
                                    "event": "ml_trigger_skip_inflight",
                                    "majority_label": majority_label,
                                    "decision": decision,
                                })
                        else:
                            now_t = time.time()
                            last_t = self.last_ml_trigger_ts.get(dev, 0.0)

                            if now_t - last_t >= ML_COOLDOWN_S:
                                self.last_ml_trigger_ts[dev] = now_t

                                ml_meta = {
                                    "majority_label": majority_label,
                                    "label_counts": label_counts,
                                    "majority_frac": round(majority_frac, 3),
                                    "n_windows": len(windows),
                                    "n_ok": ok_cnt,
                                    "decision": decision,
                                    "window_id_from": windows[0].get("window_id"),
                                    "window_id_to": windows[-1].get("window_id"),
                                }

                                if decision.get("action") == "FULL":
                                    asyncio.create_task(self.attest_full_and_log(dev, trigger="ML_MAJORITY", ml=ml_meta))
                                elif decision.get("action") == "PARTIAL":
                                    k = int(decision.get("k", 3))
                                    asyncio.create_task(self.attest_partial_and_log(dev, k=k, trigger="ML_MAJORITY", ml=ml_meta))

            except Exception as e:
                # αν ο loop σκάσει, μην πεθάνει. Γράψε το.
                fp_evt = self.events_fp.get(dev)
                if fp_evt:
                    self._jwrite(fp_evt, {
                        "ts_ms": ts_ms(),
                        "device": dev,
                        "event": "collector_exception",
                        "err": str(e),
                    })

            await asyncio.sleep(period_ms / 1000.0)



    # -------- CLI (runs in thread) --------
    def cli_thread(self):
        print("\nCLI commands:")
        print("  list")
        print("  use <device_id>")
        print("  ping")
        print("  windows <N>          (manual fetch: up to N new windows since last fetch)")
        print("  attest_full")
        print("  attest_partial <k>")
        print("  provision_golden     (fetch fw_hash_hex and store to golden.json)")
        print("  force_provision_golden  (overwrite golden!)")
        print("  provision_blocks      (fetch all block hashes and store to golden.json)")
        print("  force_provision_blocks (overwrite blocks!)")
        print("  quit\n")

        while True:
            try:
                cmd = input("verifier> ").strip()
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
            elif cmd.startswith("windows"):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 20
                since = self.last_seen.get(dev, 0)
                coro = self.send_request(dev, {"type": "GET_WINDOWS", "since": since, "max": n}, timeout=8.0)
            elif cmd == "attest_full":
                coro = self.attest_full_with_retry(dev)
            elif cmd.startswith("attest_partial"):
                parts = cmd.split()
                k = int(parts[1]) if len(parts) > 1 else 3

                coro = self.attest_partial_and_log(dev, k=k, trigger="CLI")

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
                resp = fut.result(timeout=12.0)
                if resp.get("type") == "WINDOWS":
                    to_id = resp.get("to", 0)
                    if to_id:
                        self.last_seen[dev] = int(to_id) + 1
                print(f"[{now_s()}] [RESP] {resp}")
            except Exception as e:
                print("error waiting:", e)

    async def attest_full_once(self, dev: str, timeout: float = 8.0) -> dict:
        nonce = secrets.token_hex(8)
        resp = await self.send_request_timed(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "FULL_HASH_PROVER",
            "region": "fw",
            "nonce": nonce
        }, timeout=timeout)
        return resp

    def _update_trust_from_attest(self, dev: str, resp: dict, attempt: int):
        dc = self.devices.get(dev)
        if not dc:
            return

        reason = resp.get("verify_reason", resp.get("reason", "unknown"))
        if reason == "missing_golden_full_hash":
            dc.trust_state = TRUST_UNKNOWN
            dc.attest_fail_streak = 0
            print(f"[{now_s()}] [ATTEST] {dev}: no golden yet -> TRUST_UNKNOWN (not a fail)")
            return

        ok = bool(resp.get("verify_ok", False))

        if reason in ("missing_golden_blocks", "no_golden_blocks"):
            dc.trust_state = TRUST_UNKNOWN
            dc.attest_fail_streak = 0
            print(f"[{now_s()}] [ATTEST] {dev}: no golden blocks yet -> TRUST_UNKNOWN (not a fail)")
            return


        if ok:
            dc.trust_state = TRUST_TRUSTED
            dc.attest_fail_streak = 0
            dc.last_attest_ok_ts = time.time()
            print(f"[{now_s()}] [ATTEST] {dev} OK (attempt={attempt}) reason={reason}")
        else:
            dc.attest_fail_streak += 1
            dc.last_attest_fail_ts = time.time()
            print(f"[{now_s()}] [ATTEST] {dev} FAIL (attempt={attempt}) reason={reason}")
            if dc.attest_fail_streak >= 2:
                dc.trust_state = TRUST_UNTRUSTED
                print(f"[{now_s()}] [TRUST] {dev} => UNTRUSTED (2 consecutive fails)")


    async def attest_full_with_retry(self, dev: str) -> dict:
        # 1η προσπάθεια
        resp1 = await self.attest_full_once(dev, timeout=8.0)
        self._update_trust_from_attest(dev, resp1, attempt=1)

        if resp1.get("verify_ok", False):
            return resp1

        # Retry 1 φορά (μικρή καθυστέρηση για να αποφύγεις transient glitches)
        await asyncio.sleep(1.0)
        resp2 = await self.attest_full_once(dev, timeout=8.0)
        self._update_trust_from_attest(dev, resp2, attempt=2)
        return resp2
    
    async def attest_full_and_log(self, dev: str, trigger: str = "UNKNOWN", ml: dict | None = None) -> dict:
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
        #indices = sorted(random.sample(range(bc), k))
        lru = self._get_block_lru(dev)
        if lru is None:
            return {"type": "ERROR", "reason": "no_golden_blocks"}

        indices = lru.pick(k)
        # προαιρετικά: keep sorted for nicer logs / deterministic order on wire
        indices = sorted(indices)

        nonce = secrets.token_hex(8)

        resp = await self.send_request_timed(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "PARTIAL_BLOCKS",
            "region": "fw",
            "nonce": nonce,
            "indices": indices
        }, timeout=timeout)

        # βάλε για logging (εύκολο!)
        if isinstance(resp, dict):
            resp["_k"] = k
            resp["_indices"] = indices

        return resp

    async def attest_partial_and_log(self, dev: str, k: int, trigger: str = "UNKNOWN", ml: dict | None = None) -> dict:
        async with self._attest_lock(dev):
            dc = self.devices.get(dev)
            if not dc:
                return {"type": "ERROR", "reason": "device_not_connected"}

            trust_before = dc.trust_state
            resp = await self.attest_partial_once(dev, k=k, timeout=12.0)
            self._update_trust_from_attest(dev, resp if isinstance(resp, dict) else {}, attempt=1)
            #update lru 
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

    
    def pick_next_partial_k(self, dev: str) -> int:
        if not PARTIAL_K:
            return 1
        cur = self.partial_k_cursor.get(dev, 0)
        k = PARTIAL_K[cur % len(PARTIAL_K)]
        self.partial_k_cursor[dev] = cur + 1
        return int(k)

    # i keep the full attestation automatic 
    async def periodic_attest_loop(self, dev: str):
        await asyncio.sleep(1.0)

        next_full = time.time() + FULL_ATTEST_PERIOD_S
        next_partial = time.time() + PARTIAL_ATTEST_PERIOD_S

        while True:
            if dev not in self.devices:
                return

            now = time.time()

            # PARTIAL (πιο συχνό)
            #uncomment to enable auto partial attestation 
            #if now >= next_partial:
            #    await asyncio.sleep(random.random() * ATTEST_JITTER_S)
            #    try:
            #        k = self.pick_next_partial_k(dev)
            #        await self.attest_partial_and_log(dev, k=k)
            #    except Exception as e:
            #        fp = self.events_fp.get(dev)
            #        if fp:
            #            self._jwrite(fp, {
            #                "ts_ms": ts_ms(),
            #                "device": dev,
            #                "event": "attest_partial_error",
            #                "err": str(e)
            #            })
            #    next_partial = time.time() + PARTIAL_ATTEST_PERIOD_S

            # FULL (πιο αραιό)
            if now >= next_full:
                await asyncio.sleep(random.random() * ATTEST_JITTER_S)
                try:
                    await self.attest_full_and_log(dev,trigger="PERIODIC" )
                except Exception as e:
                    fp = self.events_fp.get(dev)
                    if fp:
                        self._jwrite(fp, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "attest_full_error",
                            "err": str(e)
                        })
                next_full = time.time() + FULL_ATTEST_PERIOD_S

            await asyncio.sleep(0.2)





async def main():
    try:
        with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
            golden = json.load(f)
    except FileNotFoundError:
        golden = {}


    srv = VerifierServer(golden)
    srv.loop = asyncio.get_running_loop()

    # start CLI in a background thread
    t = threading.Thread(target=srv.cli_thread, daemon=True)
    t.start()

    server = await asyncio.start_server(srv.handle_client, HOST, PORT)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"[{now_s()}] Verifier listening on {addrs}")
    print(f"[{now_s()}] AUTO_COLLECT={AUTO_COLLECT} period_ms={COLLECT_PERIOD_MS} max_windows={COLLECT_MAX_WINDOWS}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
