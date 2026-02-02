# verifier_full_every_5s.py
import asyncio
import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

# ====== CONFIG ======
HOST = "0.0.0.0"
PORT = 4242

GOLDEN_PATH = "golden.json"
LOG_DIR = "logs_full5s"
ATTEST_PERIOD_S = 5.0
DURATION_S = 60.0  # <-- run exactly 1 minute per device

AUTO_PROVISION_ON_REGISTER = True
AUTO_PROVISION_DELAY_S = 0.2
DO_INITIAL_FULL_ATTEST = True

# ====== small utils ======
def ts_ms() -> int:
    return int(time.time() * 1000)

def jdump(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")

def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def save_json_atomic(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(path)

def unhex(s: str) -> bytes:
    return bytes.fromhex(s)

import hashlib
def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

# ====== connection state ======
@dataclass
class PendingReq:
    fut: asyncio.Future
    sent_msg: dict
    t0: float
    req_bytes: int

@dataclass
class DeviceConn:
    device_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    pending: Dict[str, PendingReq] = field(default_factory=dict)

    def is_alive(self) -> bool:
        return not self.writer.is_closing()

# ====== verifier ======
class VerifierFull5s:
    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.devices: Dict[str, DeviceConn] = {}
        self.golden = load_json(GOLDEN_PATH)

        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        self.logs_fp: Dict[str, Any] = {}   # per device jsonl fp

        self.attest_tasks: Dict[str, asyncio.Task] = {}
        self.lock: Dict[str, asyncio.Lock] = {}

        # keep per-device counters for summary
        self.stats: Dict[str, Dict[str, float]] = {}  # dev -> counters

    def _fp(self, dev: str):
        fp = self.logs_fp.get(dev)
        if fp:
            return fp
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = Path(LOG_DIR) / f"attest_full5s_{dev}_{stamp}.jsonl"
        fp = open(path, "a", encoding="utf-8", buffering=1)
        self.logs_fp[dev] = fp
        print(f"[LOG] {dev} -> {path}")
        return fp

    def _jwrite(self, dev: str, obj: dict):
        fp = self._fp(dev)
        fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        fp.flush()

    def _lk(self, dev: str) -> asyncio.Lock:
        lk = self.lock.get(dev)
        if lk is None:
            lk = asyncio.Lock()
            self.lock[dev] = lk
        return lk

    def _st(self, dev: str) -> Dict[str, float]:
        st = self.stats.get(dev)
        if st is None:
            st = {
                "full_count": 0.0,
                "sum_req_bytes": 0.0,
                "sum_resp_bytes": 0.0,
                "sum_rtt_ms": 0.0,
                "ok_count": 0.0,
                "fail_count": 0.0,
            }
            self.stats[dev] = st
        return st

    # ----- golden helpers -----
    def has_golden_full(self, dev: str, region: str = "fw") -> bool:
        try:
            _ = self.golden[dev][region]["sha256"]
            return True
        except Exception:
            return False

    def golden_full_hash(self, dev: str, region: str = "fw") -> Optional[bytes]:
        try:
            return unhex(self.golden[dev][region]["sha256"])
        except Exception:
            return None

    def set_golden_full_hash(self, dev: str, region: str, fw_hash_hex: str):
        self.golden.setdefault(dev, {})
        self.golden[dev].setdefault(region, {})
        self.golden[dev][region]["sha256"] = fw_hash_hex.lower()
        save_json_atomic(GOLDEN_PATH, self.golden)

    # ----- networking -----
    async def rx_loop(self, dc: DeviceConn):
        while True:
            line = await dc.reader.readline()
            if not line:
                return
            try:
                msg = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            req_id = msg.get("req_id")
            if req_id and req_id in dc.pending:
                pending = dc.pending.pop(req_id)
                resp = self.verify_if_needed(dc.device_id, pending.sent_msg, msg)

                # attach timing/size meta
                rtt_ms = (time.perf_counter() - pending.t0) * 1000.0
                resp_bytes = len(jdump(resp)) if isinstance(resp, dict) else 0
                if isinstance(resp, dict):
                    resp["_rtt_ms"] = round(rtt_ms, 2)
                    resp["_req_bytes"] = int(pending.req_bytes)
                    resp["_resp_bytes"] = int(resp_bytes)

                if not pending.fut.done():
                    pending.fut.set_result(resp)

    async def send_request(self, dev: str, msg: dict, timeout: float = 8.0) -> dict:
        dc = self.devices.get(dev)
        if not dc or not dc.is_alive():
            return {"type": "ERROR", "reason": "device_not_connected"}

        req_id = secrets.token_hex(8)
        msg = dict(msg)
        msg["req_id"] = req_id

        fut = self.loop.create_future()
        req_bytes = len(jdump({**msg, "req_id": "0000000000000000"}))
        dc.pending[req_id] = PendingReq(fut=fut, sent_msg=msg, t0=time.perf_counter(), req_bytes=req_bytes)

        dc.writer.write(jdump(msg))
        await dc.writer.drain()

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            dc.pending.pop(req_id, None)
            return {"type": "ERROR", "reason": "timeout_waiting_response", "req_id": req_id}

    # ----- protocol verification -----
    def verify_if_needed(self, dev: str, sent: dict, received: dict) -> dict:
        mode = sent.get("mode")
        rtype = received.get("type")

        if mode != "FULL_HASH_PROVER" or rtype != "ATTEST_RESPONSE":
            return received

        golden = self.golden_full_hash(dev, region=sent.get("region", "fw"))
        if golden is None:
            received["verify_ok"] = False
            received["verify_reason"] = "missing_golden_full_hash"
            return received

        nonce_hex = sent.get("nonce")
        resp_hex = received.get("response_hex")

        if not nonce_hex or not resp_hex:
            received["verify_ok"] = False
            received["verify_reason"] = "missing_fields"
            return received

        nonce = unhex(nonce_hex)
        expected = sha256(nonce + golden)
        got = unhex(resp_hex)

        ok = (got == expected)
        received["verify_ok"] = ok
        received["verify_reason"] = "nonce_bound_match" if ok else "nonce_bound_mismatch"
        return received

    # ----- provisioning -----
    async def provision_golden_full(self, dev: str, region: str = "fw") -> dict:
        if self.has_golden_full(dev, region):
            return {"type": "OK", "event": "golden_already_exists"}

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
        return {"type": "OK", "event": "golden_provisioned", "fw_hash_hex": fw_hex}

    # ----- one FULL + stats + log -----
    async def attest_full_once_and_log(self, dev: str, trigger: str):
        async with self._lk(dev):
            nonce = secrets.token_hex(8)
            resp = await self.send_request(dev, {
                "type": "ATTEST_REQUEST",
                "mode": "FULL_HASH_PROVER",
                "region": "fw",
                "nonce": nonce
            }, timeout=8.0)

            st = self._st(dev)
            st["full_count"] += 1.0
            st["sum_req_bytes"] += float(resp.get("_req_bytes") or 0)
            st["sum_resp_bytes"] += float(resp.get("_resp_bytes") or 0)
            st["sum_rtt_ms"] += float(resp.get("_rtt_ms") or 0)
            if resp.get("verify_ok") is True:
                st["ok_count"] += 1.0
            elif resp.get("verify_ok") is False:
                st["fail_count"] += 1.0

            self._jwrite(dev, {
                "ts_ms": ts_ms(),
                "device": dev,
                "event": "attest",
                "attest_kind": "FULL",
                "trigger": trigger,
                "k": None,
                "indices": None,
                "verify_ok": resp.get("verify_ok"),
                "verify_reason": resp.get("verify_reason", resp.get("reason")),
                "rtt_ms": resp.get("_rtt_ms"),
                "req_bytes": resp.get("_req_bytes"),
                "resp_bytes": resp.get("_resp_bytes"),
            })

    def _write_summary_and_close(self, dev: str):
        st = self._st(dev)
        n = max(1.0, st["full_count"])
        avg_rtt = st["sum_rtt_ms"] / n
        passes = st["full_count"]  # each FULL covers 100% fw -> one "memory pass"

        self._jwrite(dev, {
            "ts_ms": ts_ms(),
            "device": dev,
            "event": "summary_60s",
            "duration_s": DURATION_S,
            "attest_period_s": ATTEST_PERIOD_S,
            "full_count": int(st["full_count"]),
            "passes_full_memory": int(passes),
            "ok": int(st["ok_count"]),
            "fail": int(st["fail_count"]),
            "sum_req_bytes": int(st["sum_req_bytes"]),
            "sum_resp_bytes": int(st["sum_resp_bytes"]),
            "avg_rtt_ms": round(float(avg_rtt), 2),
        })

    # ----- attestation loop (EXACT 60s) -----
    async def full_hash_loop(self, dev: str):
        # optional provisioning + initial attest
        if AUTO_PROVISION_ON_REGISTER:
            await asyncio.sleep(AUTO_PROVISION_DELAY_S)
            prov = await self.provision_golden_full(dev, "fw")
            self._jwrite(dev, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision", "resp": prov})

        if DO_INITIAL_FULL_ATTEST and self.has_golden_full(dev, "fw"):
            await self.attest_full_once_and_log(dev, trigger="INITIAL")

        # run exactly DURATION_S from here
        t_start = time.monotonic()
        t_end = t_start + DURATION_S
        next_tick = time.monotonic()  # start immediately (after INITIAL)

        while dev in self.devices and time.monotonic() < t_end:
            # ensure golden exists (best-effort)
            if not self.has_golden_full(dev, "fw") and AUTO_PROVISION_ON_REGISTER:
                prov = await self.provision_golden_full(dev, "fw")
                self._jwrite(dev, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision_retry", "resp": prov})

            if self.has_golden_full(dev, "fw"):
                await self.attest_full_once_and_log(dev, trigger="FULL_EVERY_5S")

            # schedule next tick to avoid drift
            next_tick += ATTEST_PERIOD_S
            sleep_s = max(0.0, next_tick - time.monotonic())
            # but don't oversleep past the end
            remain = max(0.0, t_end - time.monotonic())
            await asyncio.sleep(min(sleep_s, remain))

        # summary + close this device connection
        self._write_summary_and_close(dev)

        dc = self.devices.get(dev)
        if dc and dc.is_alive():
            try:
                dc.writer.close()
                await dc.writer.wait_closed()
            except Exception:
                pass

    # ----- client handler -----
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        print(f"[+] Connection from {peer}")

        # Expect HELLO
        line = await reader.readline()
        if not line:
            writer.close()
            await writer.wait_closed()
            return
        try:
            hello = json.loads(line.decode("utf-8"))
        except Exception:
            writer.close()
            await writer.wait_closed()
            return
        if hello.get("type") != "HELLO" or "device_id" not in hello:
            writer.close()
            await writer.wait_closed()
            return

        dev = hello["device_id"]
        dc = DeviceConn(device_id=dev, reader=reader, writer=writer)
        self.devices[dev] = dc

        self._jwrite(dev, {"ts_ms": ts_ms(), "device": dev, "event": "device_registered"})

        rx_task = asyncio.create_task(self.rx_loop(dc))
        attest_task = asyncio.create_task(self.full_hash_loop(dev))
        self.attest_tasks[dev] = attest_task

        try:
            await rx_task
        finally:
            # cleanup
            attest_task.cancel()
            self.attest_tasks.pop(dev, None)
            self.devices.pop(dev, None)

            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

            fp = self.logs_fp.pop(dev, None)
            try:
                if fp:
                    fp.close()
            except Exception:
                pass

            print(f"[x] Disconnected {dev}")

            # If no devices left, stop server (single-run experiment)
            if not self.devices:
                self.loop.call_soon(self.loop.stop)

async def main():
    srv = VerifierFull5s()
    srv.loop = asyncio.get_running_loop()

    server = await asyncio.start_server(srv.handle_client, HOST, PORT)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"[Verifier] FULL every {ATTEST_PERIOD_S}s for {DURATION_S:.0f}s listening on {addrs}")

    async with server:
        # serve until srv.loop.stop() is called when last device finishes
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
