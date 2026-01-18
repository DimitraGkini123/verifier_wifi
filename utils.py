# ----------------- helpers file -----------------
import os
import json
import hashlib
import tempfile
from typing import Any
import time

def save_json_atomic(path: str, obj: dict):
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix="golden_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def jdump(obj: dict) -> bytes:
    return (json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")

def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def unhex(s: str) -> bytes:
    return bytes.fromhex(s)

def now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def ts_ms() -> int:
    return int(time.time() * 1000)

