import joblib
import numpy as np
from typing import Dict, Any


class DeviceLRPolicy:
    """
    Loads your per-device sklearn pipeline bundle and exposes:
      - predict(device_id, window_dict) -> {"label": ..., "ok": bool, "reason": ...}
      - decide(label) -> {"action": "FULL"/"PARTIAL", "k": int, "reason": ...}
    """

    def __init__(self, path: str):
        self.bundle = joblib.load(path)
        self.features = list(self.bundle["features"])
        self.models = self.bundle["models"]  # device_id -> Pipeline

    def _device_key(self, device_id: str) -> int:
        """
        Your training used int device_id.
        Your runtime device_id is like "pico2w_1".
        We'll map "pico2w_1" -> 1.
        If you use other naming, adjust this.
        """
        # robust parse: take trailing digits
        digits = ""
        for ch in reversed(device_id):
            if ch.isdigit():
                digits = ch + digits
            elif digits:
                break
        if not digits:
            raise ValueError(f"Cannot parse numeric id from device_id={device_id!r}")
        return int(digits)

    def predict(self, device_id: str, window: Dict[str, Any]) -> Dict[str, Any]:
        try:
            dev_int = self._device_key(device_id)
            model = self.models.get(dev_int)
            if model is None:
                return {"ok": False, "reason": f"no_model_for_device:{dev_int}", "label": None}

            x = np.array([[float(window[f]) for f in self.features]], dtype=np.float32)
            label = model.predict(x)[0]
            return {"ok": True, "reason": "ok", "label": label, "device_int": dev_int}

        except Exception as e:
            return {"ok": False, "reason": f"predict_error:{e}", "label": None}

    def decide(self, label: Any) -> Dict[str, Any]:
        if label is None:
            return {"action": "FULL", "reason": "no_label_fallback_full"}

        try:
            # handle numpy scalars, strings, etc.
            if str(label) == "0":
                return {"action": "PARTIAL", "k": 3, "reason": "label0_partial3"}
            else:
                return {"action": "FULL", "reason": f"label{label}_full"}
        except Exception:
            return {"action": "FULL", "reason": "decide_error_fallback_full"}
