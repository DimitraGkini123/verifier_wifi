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

                # predicted label
                pred = model.predict(x)[0]

                # try probabilities
                proba = None
                model_conf = None
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(x)[0]  # shape (n_classes,)
                    proba = [float(v) for v in p.tolist()]
                    model_conf = float(np.max(p))   # max class probability

                return {
                    "ok": True,
                    "reason": "ok",
                    "label": pred,
                    "device_int": dev_int,
                    "proba": proba,           # list[float] or None
                    "model_conf": model_conf  # float in [0,1] or None
                }

            except Exception as e:
                return {"ok": False, "reason": f"predict_error:{e}", "label": None}