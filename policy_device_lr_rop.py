# policy_device_lr_rop.py
import joblib
import numpy as np
from typing import Dict, Any, Optional


class DeviceLRPolicy:
    """
    Loads sklearn bundle and exposes:
      - predict(device_id, window_dict) -> {
            ok, label, model_conf, proba(dict), rop_score
        }
    Supports:
      bundle["models"][dev_int]  (per-device)
      or bundle["model"]         (single global model)
      or bundle itself is a Pipeline
    """

    def __init__(self, path: str):
        self.bundle = joblib.load(path)

        # features
        if isinstance(self.bundle, dict) and "features" in self.bundle:
            self.features = list(self.bundle["features"])
        else:
            raise ValueError("Model bundle must contain 'features' list.")

        # model(s)
        self.models = None
        self.model = None
        if isinstance(self.bundle, dict) and "models" in self.bundle:
            self.models = self.bundle["models"]  # device_id -> Pipeline
        elif isinstance(self.bundle, dict) and "model" in self.bundle:
            self.model = self.bundle["model"]    # single Pipeline
        else:
            # assume bundle itself is the pipeline
            self.model = self.bundle

    def _device_key(self, device_id: str) -> int:
        digits = ""
        for ch in reversed(device_id):
            if ch.isdigit():
                digits = ch + digits
            elif digits:
                break
        if not digits:
            raise ValueError(f"Cannot parse numeric id from device_id={device_id!r}")
        return int(digits)

    def _get_model(self, device_id: str):
        if self.models is not None:
            dev_int = self._device_key(device_id)
            m = self.models.get(dev_int)
            return m, dev_int
        return self.model, None

    @staticmethod
    def _to_float(x) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def predict(self, device_id: str, window: Dict[str, Any]) -> Dict[str, Any]:
        try:
            model, dev_int = self._get_model(device_id)
            if model is None:
                return {"ok": False, "reason": "no_model_loaded", "label": None}

            x = np.array([[self._to_float(window.get(f)) for f in self.features]], dtype=np.float32)

            # predict label
            pred = model.predict(x)[0]

            # probabilities (if classifier supports it)
            proba_dict: Optional[Dict[str, float]] = None
            model_conf: Optional[float] = None
            rop_score: Optional[float] = None

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]  # shape (C,)
                # get class names
                classes = None
                if hasattr(model, "classes_"):
                    classes = model.classes_
                elif hasattr(model, "named_steps"):
                    # pipeline: last step likely has classes_
                    for step in reversed(list(model.named_steps.values())):
                        if hasattr(step, "classes_"):
                            classes = step.classes_
                            break

                if classes is not None:
                    proba_dict = {str(cls): float(p) for cls, p in zip(classes, probs)}
                    model_conf = float(np.max(probs))

                    # rop_score = P(light_rop) + P(heavy_rop) if present
                    rop_score = float(proba_dict.get("light_rop", 0.0) + proba_dict.get("heavy_rop", 0.0))

            return {
                "ok": True,
                "reason": "ok",
                "label": str(pred),
                "device_int": dev_int,
                "model_conf": model_conf,
                "proba": proba_dict,
                "rop_score": rop_score,
            }

        except Exception as e:
            return {"ok": False, "reason": f"predict_error:{e}", "label": None}
