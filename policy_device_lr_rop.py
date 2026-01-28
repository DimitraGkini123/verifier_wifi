# policy_device_lr_rop.py
import joblib
import numpy as np
from typing import Dict, Any, Optional, Tuple, List


class DeviceLRPolicy:

    def __init__(self, path: str):
        self.bundle = joblib.load(path)

        self.features: List[str] = []
        self.class_names: Optional[List[str]] = None

        self.models = None           # optional multi-device: dev_int -> payload/pipeline
        self.model = None            # single model (LR)
        self.scaler = None           # StandardScaler for single payload
        self.payload_device_id = None

        self._init_from_bundle(self.bundle)

    # ----------------- init helpers -----------------
    def _init_from_bundle(self, bundle: Any) -> None:
        if isinstance(bundle, dict) and "features" in bundle and ("model" in bundle or "models" in bundle):
            self.features = list(bundle["features"])

            if "class_names" in bundle and isinstance(bundle["class_names"], (list, tuple)):
                self.class_names = [str(x).strip().lower() for x in bundle["class_names"]]

            if "models" in bundle:
                self.models = bundle["models"]
                return

            self.payload_device_id = self._safe_int(bundle.get("device_id", None))
            self.model = bundle.get("model", None)
            self.scaler = bundle.get("scaler", None)
            return

        if isinstance(bundle, dict) and "model" in bundle:
            self.model = bundle["model"]
            if "features" in bundle:
                self.features = list(bundle["features"])
            if "class_names" in bundle and isinstance(bundle["class_names"], (list, tuple)):
                self.class_names = [str(x).strip().lower() for x in bundle["class_names"]]
            return

        # pipeline saved directly (rare in your case)
        self.model = bundle
        if not self.features:
            raise ValueError(
                "Loaded model does not include 'features'. "
                "Save a payload dict with 'features' like your training script does."
            )

    @staticmethod
    def _safe_int(x) -> Optional[int]:
        try:
            return None if x is None else int(x)
        except Exception:
            return None

    def _device_key(self, device_id: str) -> int:
        # pico2w_1 -> 1
        digits = ""
        for ch in reversed(str(device_id)):
            if ch.isdigit():
                digits = ch + digits
            elif digits:
                break
        if not digits:
            raise ValueError(f"Cannot parse numeric id from device_id={device_id!r}")
        return int(digits)

    def _unwrap_model_obj(self, obj: Any) -> Tuple[Any, Any, Optional[List[str]]]:
        # obj can be payload dict or pipeline
        if isinstance(obj, dict):
            m = obj.get("model", None)
            sc = obj.get("scaler", None)
            cn = None
            if "class_names" in obj and isinstance(obj["class_names"], (list, tuple)):
                cn = [str(x).strip().lower() for x in obj["class_names"]]
            return m, sc, cn
        return obj, None, None

    def _get_model_for_device(self, device_id: str):
        if self.models is not None:
            dev_int = self._device_key(device_id)
            mobj = self.models.get(dev_int)
            if mobj is None:
                mobj = self.models.get(str(dev_int))
            if mobj is None:
                for k, v in self.models.items():
                    if str(k) == str(dev_int):
                        mobj = v
                        break
            if mobj is None:
                return None, None, None, dev_int

            model, scaler, cn = self._unwrap_model_obj(mobj)
            cn = cn or self.class_names
            return model, scaler, cn, dev_int

        # single payload
        dev_int = None
        try:
            dev_int = self._device_key(device_id)
        except Exception:
            pass
        return self.model, self.scaler, self.class_names, dev_int

    # ----------------- feature helpers -----------------
    @staticmethod
    def _to_float(x) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    @staticmethod
    def _safe_div(num: float, den: float, eps: float = 1e-9) -> float:
        if not np.isfinite(num) or not np.isfinite(den):
            return float("nan")
        if abs(den) < eps:
            return 0.0
        return float(num / den)

    def _enrich_window(self, w: Dict[str, Any]) -> Dict[str, Any]:
        """
        If ratios are missing, derive them from deltas:
          cyc_per_us   = dC / dT
          lsu_per_cyc  = dL / dC
          cpi_per_cyc  = dP / dC
          exc_per_cyc  = dE / dC
          fold_per_cyc = dF / dC

        This matches your training FEATURES.
        """
        ww = dict(w)

        dC = self._to_float(ww.get("dC"))
        dT = self._to_float(ww.get("dT"))
        dL = self._to_float(ww.get("dL"))
        dP = self._to_float(ww.get("dP"))
        dE = self._to_float(ww.get("dE"))
        dF = self._to_float(ww.get("dF"))

        # Only compute if missing
        if "cyc_per_us" not in ww:
            ww["cyc_per_us"] = self._safe_div(dC, dT)

        if "lsu_per_cyc" not in ww:
            ww["lsu_per_cyc"] = self._safe_div(dL, dC)

        if "cpi_per_cyc" not in ww:
            ww["cpi_per_cyc"] = self._safe_div(dP, dC)

        if "exc_per_cyc" not in ww:
            ww["exc_per_cyc"] = self._safe_div(dE, dC)

        if "fold_per_cyc" not in ww:
            ww["fold_per_cyc"] = self._safe_div(dF, dC)

        return ww

    def _vectorize(self, window: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Strict vectorization:
          - Must have all features after enrichment
          - Must not contain NaN
        """
        w = self._enrich_window(window)

        missing = []
        vals = []
        for f in self.features:
            if f not in w:
                missing.append(f)
                vals.append(np.nan)
            else:
                vals.append(self._to_float(w.get(f)))

        if missing:
            head = missing[:6]
            more = max(0, len(missing) - len(head))
            return None, f"missing_features:{head}" + (f"(+{more})" if more else "")

        x = np.array([vals], dtype=np.float32)
        if np.isnan(x).any() or not np.isfinite(x).all():
            return None, "nan_or_inf_in_features"
        return x, None

    # ----------------- prediction -----------------
    def predict(self, device_id: str, window: Dict[str, Any]) -> Dict[str, Any]:
        try:
            model, scaler, cn, dev_int = self._get_model_for_device(device_id)
            if model is None:
                return {"ok": False, "reason": f"no_model_for_device:{dev_int}", "label": None, "device_int": dev_int}

            x, err = self._vectorize(window)
            if err is not None:
                return {"ok": False, "reason": err, "label": None, "device_int": dev_int}

            x_in = scaler.transform(x) if scaler is not None else x

            pred = model.predict(x_in)[0]

            # Your LR predicts int class 0..4 -> map to class_names if provided
            label_str = str(pred).strip().lower()
            if cn is not None:
                try:
                    label_str = cn[int(pred)]
                except Exception:
                    pass

            proba_dict: Optional[Dict[str, float]] = None
            model_conf: Optional[float] = None
            rop_score: Optional[float] = None

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x_in)[0]
                classes = getattr(model, "classes_", None)

                if classes is not None:
                    proba_dict = {}
                    for cls, p in zip(classes, probs):
                        key = str(cls).strip().lower()
                        if cn is not None:
                            try:
                                key = cn[int(cls)]
                            except Exception:
                                pass
                        proba_dict[str(key).strip().lower()] = float(p)

                    model_conf = float(np.max(probs))
                    rop_score = float(proba_dict.get("light_rop", 0.0) + proba_dict.get("heavy_rop", 0.0))

            return {
                "ok": True,
                "reason": "ok",
                "label": label_str,
                "device_int": dev_int,
                "model_conf": model_conf,
                "proba": proba_dict,
                "rop_score": rop_score,
            }

        except Exception as e:
            return {"ok": False, "reason": f"predict_error:{e}", "label": None}
