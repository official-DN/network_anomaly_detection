"""
Detection service: loads ML models and runs inference.
Handles Random Forest, XGBoost, Isolation Forest, and Autoencoder.
"""

import joblib
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from backend.config import settings
from backend.schemas.prediction import PredictionResult, BatchPredictionResult
from backend.schemas.traffic import TrafficRecord
from backend.utils.logger import logger

# Feature columns expected by trained models
FEATURE_COLUMNS = [
    "src_port", "dst_port", "protocol_num", "packets", "bytes", "duration",
    "packets_per_sec", "bytes_per_sec", "avg_packet_size", "syn_count",
    "ack_count", "fin_count", "rst_count", "fwd_packets", "bwd_packets",
]

PROTOCOL_MAP = {"TCP": 6, "UDP": 17, "ICMP": 1, "HTTP": 80, "DNS": 53, "TLS": 443}

RISK_THRESHOLDS = {
    "LOW": (0, 30),
    "MEDIUM": (30, 60),
    "HIGH": (60, 80),
    "CRITICAL": (80, 101),
}


class DetectionService:
    """Loads trained models and performs anomaly detection."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.preprocessors: Dict[str, Any] = {}
        self.model_path = settings.model_path
        self._load_models()

    def _load_models(self) -> None:
        """Attempt to load all trained models from disk."""
        model_loaders = {
            "random_forest": self._load_sklearn_model("random_forest"),
            "xgboost": self._load_sklearn_model("xgboost"),
            "isolation_forest": self._load_sklearn_model("isolation_forest"),
        }
        for name, model in model_loaders.items():
            if model:
                self.models[name] = model
                logger.info(f"Loaded model: {name}")
            else:
                logger.warning(f"Model not found: {name} — will use fallback scoring")

        # Try loading autoencoder
        ae = self._load_autoencoder()
        if ae:
            self.models["autoencoder"] = ae
            logger.info("Loaded autoencoder model")

        # Load preprocessors
        self._load_preprocessors()

    def _load_sklearn_model(self, model_name: str) -> Optional[Any]:
        """Load a sklearn/XGBoost model using joblib."""
        model_dir = self.model_path / model_name
        for ext in ["model.joblib", "model.pkl"]:
            path = model_dir / ext
            if path.exists():
                try:
                    return joblib.load(path)
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {e}")
        return None

    def _load_autoencoder(self) -> Optional[Any]:
        """Load PyTorch autoencoder."""
        try:
            import torch
            ae_path = self.model_path / "autoencoder" / "autoencoder.pt"
            threshold_path = self.model_path / "autoencoder" / "threshold.joblib"
            if ae_path.exists() and threshold_path.exists():
                from ml.train_autoencoder import AnomalyAutoencoder
                model = torch.load(ae_path, map_location="cpu")
                model.eval()
                threshold = joblib.load(threshold_path)
                return {"model": model, "threshold": threshold}
        except Exception as e:
            logger.error(f"Error loading autoencoder: {e}")
        return None

    def _load_preprocessors(self) -> None:
        """Load fitted preprocessor pipelines."""
        for model_name in ["random_forest", "xgboost", "isolation_forest", "autoencoder"]:
            scaler_path = self.model_path / model_name / "scaler.joblib"
            if scaler_path.exists():
                try:
                    self.preprocessors[model_name] = joblib.load(scaler_path)
                    logger.info(f"Loaded preprocessor for {model_name}")
                except Exception as e:
                    logger.error(f"Error loading preprocessor for {model_name}: {e}")

    def _traffic_to_features(self, record: TrafficRecord) -> np.ndarray:
        """Convert a TrafficRecord to a numpy feature vector."""
        proto_num = PROTOCOL_MAP.get(record.protocol.upper(), 0)
        duration = record.duration if record.duration > 0 else 0.001
        packets_ps = record.packets_per_sec or (record.packets / duration)
        bytes_ps = record.bytes_per_sec or (record.bytes / duration)
        avg_pkt = record.avg_packet_size or (record.bytes / max(record.packets, 1))

        features = [
            record.src_port,
            record.dst_port,
            proto_num,
            record.packets,
            record.bytes,
            record.duration,
            packets_ps,
            bytes_ps,
            avg_pkt,
            record.syn_count or 0,
            record.ack_count or 0,
            record.fin_count or 0,
            record.rst_count or 0,
            record.fwd_packets or record.packets,
            record.bwd_packets or 0,
        ]
        return np.array(features, dtype=float).reshape(1, -1)

    def _calculate_risk_score(self, anomaly_score: float, model_name: str) -> int:
        """Convert anomaly score [0,1] to risk score [0,100]."""
        return min(100, int(anomaly_score * 100))

    def _get_risk_level(self, risk_score: int) -> str:
        """Map numeric risk score to categorical risk level."""
        for level, (low, high) in RISK_THRESHOLDS.items():
            if low <= risk_score < high:
                return level
        return "CRITICAL"

    def _get_attack_label(self, model: Any, proba: np.ndarray) -> str:
        """Get attack type label from classifier."""
        attack_types = [
            "NORMAL", "DDoS", "DoS", "Port Scan", "Botnet",
            "Brute Force", "Infiltration", "Web Attack", "Heartbleed",
        ]
        if hasattr(model, "classes_"):
            classes = model.classes_
            pred_idx = np.argmax(proba[0])
            label = str(classes[pred_idx])
            return label if label in attack_types else (
                "NORMAL" if label in ["BENIGN", "Normal", "0"] else "ANOMALY"
            )
        return "ANOMALY"

    def predict_single(
        self, record: TrafficRecord, model_name: str = "random_forest"
    ) -> PredictionResult:
        """Run inference on a single traffic record."""
        start = time.perf_counter()
        features = self._traffic_to_features(record)

        if model_name not in self.models:
            return self._fallback_predict(record, model_name, features)

        model = self.models[model_name]
        scaler = self.preprocessors.get(model_name)

        if scaler is not None:
            try:
                features = scaler.transform(features)
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}")

        try:
            if model_name in ("random_forest", "xgboost"):
                return self._supervised_predict(model, features, model_name, record)
            elif model_name == "isolation_forest":
                return self._isolation_predict(model, features, record)
            elif model_name == "autoencoder":
                return self._autoencoder_predict(model, features, record)
        except Exception as e:
            logger.error(f"Prediction error [{model_name}]: {e}")
            return self._fallback_predict(record, model_name, features)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Inference [{model_name}]: {elapsed_ms:.2f}ms")

    def _supervised_predict(
        self, model: Any, features: np.ndarray, model_name: str, record: TrafficRecord
    ) -> PredictionResult:
        """Random Forest / XGBoost prediction."""
        proba = model.predict_proba(features)
        pred_class = self._get_attack_label(model, proba)
        is_anomaly = pred_class not in ("NORMAL", "BENIGN", "Normal", "0")

        # Anomaly score = 1 - confidence of NORMAL class
        normal_idx = 0
        if hasattr(model, "classes_"):
            normal_classes = {"NORMAL", "BENIGN", "Normal", "0"}
            for i, c in enumerate(model.classes_):
                if str(c) in normal_classes:
                    normal_idx = i
                    break
        confidence = float(np.max(proba[0]))
        anomaly_score = 1.0 - float(proba[0][normal_idx]) if is_anomaly else float(proba[0][normal_idx])

        risk_score = self._calculate_risk_score(anomaly_score if is_anomaly else 1 - anomaly_score, model_name)

        # Feature importance explanation
        explanation = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            explanation = {
                col: float(importances[i])
                for i, col in enumerate(FEATURE_COLUMNS)
                if i < len(importances)
            }

        return PredictionResult(
            prediction="ANOMALY" if is_anomaly else "NORMAL",
            attack_type=pred_class if is_anomaly else None,
            anomaly_score=anomaly_score if is_anomaly else 0.0,
            confidence=confidence,
            risk_level=self._get_risk_level(risk_score if is_anomaly else 0),
            risk_score=risk_score if is_anomaly else 0,
            model_used=model_name,
            explanation=explanation,
        )

    def _isolation_predict(
        self, model: Any, features: np.ndarray, record: TrafficRecord
    ) -> PredictionResult:
        """Isolation Forest prediction."""
        pred = model.predict(features)[0]  # 1=normal, -1=anomaly
        score = model.score_samples(features)[0]  # more negative = more anomalous
        # Normalize score to [0,1]
        anomaly_score = float(np.clip(1 - (score + 0.5) / 0.5, 0, 1))
        is_anomaly = pred == -1
        risk_score = self._calculate_risk_score(anomaly_score if is_anomaly else 0.0, "isolation_forest")

        return PredictionResult(
            prediction="ANOMALY" if is_anomaly else "NORMAL",
            attack_type="Unknown Anomaly" if is_anomaly else None,
            anomaly_score=anomaly_score if is_anomaly else 0.0,
            confidence=anomaly_score,
            risk_level=self._get_risk_level(risk_score if is_anomaly else 0),
            risk_score=risk_score if is_anomaly else 0,
            model_used="isolation_forest",
        )

    def _autoencoder_predict(
        self, ae_bundle: Dict, features: np.ndarray, record: TrafficRecord
    ) -> PredictionResult:
        """Autoencoder reconstruction-error prediction."""
        import torch
        model = ae_bundle["model"]
        threshold = ae_bundle["threshold"]
        x = torch.FloatTensor(features)
        with torch.no_grad():
            reconstructed = model(x)
            mse = float(torch.mean((x - reconstructed) ** 2).item())
        is_anomaly = mse > threshold
        anomaly_score = min(1.0, mse / (threshold * 2))
        risk_score = self._calculate_risk_score(anomaly_score, "autoencoder")

        return PredictionResult(
            prediction="ANOMALY" if is_anomaly else "NORMAL",
            attack_type="Reconstructed Anomaly" if is_anomaly else None,
            anomaly_score=anomaly_score,
            confidence=anomaly_score,
            risk_level=self._get_risk_level(risk_score if is_anomaly else 0),
            risk_score=risk_score if is_anomaly else 0,
            model_used="autoencoder",
        )

    def _fallback_predict(
        self, record: TrafficRecord, model_name: str, features: np.ndarray
    ) -> PredictionResult:
        """Heuristic fallback when no model is available."""
        # Simple rule-based heuristics for demo
        packets_per_sec = record.packets / max(record.duration, 0.001)
        anomaly_indicators = 0
        if packets_per_sec > 1000:
            anomaly_indicators += 2
        if record.dst_port in [22, 23, 3389, 445] and record.packets > 50:
            anomaly_indicators += 1
        if record.syn_count and record.ack_count and record.syn_count > record.ack_count * 10:
            anomaly_indicators += 2
        if record.bytes > 1_000_000 and record.duration < 1:
            anomaly_indicators += 1

        is_anomaly = anomaly_indicators >= 2
        anomaly_score = min(1.0, anomaly_indicators * 0.2)
        risk_score = self._calculate_risk_score(anomaly_score, model_name)

        return PredictionResult(
            prediction="ANOMALY" if is_anomaly else "NORMAL",
            attack_type="Heuristic Detection" if is_anomaly else None,
            anomaly_score=anomaly_score,
            confidence=0.5,
            risk_level=self._get_risk_level(risk_score if is_anomaly else 0),
            risk_score=risk_score if is_anomaly else 0,
            model_used=f"{model_name}_heuristic",
        )

    def predict_batch(
        self, records: List[TrafficRecord], model_name: str = "random_forest"
    ) -> BatchPredictionResult:
        """Run batch inference on multiple records."""
        start = time.perf_counter()
        results = [self.predict_single(r, model_name) for r in records]
        elapsed_ms = (time.perf_counter() - start) * 1000

        anomalies = sum(1 for r in results if r.prediction == "ANOMALY")
        return BatchPredictionResult(
            total=len(results),
            anomalies=anomalies,
            normal=len(results) - anomalies,
            anomaly_rate=anomalies / max(len(results), 1),
            results=results,
            processing_time_ms=elapsed_ms,
        )

    def get_available_models(self) -> List[str]:
        """Return list of loaded model names."""
        return list(self.models.keys())


# Module-level singleton
detection_service = DetectionService()
