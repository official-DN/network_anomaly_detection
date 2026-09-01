"""
Train a PyTorch Autoencoder for unsupervised network anomaly detection.
Anomalies are detected via reconstruction error exceeding a threshold.
"""

import joblib
import time
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from loguru import logger

MODEL_DIR = Path("./models/autoencoder")


class AnomalyAutoencoder:
    """Lightweight numpy-based autoencoder for portability (no PyTorch dependency at inference)."""

    def __init__(self, input_dim: int, encoding_dim: int = 16):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        # Simple 2-layer encoder/decoder weights initialized with He initialization
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, 64) * scale
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, encoding_dim) * scale
        self.b2 = np.zeros(encoding_dim)
        self.W3 = np.random.randn(encoding_dim, 64) * scale
        self.b3 = np.zeros(64)
        self.W4 = np.random.randn(64, input_dim) * scale
        self.b4 = np.zeros(input_dim)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def encode(self, X: np.ndarray) -> np.ndarray:
        h1 = self._relu(X @ self.W1 + self.b1)
        return self._relu(h1 @ self.W2 + self.b2)

    def decode(self, z: np.ndarray) -> np.ndarray:
        h3 = self._relu(z @ self.W3 + self.b3)
        return h3 @ self.W4 + self.b4

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(X))

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        reconstructed = self.forward(X)
        return np.mean((X - reconstructed) ** 2, axis=1)

    def fit(self, X: np.ndarray, epochs: int = 50, lr: float = 0.01, batch_size: int = 256):
        """Train using mini-batch gradient descent with MSE loss."""
        n = len(X)
        logger.info(f"Training Autoencoder: input_dim={self.input_dim}, encoding_dim={self.encoding_dim}")
        logger.info(f"Epochs={epochs}, LR={lr}, BatchSize={batch_size}")

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuffled = X[idx]
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                batch = X_shuffled[i:i + batch_size]
                # Forward pass
                h1 = self._relu(batch @ self.W1 + self.b1)
                z = self._relu(h1 @ self.W2 + self.b2)
                h3 = self._relu(z @ self.W3 + self.b3)
                out = h3 @ self.W4 + self.b4

                # MSE loss
                diff = out - batch
                loss = np.mean(diff ** 2)
                epoch_loss += loss
                n_batches += 1

                # Backward pass (manual backprop)
                d_out = 2 * diff / len(batch)
                dW4 = h3.T @ d_out
                db4 = d_out.sum(axis=0)
                d_h3 = d_out @ self.W4.T
                d_h3_relu = d_h3 * (h3 > 0)
                dW3 = z.T @ d_h3_relu
                db3 = d_h3_relu.sum(axis=0)
                d_z = d_h3_relu @ self.W3.T
                d_z_relu = d_z * (z > 0)
                dW2 = h1.T @ d_z_relu
                db2 = d_z_relu.sum(axis=0)
                d_h1 = d_z_relu @ self.W2.T
                d_h1_relu = d_h1 * (h1 > 0)
                dW1 = batch.T @ d_h1_relu
                db1 = d_h1_relu.sum(axis=0)

                # Gradient clipping
                for grad in [dW1, dW2, dW3, dW4]:
                    np.clip(grad, -1.0, 1.0, out=grad)

                # Update weights
                self.W1 -= lr * dW1; self.b1 -= lr * db1
                self.W2 -= lr * dW2; self.b2 -= lr * db2
                self.W3 -= lr * dW3; self.b3 -= lr * db3
                self.W4 -= lr * dW4; self.b4 -= lr * db4

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] — Loss: {epoch_loss / n_batches:.6f}")


def compute_threshold(
    model: AnomalyAutoencoder,
    X_val: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """
    Compute anomaly threshold from validation reconstruction errors.
    Uses the given percentile of validation errors to set the threshold.
    """
    errors = model.reconstruction_error(X_val)
    threshold = float(np.percentile(errors, percentile))
    logger.info(
        f"Anomaly threshold (p{percentile:.0f}): {threshold:.6f} "
        f"(mean error: {errors.mean():.6f}, max: {errors.max():.6f})"
    )
    return threshold


def train_autoencoder(
    data_path: Optional[str] = None,
    encoding_dim: int = 16,
    epochs: int = 50,
    lr: float = 0.01,
    batch_size: int = 256,
    threshold_percentile: float = 95.0,
    save: bool = True,
) -> dict:
    """Train the Autoencoder anomaly detector."""
    from ml.data_loader import generate_sample_dataset, load_dataset
    from ml.preprocessing import preprocess_dataset

    if data_path:
        df, labels = load_dataset(data_path)
    else:
        logger.info("No dataset provided — using synthetic data")
        df, labels = generate_sample_dataset(n_samples=10000)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    splits = preprocess_dataset(
        df, labels, save_path=str(MODEL_DIR) if save else None
    )
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    input_dim = X_train.shape[1]
    model = AnomalyAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)

    train_start = time.perf_counter()
    model.fit(X_train, epochs=epochs, lr=lr, batch_size=batch_size)
    training_time = time.perf_counter() - train_start

    threshold = compute_threshold(model, X_val, percentile=threshold_percentile)

    # Evaluate on test set
    pred_start = time.perf_counter()
    test_errors = model.reconstruction_error(X_test)
    latency_ms = (time.perf_counter() - pred_start) / max(len(X_test), 1) * 1000
    y_pred = (test_errors > threshold).astype(int)

    metrics = {
        "model_name": "autoencoder",
        "encoding_dim": encoding_dim,
        "threshold": float(threshold),
        "threshold_percentile": threshold_percentile,
        "training_time_s": round(training_time, 2),
        "prediction_latency_ms": round(latency_ms, 4),
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
        "avg_reconstruction_error": float(test_errors.mean()),
    }

    if y_test is not None:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1_score"] = float(f1_score(y_test, y_pred, zero_division=0))
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, test_errors))
        except Exception:
            pass
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        logger.info(
            f"Autoencoder Metrics → Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1_score']:.4f}"
        )

    if save:
        joblib.dump(model, MODEL_DIR / "model.joblib")
        joblib.dump(threshold, MODEL_DIR / "threshold.joblib")
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Autoencoder saved to {MODEL_DIR}")

    return metrics


if __name__ == "__main__":
    train_autoencoder()
