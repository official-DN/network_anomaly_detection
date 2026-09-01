"""
Data preprocessing pipeline: cleaning, encoding, scaling.
All transformers fitted ONLY on training data to prevent data leakage.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from loguru import logger


NUMERIC_FEATURES = [
    "src_port", "dst_port", "duration", "packets", "bytes",
    "fwd_pkts", "bwd_pkts", "pkt_rate", "byte_rate",
    "syn_count", "ack_count", "fin_count", "rst_count",
]

CATEGORICAL_FEATURES = ["protocol"]

PROTOCOL_MAP = {"TCP": 0, "UDP": 1, "ICMP": 2, "HTTP": 3, "DNS": 4, "TLS": 5}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw DataFrame:
    - Remove infinite values
    - Drop fully-NA rows
    - Remove duplicates
    - Clip extreme outliers
    """
    original_len = len(df)

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows that are all NaN
    df = df.dropna(how="all")

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Clip numeric columns to reasonable bounds
    for col in df.select_dtypes(include=[np.number]).columns:
        p1 = df[col].quantile(0.001)
        p99 = df[col].quantile(0.999)
        df[col] = df[col].clip(lower=p1, upper=p99)

    logger.info(f"Cleaned: {original_len} → {len(df)} rows ({original_len - len(df)} removed)")
    return df.reset_index(drop=True)


def encode_protocol(df: pd.DataFrame) -> pd.DataFrame:
    """Encode protocol column to numeric."""
    if "protocol" in df.columns:
        df["protocol_num"] = df["protocol"].map(PROTOCOL_MAP).fillna(99).astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Determine which feature columns are available in the DataFrame."""
    available = []
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            available.append(col)
    if "protocol_num" in df.columns:
        available.append("protocol_num")
    # Include any other numeric columns not in our predefined list
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in available and col not in ("label", "Label"):
            available.append(col)
    return available


def build_preprocessing_pipeline() -> Pipeline:
    """Build a sklearn Pipeline for imputation and scaling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def preprocess_dataset(
    df: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> dict:
    """
    Full preprocessing pipeline:
    1. Clean data
    2. Encode categorical features
    3. Select feature columns
    4. Split into train/val/test (fitted only on train)
    5. Apply imputation and scaling
    6. Return split datasets

    Returns dict with keys: X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    """
    df = clean_dataframe(df)
    df = encode_protocol(df)
    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")

    X = df[feature_cols].copy()
    y = labels.values if labels is not None else None

    # Handle labels alignment after cleaning
    if y is not None and len(y) != len(X):
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y[:min_len]

    if y is not None:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_ratio,
            random_state=random_state, stratify=y_train_full
        )
    else:
        X_train_full, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val = train_test_split(X_train_full, test_size=val_ratio, random_state=random_state)
        y_train = y_val = y_test = None

    # Fit ONLY on training data
    pipeline = build_preprocessing_pipeline()
    X_train_scaled = pipeline.fit_transform(X_train)
    X_val_scaled = pipeline.transform(X_val)
    X_test_scaled = pipeline.transform(X_test)

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    if y_train is not None:
        logger.info(
            f"Train anomaly rate: {y_train.mean():.2%} | "
            f"Test anomaly rate: {y_test.mean():.2%}"
        )

    if save_path:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path / "scaler.joblib")
        joblib.dump(feature_cols, path / "feature_cols.joblib")
        logger.info(f"Preprocessor saved to {save_path}")

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "pipeline": pipeline,
    }
