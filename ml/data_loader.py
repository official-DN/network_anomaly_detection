"""
Dataset loader with configurable column adapters.
Supports: CICIDS2017, UNSW-NB15, NSL-KDD, CIC-DDoS2019, and synthetic data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from loguru import logger

# Column mappings for common cybersecurity datasets
FEATURE_MAPPING: Dict[str, List[str]] = {
    "src_ip":       ["src_ip", "Source IP", " Source IP", "srcip", "Src IP"],
    "dst_ip":       ["dst_ip", "Destination IP", " Destination IP", "dstip", "Dst IP"],
    "src_port":     ["src_port", "Source Port", " Source Port", "sport", "Src Port"],
    "dst_port":     ["dst_port", "Destination Port", " Destination Port", "dport", "Dst Port"],
    "protocol":     ["protocol", "Protocol", "proto"],
    "duration":     ["duration", "Flow Duration", " Flow Duration", "dur"],
    "packets":      ["packets", "Total Fwd Packets", " Total Fwd Packets", "tot_pkts"],
    "bytes":        ["bytes", "Total Length of Fwd Packets", " Total Length of Fwd Packets", "tot_bytes"],
    "fwd_pkts":     ["Fwd Packets", " Fwd Packet Length Max", "spkts"],
    "bwd_pkts":     ["Bwd Packets", " Bwd Packet Length Max", "dpkts"],
    "pkt_rate":     ["Flow Packets/s", " Flow Packets/s"],
    "byte_rate":    ["Flow Bytes/s", " Flow Bytes/s"],
    "syn_count":    ["SYN Flag Count", " SYN Flag Cnt", "SYN"],
    "ack_count":    ["ACK Flag Count", " ACK Flag Cnt", "ACK"],
    "fin_count":    ["FIN Flag Count", " FIN Flag Cnt", "FIN"],
    "rst_count":    ["RST Flag Count", " RST Flag Cnt", "RST"],
    "label":        ["Label", "label", "class", "Class", "attack_cat", "attack"],
}

BENIGN_LABELS = {"BENIGN", "Normal", "normal", "0", "Benign", "benign", "-"}


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name in the DataFrame."""
    df_cols_stripped = {c.strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.strip() in df_cols_stripped:
            return df_cols_stripped[candidate.strip()]
    return None


def _map_label(label_series: pd.Series) -> pd.Series:
    """Convert dataset-specific labels to binary: 0=NORMAL, 1=ANOMALY."""
    return label_series.apply(lambda x: 0 if str(x).strip() in BENIGN_LABELS else 1)


def load_dataset(
    file_path: str,
    max_rows: Optional[int] = None,
    label_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load a cybersecurity dataset from CSV with automatic column mapping.

    Returns:
        features: DataFrame with standardized feature columns
        labels: binary label Series (0=normal, 1=anomaly), or None if no label
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    logger.info(f"Loading dataset: {path.name}")
    df = pd.read_csv(file_path, nrows=max_rows, low_memory=False)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Extract labels
    label_col = label_column or _find_column(df, FEATURE_MAPPING["label"])
    labels = None
    if label_col and label_col in df.columns:
        labels = _map_label(df[label_col])
        df = df.drop(columns=[label_col])
        logger.info(f"Label column: '{label_col}' | Anomaly rate: {labels.mean():.2%}")

    # Map columns to standard schema
    mapped_df = pd.DataFrame()
    feature_cols = [k for k in FEATURE_MAPPING if k != "label"]

    for field in feature_cols:
        col = _find_column(df, FEATURE_MAPPING[field])
        if col:
            mapped_df[field] = df[col]

    # Add any numeric columns not yet mapped
    for col in df.select_dtypes(include=[np.number]).columns:
        col_clean = col.strip()
        if col_clean not in mapped_df.columns:
            mapped_df[col_clean] = df[col]

    logger.info(f"Mapped {len(mapped_df.columns)} features")
    return mapped_df, labels


def load_cicids2017(data_dir: str, max_rows: Optional[int] = None):
    """Load CICIDS2017 multi-file dataset."""
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    dfs, labels = [], []
    for f in csv_files:
        df, lbl = load_dataset(str(f), max_rows=max_rows)
        dfs.append(df)
        if lbl is not None:
            labels.append(lbl)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_labels = pd.concat(labels, ignore_index=True) if labels else None
    logger.info(f"CICIDS2017: {len(combined_df)} total rows from {len(csv_files)} files")
    return combined_df, combined_labels


def generate_sample_dataset(n_samples: int = 5000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic network traffic data for development/testing.
    Returns realistic-looking flow records without real network interaction.
    """
    np.random.seed(42)
    n_normal = int(n_samples * 0.8)
    n_anomaly = n_samples - n_normal

    def make_normal(n):
        return pd.DataFrame({
            "src_ip": [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n)],
            "dst_ip": [f"10.0.{np.random.randint(0,5)}.{np.random.randint(1,100)}" for _ in range(n)],
            "src_port": np.random.randint(1024, 65535, n),
            "dst_port": np.random.choice([80, 443, 8080, 22, 25, 53, 3306], n),
            "protocol": np.random.choice(["TCP", "UDP", "ICMP"], n, p=[0.7, 0.25, 0.05]),
            "duration": np.abs(np.random.normal(2.5, 1.5, n)),
            "packets": np.random.randint(2, 100, n),
            "bytes": np.random.randint(200, 100000, n),
            "fwd_pkts": np.random.randint(1, 50, n),
            "bwd_pkts": np.random.randint(1, 50, n),
            "pkt_rate": np.abs(np.random.normal(30, 15, n)),
            "byte_rate": np.abs(np.random.normal(15000, 8000, n)),
            "syn_count": np.random.randint(0, 3, n),
            "ack_count": np.random.randint(1, 10, n),
            "fin_count": np.random.randint(0, 2, n),
            "rst_count": np.zeros(n, dtype=int),
        })

    def make_anomaly(n):
        attack_types = ["ddos", "portscan", "bruteforce", "botnet", "hightraffic"]
        n_each = n // len(attack_types)
        dfs = []

        # DDoS: very high packet rate, large bytes
        dfs.append(pd.DataFrame({
            "src_ip": [f"198.51.{np.random.randint(100,255)}.{np.random.randint(1,255)}" for _ in range(n_each)],
            "dst_ip": [f"10.0.0.{np.random.randint(1,10)}" for _ in range(n_each)],
            "src_port": np.random.randint(1024, 65535, n_each),
            "dst_port": np.full(n_each, 80),
            "protocol": np.full(n_each, "TCP"),
            "duration": np.abs(np.random.normal(0.1, 0.05, n_each)),
            "packets": np.random.randint(5000, 50000, n_each),
            "bytes": np.random.randint(500000, 5000000, n_each),
            "fwd_pkts": np.random.randint(2500, 25000, n_each),
            "bwd_pkts": np.zeros(n_each, dtype=int),
            "pkt_rate": np.random.randint(10000, 100000, n_each),
            "byte_rate": np.random.randint(1000000, 10000000, n_each),
            "syn_count": np.random.randint(4000, 40000, n_each),
            "ack_count": np.zeros(n_each, dtype=int),
            "fin_count": np.zeros(n_each, dtype=int),
            "rst_count": np.zeros(n_each, dtype=int),
        }))

        # Port scan: many low-packet flows to sequential ports
        dfs.append(pd.DataFrame({
            "src_ip": [f"203.0.{np.random.randint(100,150)}.{np.random.randint(1,50)}" for _ in range(n_each)],
            "dst_ip": [f"10.0.1.{np.random.randint(1,10)}" for _ in range(n_each)],
            "src_port": np.random.randint(40000, 60000, n_each),
            "dst_port": np.arange(1, n_each + 1) % 65535,
            "protocol": np.full(n_each, "TCP"),
            "duration": np.random.uniform(0.001, 0.01, n_each),
            "packets": np.ones(n_each, dtype=int),
            "bytes": np.random.randint(40, 80, n_each),
            "fwd_pkts": np.ones(n_each, dtype=int),
            "bwd_pkts": np.zeros(n_each, dtype=int),
            "pkt_rate": np.random.randint(100, 500, n_each),
            "byte_rate": np.random.randint(4000, 40000, n_each),
            "syn_count": np.ones(n_each, dtype=int),
            "ack_count": np.zeros(n_each, dtype=int),
            "fin_count": np.zeros(n_each, dtype=int),
            "rst_count": np.zeros(n_each, dtype=int),
        }))

        remainder = n - n_each * 2
        if remainder > 0:
            extra = make_normal(remainder)
            extra["packets"] = np.random.randint(1000, 5000, remainder)
            extra["pkt_rate"] = np.random.randint(1000, 5000, remainder)
            dfs.append(extra)

        return pd.concat(dfs, ignore_index=True)

    normal_df = make_normal(n_normal)
    anomaly_df = make_anomaly(n_anomaly)

    df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    labels = pd.Series([0] * n_normal + [1] * n_anomaly)

    # Shuffle
    idx = np.random.permutation(len(df))
    df = df.iloc[idx].reset_index(drop=True)
    labels = labels.iloc[idx].reset_index(drop=True)

    logger.info(f"Generated synthetic dataset: {n_samples} samples, {labels.mean():.2%} anomalies")
    return df, labels
