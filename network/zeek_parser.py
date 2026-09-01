"""
Zeek log parser — converts Zeek conn.log / dns.log / http.log
into the internal TrafficRecord schema.
"""

import pandas as pd
from pathlib import Path
from typing import List
from loguru import logger
from backend.schemas.traffic import TrafficRecord


PROTO_MAP = {"tcp": "TCP", "udp": "UDP", "icmp": "ICMP"}


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def parse_conn_log(log_path: str) -> List[TrafficRecord]:
    """
    Parse a Zeek conn.log TSV file into TrafficRecord list.
    Expected columns: ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p,
                      proto, service, duration, orig_bytes, resp_bytes,
                      orig_pkts, resp_pkts
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Zeek log not found: {log_path}")

    # Zeek logs use TSV with # comment lines for headers
    rows = []
    field_names = []
    with open(path) as f:
        for line in f:
            if line.startswith("#fields"):
                field_names = line.strip().split("\t")[1:]
            elif line.startswith("#"):
                continue
            else:
                values = line.strip().split("\t")
                if field_names and values:
                    rows.append(dict(zip(field_names, values)))

    if not rows:
        logger.warning(f"No data rows in {log_path}")
        return []

    df = pd.DataFrame(rows)
    logger.info(f"Parsed {len(df)} Zeek conn.log rows from {path.name}")

    records = []
    for _, row in df.iterrows():
        try:
            proto_raw = str(row.get("proto", "tcp")).lower()
            proto = PROTO_MAP.get(proto_raw, "TCP")
            orig_pkts = _safe_int(row.get("orig_pkts", 0))
            resp_pkts = _safe_int(row.get("resp_pkts", 0))
            orig_bytes = _safe_int(row.get("orig_bytes", 0))
            resp_bytes = _safe_int(row.get("resp_bytes", 0))
            duration = _safe_float(row.get("duration", 0))
            total_pkts = orig_pkts + resp_pkts
            total_bytes = orig_bytes + resp_bytes

            record = TrafficRecord(
                src_ip=str(row.get("id.orig_h", "0.0.0.0")),
                dst_ip=str(row.get("id.resp_h", "0.0.0.0")),
                src_port=_safe_int(row.get("id.orig_p", 0)),
                dst_port=_safe_int(row.get("id.resp_p", 0)),
                protocol=proto,
                packets=total_pkts,
                bytes=total_bytes,
                duration=duration,
                packets_per_sec=round(total_pkts / max(duration, 0.001), 2),
                bytes_per_sec=round(total_bytes / max(duration, 0.001), 2),
                avg_packet_size=round(total_bytes / max(total_pkts, 1), 2),
                fwd_packets=orig_pkts,
                bwd_packets=resp_pkts,
            )
            records.append(record)
        except Exception as e:
            logger.debug(f"Zeek row parse error: {e}")
            continue

    logger.info(f"Converted {len(records)} Zeek flows to TrafficRecord")
    return records


def parse_http_log(log_path: str) -> List[dict]:
    """Parse Zeek http.log — returns dicts with method, host, uri, status."""
    path = Path(log_path)
    if not path.exists():
        return []

    rows, field_names = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#fields"):
                field_names = line.strip().split("\t")[1:]
            elif not line.startswith("#"):
                values = line.strip().split("\t")
                if field_names:
                    rows.append(dict(zip(field_names, values)))

    return rows


def parse_dns_log(log_path: str) -> List[dict]:
    """Parse Zeek dns.log — returns dicts with query, qtype, answers."""
    path = Path(log_path)
    if not path.exists():
        return []

    rows, field_names = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#fields"):
                field_names = line.strip().split("\t")[1:]
            elif not line.startswith("#"):
                values = line.strip().split("\t")
                if field_names:
                    rows.append(dict(zip(field_names, values)))

    return rows
