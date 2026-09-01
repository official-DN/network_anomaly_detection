"""
Traffic API routes: file upload, recent events, and statistics.
"""

import io
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends

from backend.schemas.traffic import TrafficRecord, TrafficStatistics
from backend.services.traffic_service import get_recent_events, get_traffic_statistics
from backend.services.detection_service import detection_service
from backend.services.alert_service import create_alert
from backend.services.traffic_service import store_traffic_event
from backend.schemas.traffic import TrafficEvent
from backend.schemas.alert import Alert, RiskLevel, AlertStatus
from backend.schemas.prediction import BatchPredictionResult
from backend.utils.logger import logger

router = APIRouter(prefix="/api/traffic", tags=["Traffic"])

ALLOWED_PCAP_EXTENSIONS = {".pcap", ".pcapng"}
ALLOWED_CSV_EXTENSIONS = {".csv"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def _parse_csv_to_records(df: pd.DataFrame) -> List[TrafficRecord]:
    """Parse a DataFrame into TrafficRecord list using flexible column mapping."""
    COLUMN_MAP = {
        "src_ip": ["src_ip", "source_ip", " Source IP", "Src IP", "srcip"],
        "dst_ip": ["dst_ip", "destination_ip", " Destination IP", "Dst IP", "dstip"],
        "src_port": ["src_port", " Source Port", "Src Port", "srcport"],
        "dst_port": ["dst_port", " Destination Port", "Dst Port", "dstport"],
        "protocol": ["protocol", "Protocol", "proto"],
        "packets": ["packets", "tot_pkts", " Total Fwd Packets", "Total Fwd Packets"],
        "bytes": ["bytes", "tot_bytes", " Total Length of Fwd Packets", "Total Length of Fwd Packets"],
        "duration": ["duration", " Flow Duration", "Flow Duration"],
    }

    def find_col(target_names: List[str]) -> Optional[str]:
        for name in target_names:
            for col in df.columns:
                if col.strip().lower() == name.strip().lower():
                    return col
        return None

    mapped = {field: find_col(candidates) for field, candidates in COLUMN_MAP.items()}
    records = []

    for _, row in df.iterrows():
        try:
            rec = TrafficRecord(
                src_ip=str(row[mapped["src_ip"]]) if mapped["src_ip"] else "0.0.0.0",
                dst_ip=str(row[mapped["dst_ip"]]) if mapped["dst_ip"] else "0.0.0.0",
                src_port=int(row[mapped["src_port"]]) if mapped["src_port"] else 0,
                dst_port=int(row[mapped["dst_port"]]) if mapped["dst_port"] else 0,
                protocol=str(row[mapped["protocol"]]) if mapped["protocol"] else "TCP",
                packets=max(0, int(float(row[mapped["packets"]]))) if mapped["packets"] else 1,
                bytes=max(0, int(float(row[mapped["bytes"]]))) if mapped["bytes"] else 0,
                duration=max(0.0, float(row[mapped["duration"]])) if mapped["duration"] else 0.0,
            )
            records.append(rec)
        except Exception:
            continue

    return records


@router.post("/analyze")
async def analyze_traffic(record: TrafficRecord, model: str = "random_forest"):
    """Analyze a single traffic record (JSON body)."""
    result = detection_service.predict_single(record, model)
    event = TrafficEvent(
        **record.model_dump(),
        prediction=result.prediction,
        anomaly_score=result.anomaly_score,
        confidence=result.confidence,
        risk_level=result.risk_level,
        model_used=result.model_used,
    )
    await store_traffic_event(event)
    return {"success": True, "data": result.model_dump()}


@router.post("/upload-pcap")
async def upload_pcap(file: UploadFile = File(...), model: str = "random_forest"):
    """
    Upload a PCAP/PCAPNG file for offline analysis.
    Requires scapy to be installed.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_PCAP_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {ALLOWED_PCAP_EXTENSIONS}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 100MB)")

    try:
        from network.pcap_reader import PCAPReader
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        reader = PCAPReader(tmp_path)
        records = reader.extract_records()
        if not records:
            return {"success": True, "data": {"message": "No flows extracted", "analyzed": 0}}

        result = detection_service.predict_batch(records, model)
        return {
            "success": True,
            "data": {
                "filename": file.filename,
                "total_flows": result.total,
                "anomalies": result.anomalies,
                "normal": result.normal,
                "anomaly_rate": result.anomaly_rate,
                "processing_time_ms": result.processing_time_ms,
            }
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="PCAP analysis requires scapy (pip install scapy)")
    except Exception as e:
        logger.error(f"PCAP analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-csv", response_model=dict)
async def upload_csv(file: UploadFile = File(...), model: str = "random_forest"):
    """Upload a CSV dataset for batch analysis."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_CSV_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 100MB)")

    try:
        df = pd.read_csv(io.BytesIO(content))
        df = df.replace([float("inf"), float("-inf")], None).dropna(how="all")
        records = _parse_csv_to_records(df)

        if not records:
            return {"success": False, "error": {"code": "PARSE_ERROR", "message": "No valid records parsed"}}

        result = detection_service.predict_batch(records[:5000], model)  # cap at 5000

        for rec, pred in zip(records[:5000], result.results):
            event = TrafficEvent(
                **rec.model_dump(),
                prediction=pred.prediction,
                anomaly_score=pred.anomaly_score,
                confidence=pred.confidence,
                risk_level=pred.risk_level,
                model_used=pred.model_used,
            )
            await store_traffic_event(event)
            if pred.prediction == "ANOMALY" and pred.risk_score >= 30:
                alert = Alert(
                    timestamp=datetime.utcnow(),
                    attack_type=pred.attack_type or "CSV Anomaly",
                    src_ip=rec.src_ip,
                    dst_ip=rec.dst_ip,
                    src_port=rec.src_port,
                    dst_port=rec.dst_port,
                    protocol=rec.protocol,
                    anomaly_score=pred.anomaly_score,
                    confidence=pred.confidence,
                    risk_level=RiskLevel(pred.risk_level),
                    model_used=pred.model_used,
                    status=AlertStatus.NEW,
                )
                await create_alert(alert)

        return {
            "success": True,
            "data": {
                "filename": file.filename,
                "rows_in_file": len(df),
                "records_analyzed": result.total,
                "anomalies": result.anomalies,
                "normal": result.normal,
                "anomaly_rate": round(result.anomaly_rate * 100, 2),
                "processing_time_ms": round(result.processing_time_ms, 2),
            }
        }
    except Exception as e:
        logger.error(f"CSV upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent")
async def recent_traffic(limit: int = 100):
    """Get most recent traffic events."""
    events = await get_recent_events(limit=min(limit, 1000))
    return {"success": True, "data": events, "count": len(events)}


@router.get("/statistics")
async def traffic_statistics():
    """Get aggregate traffic statistics."""
    stats = await get_traffic_statistics()
    return {"success": True, "data": stats.model_dump()}
