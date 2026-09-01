"""
PCAP/PCAPNG file reader using Scapy.
Groups packets into flows and extracts ML-ready features.
"""

from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime
import ipaddress

from loguru import logger
from backend.schemas.traffic import TrafficRecord


class FlowKey:
    """5-tuple flow identifier (bidirectional)."""
    def __init__(self, src_ip, dst_ip, src_port, dst_port, proto):
        # Normalize direction: smaller IP always first
        if src_ip <= dst_ip:
            self.key = (src_ip, dst_ip, src_port, dst_port, proto)
            self.fwd = True
        else:
            self.key = (dst_ip, src_ip, dst_port, src_port, proto)
            self.fwd = False

    def __hash__(self): return hash(self.key)
    def __eq__(self, other): return self.key == other.key


class PCAPReader:
    """Reads a PCAP file and extracts network flow features."""

    def __init__(self, pcap_path: str):
        self.pcap_path = Path(pcap_path)
        if not self.pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    def _safe_ip(self, ip_str: str) -> str:
        """Validate and return IP or fallback."""
        try:
            ipaddress.ip_address(ip_str)
            return ip_str
        except ValueError:
            return "0.0.0.0"

    def extract_flows(self) -> List[Dict]:
        """
        Read PCAP and group packets into bidirectional flows.
        Returns list of flow feature dicts.
        """
        try:
            from scapy.all import rdpcap, IP, TCP, UDP, ICMP
        except ImportError:
            raise RuntimeError("scapy is required for PCAP analysis — pip install scapy")

        logger.info(f"Reading PCAP: {self.pcap_path.name}")
        try:
            packets = rdpcap(str(self.pcap_path))
        except Exception as e:
            raise RuntimeError(f"Failed to read PCAP: {e}")

        logger.info(f"Loaded {len(packets)} packets")

        # Group packets by 5-tuple flow
        flows: Dict[tuple, Dict] = defaultdict(lambda: {
            "src_ip": None, "dst_ip": None,
            "src_port": 0, "dst_port": 0, "protocol": "TCP",
            "packet_times": [], "packet_sizes": [],
            "fwd_packets": 0, "bwd_packets": 0,
            "fwd_bytes": 0, "bwd_bytes": 0,
            "syn": 0, "ack": 0, "fin": 0, "rst": 0, "psh": 0,
        })

        for pkt in packets:
            if not pkt.haslayer(IP):
                continue
            ip = pkt[IP]
            src_ip = self._safe_ip(ip.src)
            dst_ip = self._safe_ip(ip.dst)
            proto = "TCP"
            src_port = dst_port = 0

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                src_port = int(tcp.sport)
                dst_port = int(tcp.dport)
                proto = "TCP"
                fk = FlowKey(src_ip, dst_ip, src_port, dst_port, proto)
                flow = flows[fk.key]
                # TCP flags
                flags = int(tcp.flags)
                flow["syn"] += bool(flags & 0x02)
                flow["ack"] += bool(flags & 0x10)
                flow["fin"] += bool(flags & 0x01)
                flow["rst"] += bool(flags & 0x04)
                flow["psh"] += bool(flags & 0x08)
            elif pkt.haslayer(UDP):
                from scapy.all import UDP as ScapyUDP
                udp = pkt[ScapyUDP]
                src_port = int(udp.sport)
                dst_port = int(udp.dport)
                proto = "UDP"
                fk = FlowKey(src_ip, dst_ip, src_port, dst_port, proto)
                flow = flows[fk.key]
            elif pkt.haslayer(ICMP):
                proto = "ICMP"
                fk = FlowKey(src_ip, dst_ip, 0, 0, proto)
                flow = flows[fk.key]
            else:
                continue

            flow = flows[fk.key]
            pkt_size = len(pkt)
            pkt_time = float(pkt.time) if hasattr(pkt, "time") else 0.0

            if flow["src_ip"] is None:
                flow["src_ip"] = src_ip
                flow["dst_ip"] = dst_ip
                flow["src_port"] = src_port
                flow["dst_port"] = dst_port
                flow["protocol"] = proto

            flow["packet_times"].append(pkt_time)
            flow["packet_sizes"].append(pkt_size)

            if fk.fwd:
                flow["fwd_packets"] += 1
                flow["fwd_bytes"] += pkt_size
            else:
                flow["bwd_packets"] += 1
                flow["bwd_bytes"] += pkt_size

        # Compute flow statistics
        flow_records = []
        for flow in flows.values():
            times = flow["packet_times"]
            sizes = flow["packet_sizes"]
            if not times:
                continue
            duration = max(times) - min(times) if len(times) > 1 else 0.001
            total_pkts = len(sizes)
            total_bytes = sum(sizes)

            flow_records.append({
                "src_ip": flow["src_ip"] or "0.0.0.0",
                "dst_ip": flow["dst_ip"] or "0.0.0.0",
                "src_port": flow["src_port"],
                "dst_port": flow["dst_port"],
                "protocol": flow["protocol"],
                "packets": total_pkts,
                "bytes": total_bytes,
                "duration": round(duration, 6),
                "packets_per_sec": round(total_pkts / max(duration, 0.001), 2),
                "bytes_per_sec": round(total_bytes / max(duration, 0.001), 2),
                "avg_packet_size": round(total_bytes / max(total_pkts, 1), 2),
                "fwd_packets": flow["fwd_packets"],
                "bwd_packets": flow["bwd_packets"],
                "syn_count": flow["syn"],
                "ack_count": flow["ack"],
                "fin_count": flow["fin"],
                "rst_count": flow["rst"],
            })

        logger.info(f"Extracted {len(flow_records)} flows from {len(packets)} packets")
        return flow_records

    def extract_records(self) -> List[TrafficRecord]:
        """Extract flows and return as TrafficRecord list."""
        flows = self.extract_flows()
        records = []
        for flow in flows:
            try:
                records.append(TrafficRecord(**flow))
            except Exception:
                continue
        return records
