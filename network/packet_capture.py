"""
Passive network packet capture using Scapy.

⚠️  IMPORTANT: Only use on networks where you have explicit authorization.
    Packet capture requires administrator/root privileges.
    This module operates in PASSIVE MONITORING mode only.
"""

import threading
import time
from typing import Callable, Optional, List
from loguru import logger
from backend.schemas.traffic import TrafficRecord

CAPTURE_WARNING = (
    "⚠️  AUTHORIZATION REQUIRED: Packet capture must only be used on networks "
    "where you have explicit written authorization. "
    "Running without authorization may violate computer fraud laws."
)


class PassiveCapture:
    """
    Passive network traffic monitor using Scapy sniff.
    Captures packets and emits TrafficRecord objects via callback.
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        callback: Optional[Callable[[TrafficRecord], None]] = None,
        packet_filter: str = "ip",
        max_packets: int = 1000,
    ):
        self.interface = interface
        self.callback = callback
        self.packet_filter = packet_filter
        self.max_packets = max_packets
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._packet_count = 0
        logger.warning(CAPTURE_WARNING)

    def _process_packet(self, pkt) -> None:
        """Process a single captured packet."""
        try:
            from scapy.all import IP, TCP, UDP, ICMP
            if not pkt.haslayer(IP):
                return
            ip = pkt[IP]
            src_ip, dst_ip = ip.src, ip.dst
            proto, src_port, dst_port = "IP", 0, 0
            syn = ack = fin = rst = 0

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                proto = "TCP"
                src_port, dst_port = int(tcp.sport), int(tcp.dport)
                flags = int(tcp.flags)
                syn = bool(flags & 0x02)
                ack = bool(flags & 0x10)
                fin = bool(flags & 0x01)
                rst = bool(flags & 0x04)
            elif pkt.haslayer(UDP):
                from scapy.all import UDP as SUDP
                udp = pkt[SUDP]
                proto = "UDP"
                src_port, dst_port = int(udp.sport), int(udp.dport)
            elif pkt.haslayer(ICMP):
                proto = "ICMP"

            pkt_size = len(pkt)
            record = TrafficRecord(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=proto,
                packets=1,
                bytes=pkt_size,
                duration=0.0,
                avg_packet_size=float(pkt_size),
                syn_count=syn,
                ack_count=ack,
                fin_count=fin,
                rst_count=rst,
            )

            self._packet_count += 1
            if self.callback:
                self.callback(record)

        except Exception as e:
            logger.debug(f"Packet processing error: {e}")

    def _stop_filter(self, _) -> bool:
        return self._stop_event.is_set() or self._packet_count >= self.max_packets

    def start(self) -> None:
        """Start passive capture in a background thread."""
        try:
            from scapy.all import sniff
        except ImportError:
            raise RuntimeError("scapy is required — pip install scapy")

        logger.info(
            f"Starting passive capture on interface={self.interface or 'default'}, "
            f"filter='{self.packet_filter}', max_packets={self.max_packets}"
        )

        def _capture():
            sniff(
                iface=self.interface,
                filter=self.packet_filter,
                prn=self._process_packet,
                stop_filter=self._stop_filter,
                store=False,
            )
            logger.info(f"Capture stopped. Total packets: {self._packet_count}")

        self._thread = threading.Thread(target=_capture, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Passive capture stopped")

    @property
    def packet_count(self) -> int:
        return self._packet_count
