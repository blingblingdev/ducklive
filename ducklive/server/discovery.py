"""mDNS/Bonjour service advertisement for LAN discovery."""

from __future__ import annotations

import asyncio
import logging
import socket

from zeroconf import IPVersion
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

from ducklive.common.config import MDNS_SERVICE_TYPE, MDNS_SERVICE_NAME

logger = logging.getLogger(__name__)


class ServiceAdvertiser:
    """Advertise DuckLive server on the local network via mDNS/Bonjour.

    Uses async zeroconf to avoid blocking the asyncio event loop.
    """

    def __init__(self, ws_port: int = 8765, dashboard_port: int = 8080):
        self.ws_port = ws_port
        self.dashboard_port = dashboard_port
        self._zeroconf: AsyncZeroconf | None = None
        self._info: AsyncServiceInfo | None = None

    async def start(self) -> None:
        """Start advertising the service."""
        hostname = socket.gethostname()
        local_ip = self._get_local_ip()

        self._info = AsyncServiceInfo(
            MDNS_SERVICE_TYPE,
            MDNS_SERVICE_NAME,
            addresses=[socket.inet_aton(local_ip)],
            port=self.ws_port,
            properties={
                "version": "0.1.0",
                "dashboard": str(self.dashboard_port),
                "hostname": hostname,
            },
            server=f"{hostname}.local.",
        )

        self._zeroconf = AsyncZeroconf(ip_version=IPVersion.V4Only)
        await self._zeroconf.async_register_service(self._info)
        logger.info(
            f"mDNS: Advertising DuckLive at {local_ip}:{self.ws_port} "
            f"(dashboard: {self.dashboard_port})"
        )

    async def stop(self) -> None:
        """Stop advertising."""
        if self._zeroconf and self._info:
            await self._zeroconf.async_unregister_service(self._info)
            await self._zeroconf.async_close()
            logger.info("mDNS: Stopped advertising")

    def _get_local_ip(self) -> str:
        """Get the local LAN IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
