"""mDNS/Bonjour service discovery — find DuckLive servers on LAN."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from zeroconf import ServiceBrowser, ServiceStateChange, Zeroconf

from ducklive.common.config import MDNS_SERVICE_TYPE

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredServer:
    """A DuckLive server found on the network."""

    name: str
    host: str
    ws_port: int
    dashboard_port: int
    version: str

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.ws_port}"

    @property
    def dashboard_url(self) -> str:
        return f"http://{self.host}:{self.dashboard_port}"


class ServerDiscovery:
    """Discover DuckLive servers on the local network via mDNS."""

    def __init__(self):
        self._servers: dict[str, DiscoveredServer] = {}
        self._zeroconf: Zeroconf | None = None
        self._browser: ServiceBrowser | None = None

    def start(self) -> None:
        """Start listening for DuckLive servers."""
        self._zeroconf = Zeroconf()
        self._browser = ServiceBrowser(
            self._zeroconf, MDNS_SERVICE_TYPE, handlers=[self._on_change]
        )
        logger.info("🔍 Searching for DuckLive servers on the network...")

    def stop(self) -> None:
        """Stop listening."""
        if self._zeroconf:
            self._zeroconf.close()

    def discover(self, timeout: float = 5.0) -> list[DiscoveredServer]:
        """Discover servers, waiting up to timeout seconds.

        Returns list of discovered servers.
        """
        self.start()
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self._servers:
                break
            time.sleep(0.1)

        servers = list(self._servers.values())
        self.stop()
        return servers

    @property
    def servers(self) -> list[DiscoveredServer]:
        return list(self._servers.values())

    def _on_change(
        self, zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        if state_change is ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                addresses = info.parsed_addresses()
                if not addresses:
                    return
                host = addresses[0]
                props = {k.decode(): v.decode() if isinstance(v, bytes) else v for k, v in info.properties.items()}

                server = DiscoveredServer(
                    name=name,
                    host=host,
                    ws_port=info.port,
                    dashboard_port=int(props.get("dashboard", 8080)),
                    version=props.get("version", "unknown"),
                )
                self._servers[name] = server
                logger.info(f"✅ Found DuckLive server: {host}:{info.port} (v{server.version})")

        elif state_change is ServiceStateChange.Removed:
            self._servers.pop(name, None)
            logger.info(f"Server removed: {name}")
