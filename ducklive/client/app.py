"""DuckLive Client — browser-based capture + virtual device output.

The client has two parts:
  1. Web UI (browser) — captures camera/mic via getUserMedia, sends to server,
     shows processed preview
  2. Python process — receives processed stream from server, outputs to
     virtual camera (pyvirtualcam) + virtual microphone (BlackHole)

The browser connects directly to the server's /feed WebSocket endpoint.
The Python process connects to the server's /stream WebSocket endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time

from rich.console import Console

from ducklive.client.discover import ServerDiscovery
from ducklive.client.receiver import StreamReceiver
from ducklive.client.virtual_cam import VirtualCamera
from ducklive.client.virtual_mic import VirtualMicrophone

logger = logging.getLogger(__name__)
console = Console()


class DuckLiveClient:
    """Main client application."""

    def __init__(
        self,
        server_url: str | None = None,
        enable_camera: bool = True,
        enable_audio: bool = True,
        webui_port: int = 8081,
    ):
        self.server_url = server_url  # ws://host:port
        self.enable_camera = enable_camera
        self.enable_audio = enable_audio
        self.webui_port = webui_port

        self.receiver = StreamReceiver()
        self.virtual_cam = VirtualCamera() if enable_camera else None
        self.virtual_mic = VirtualMicrophone() if enable_audio else None

        self._running = False
        self._discovered_server = None

    @property
    def server_ws_base(self) -> str | None:
        """Base WebSocket URL (ws://host:port) without path."""
        return self.server_url

    @property
    def server_dashboard_url(self) -> str | None:
        """HTTP URL of the server's dashboard/API (e.g. http://host:8080)."""
        if self._discovered_server:
            return self._discovered_server.dashboard_url
        # Derive from WS URL: ws://host:8765 → http://host:8080
        if self.server_url:
            from urllib.parse import urlparse
            parsed = urlparse(self.server_url)
            host = parsed.hostname or "localhost"
            # Default dashboard port
            return f"http://{host}:8080"
        return None

    async def start(self) -> None:
        """Start the client."""
        console.print("[bold green]🦆 DuckLive Client starting...[/]")

        # 1. Find server
        ws_url = self.server_url
        if not ws_url:
            ws_url = await self._discover_server()
            if not ws_url:
                console.print("[bold red]❌ No DuckLive server found on the network.[/]")
                console.print("[dim]Hint: start with --server-url ws://host:8765[/]")
                return

        # Store resolved URL
        self.server_url = ws_url

        # 2. Connect Python receiver to /stream (for virtual device output)
        stream_url = ws_url.rstrip("/") + "/stream"
        try:
            await self.receiver.connect(stream_url)
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not connect to server stream: {e}[/]")
            console.print("[dim]Web UI will still work for preview.[/]")

        # 3. Start virtual devices
        if self.virtual_cam:
            try:
                self.virtual_cam.start()
            except Exception as e:
                console.print(f"[yellow]⚠️  Virtual camera unavailable: {e}[/]")
                self.virtual_cam = None

        if self.virtual_mic:
            try:
                self.virtual_mic.start()
            except Exception as e:
                console.print(f"[yellow]⚠️  Virtual microphone unavailable: {e}[/]")
                self.virtual_mic = None

        self._running = True
        console.print("[bold green]✅ DuckLive Client is running![/]")
        console.print(f"   📡 Server: {ws_url}")
        console.print(f"   🌐 Open in browser: [bold]http://localhost:{self.webui_port}[/]")
        console.print(f"   📹 Browser captures camera → sends to server → shows processed preview")
        if self.virtual_cam:
            console.print(f"   🎥 Virtual camera: {self.virtual_cam.device_name}")
        if self.virtual_mic:
            console.print(f"   🎤 Virtual mic: writing to BlackHole")

        # 4. Run all loops concurrently
        tasks = [self._run_webui()]
        if self.receiver.is_connected:
            tasks.append(self.receiver.receive_loop())
            tasks.append(self._output_loop())

        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        """Stop the client."""
        self._running = False
        await self.receiver.disconnect()
        if self.virtual_cam:
            self.virtual_cam.stop()
        if self.virtual_mic:
            self.virtual_mic.stop()
        console.print("[dim]DuckLive Client stopped.[/]")

    async def _discover_server(self) -> str | None:
        """Auto-discover a DuckLive server on the network."""
        console.print("[dim]🔍 Searching for DuckLive server...[/]")
        discovery = ServerDiscovery()
        servers = await asyncio.to_thread(discovery.discover, timeout=5.0)

        if not servers:
            return None

        server = servers[0]
        self._discovered_server = server
        console.print(f"[green]Found server: {server.host}[/]")
        return server.ws_url

    async def _output_loop(self) -> None:
        """Pull processed frames from receiver → push to virtual devices."""
        target_interval = 1.0 / 30

        while self._running:
            loop_start = time.perf_counter()

            if self.virtual_cam:
                jpeg = self.receiver.get_video_frame()
                if jpeg:
                    self.virtual_cam.send_jpeg(jpeg)

            if self.virtual_mic:
                chunks = self.receiver.get_all_audio_chunks()
                for chunk in chunks:
                    self.virtual_mic.write(chunk)

            elapsed = time.perf_counter() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _run_webui(self) -> None:
        """Run the client Web UI (serves camera capture interface)."""
        from ducklive.client.webui import create_client_app
        import uvicorn

        app = create_client_app(self)
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.webui_port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()


def start_client(
    server_url: str | None = None,
    enable_camera: bool = True,
    enable_audio: bool = True,
    webui_port: int = 8081,
    dev: bool = False,
) -> None:
    """Entry point — start the DuckLive client."""
    logging.basicConfig(
        level=logging.DEBUG if dev else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    client = DuckLiveClient(
        server_url=server_url,
        enable_camera=enable_camera,
        enable_audio=enable_audio,
        webui_port=webui_port,
    )

    async def _run():
        try:
            await client.start()
        except KeyboardInterrupt:
            pass
        finally:
            await client.stop()

    asyncio.run(_run())
