"""WebSocket streaming server — receives raw frames from feed clients,
broadcasts processed frames to stream/feed clients, and sends all frames
to dashboard clients.

Three types of connections:
  - Feed connections (ws://host:port/feed)
      → Send raw video/audio upstream, receive processed frames back
  - Stream connections (ws://host:port/stream)
      → Receive processed frames only (for virtual cam/mic)
  - Dashboard connections (ws://host:port/dashboard)
      → Receive original + processed + audio levels (monitoring)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum

import websockets
from websockets.server import WebSocketServerProtocol

from ducklive.common.protocol import DASHBOARD_ONLY_TYPES, Frame, FrameType, HEADER_SIZE

logger = logging.getLogger(__name__)


class ClientRole(str, Enum):
    STREAM = "stream"        # Regular client: only processed frames
    DASHBOARD = "dashboard"  # Dashboard: original + processed + levels
    FEED = "feed"            # Feed client: sends raw, receives processed


@dataclass
class ConnectedClient:
    ws: WebSocketServerProtocol
    id: str
    address: str
    role: ClientRole
    frames_sent: int = 0
    frames_received: int = 0  # For feed clients: raw frames received


class StreamServer:
    """WebSocket server that:
    - Receives raw frames from FEED clients → puts in processing queues
    - Broadcasts processed frames to FEED + STREAM clients
    - Broadcasts all frames (original + processed + levels) to DASHBOARD clients
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        max_clients: int = 5,
        raw_video_queue: asyncio.Queue | None = None,
        raw_audio_queue: asyncio.Queue | None = None,
    ):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.raw_video_queue = raw_video_queue
        self.raw_audio_queue = raw_audio_queue
        self._clients: dict[str, ConnectedClient] = {}
        self._server = None
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._running = True
        self._server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,  # 10MB max frame
            ping_interval=20,
            ping_timeout=10,
        )
        logger.info(f"Stream server listening on ws://{self.host}:{self.port}")
        logger.info(f"  Feed endpoint:      ws://host:{self.port}/feed")
        logger.info(f"  Stream endpoint:    ws://host:{self.port}/stream")
        logger.info(f"  Dashboard endpoint: ws://host:{self.port}/dashboard")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        async with self._lock:
            for client in self._clients.values():
                await client.ws.close()
            self._clients.clear()

    async def broadcast_frame(self, frame: Frame) -> None:
        """Send a frame to appropriate clients based on frame type and client role."""
        if not self._clients:
            return

        data = frame.pack()
        is_dashboard_only = frame.type in DASHBOARD_ONLY_TYPES
        disconnected = []

        async with self._lock:
            for client_id, client in self._clients.items():
                # Dashboard-only frames skip STREAM and FEED clients
                if is_dashboard_only and client.role != ClientRole.DASHBOARD:
                    continue

                try:
                    await client.ws.send(data)
                    client.frames_sent += 1
                except websockets.ConnectionClosed:
                    disconnected.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending to {client_id}: {e}")
                    disconnected.append(client_id)

            for cid in disconnected:
                del self._clients[cid]
                logger.info(f"Client disconnected: {cid}")

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def stream_client_count(self) -> int:
        return sum(1 for c in self._clients.values() if c.role == ClientRole.STREAM)

    @property
    def dashboard_client_count(self) -> int:
        return sum(1 for c in self._clients.values() if c.role == ClientRole.DASHBOARD)

    @property
    def feed_client_count(self) -> int:
        return sum(1 for c in self._clients.values() if c.role == ClientRole.FEED)

    @property
    def has_feed(self) -> bool:
        return self.feed_client_count > 0

    @property
    def feed_source(self) -> str | None:
        """Get the address of the active feed client."""
        for c in self._clients.values():
            if c.role == ClientRole.FEED:
                return c.address
        return None

    @property
    def clients_info(self) -> list[dict]:
        return [
            {
                "id": c.id,
                "address": c.address,
                "role": c.role.value,
                "frames_sent": c.frames_sent,
                "frames_received": c.frames_received,
            }
            for c in self._clients.values()
        ]

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        """Handle a new WebSocket connection."""
        client_id = f"{ws.remote_address[0]}:{ws.remote_address[1]}"

        # Determine role from request path
        path = "/stream"
        if hasattr(ws, "request") and hasattr(ws.request, "path"):
            path = ws.request.path
        elif hasattr(ws, "path"):
            path = ws.path

        if "/dashboard" in path:
            role = ClientRole.DASHBOARD
        elif "/feed" in path:
            role = ClientRole.FEED
        else:
            role = ClientRole.STREAM

        # Only allow one FEED client at a time
        if role == ClientRole.FEED and self.has_feed:
            await ws.close(1013, "Another feed client is already connected")
            logger.warning(f"Rejected feed client {client_id}: feed slot occupied")
            return

        # Check max clients for stream connections
        if role == ClientRole.STREAM and self.stream_client_count >= self.max_clients:
            await ws.close(1013, "Max clients reached")
            logger.warning(f"Rejected stream client {client_id}: max clients reached")
            return

        client = ConnectedClient(
            ws=ws, id=client_id, address=ws.remote_address[0], role=role
        )
        async with self._lock:
            self._clients[client_id] = client
        logger.info(f"Client connected: {client_id} (role={role.value})")

        try:
            async for message in ws:
                if isinstance(message, str):
                    await self._handle_control(client, message)
                elif isinstance(message, bytes) and client.role == ClientRole.FEED:
                    await self._handle_raw_frame(client, message)
        except websockets.ConnectionClosed:
            pass
        finally:
            async with self._lock:
                self._clients.pop(client_id, None)
            logger.info(f"Client disconnected: {client_id} (role={role.value})")

    async def _handle_control(self, client: ConnectedClient, message: str) -> None:
        """Handle a text control message from a client."""
        try:
            msg = json.loads(message)
            msg_type = msg.get("type")
            logger.debug(f"Control from {client.id}: {msg_type}")
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from {client.id}")

    async def _handle_raw_frame(self, client: ConnectedClient, data: bytes) -> None:
        """Handle a raw binary frame from a FEED client."""
        if len(data) < HEADER_SIZE:
            return

        try:
            frame = Frame.unpack(data)
        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid frame from {client.id}: {e}")
            return

        client.frames_received += 1

        if frame.type == FrameType.RAW_VIDEO and self.raw_video_queue:
            # Drop oldest if queue is full (non-blocking)
            try:
                self.raw_video_queue.put_nowait(frame.payload)
            except asyncio.QueueFull:
                try:
                    self.raw_video_queue.get_nowait()  # Drop oldest
                except asyncio.QueueEmpty:
                    pass
                self.raw_video_queue.put_nowait(frame.payload)

        elif frame.type == FrameType.RAW_AUDIO and self.raw_audio_queue:
            try:
                self.raw_audio_queue.put_nowait(frame.payload)
            except asyncio.QueueFull:
                try:
                    self.raw_audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self.raw_audio_queue.put_nowait(frame.payload)
