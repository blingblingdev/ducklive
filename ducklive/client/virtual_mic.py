"""Virtual microphone output — routes processed audio to a virtual audio device.

On macOS: uses BlackHole (https://existential.audio/blackhole/) as the
          virtual audio device. The user selects "BlackHole 2ch" as their
          microphone in Zoom/Teams.

On Windows: uses VB-Audio Virtual Cable.

This module writes PCM audio to the virtual audio device via PyAudio.
"""

from __future__ import annotations

import logging

import numpy as np

from ducklive.common.protocol import AUDIO_CHANNELS, AUDIO_CHUNK_SAMPLES, AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


class VirtualMicrophone:
    """Output audio to a virtual microphone device (BlackHole / VB-Cable)."""

    def __init__(
        self,
        device_name: str = "BlackHole",
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self._pa = None
        self._stream = None
        self._running = False
        self._device_index: int | None = None

    def start(self) -> None:
        """Start outputting to the virtual audio device."""
        import pyaudio

        self._pa = pyaudio.PyAudio()

        # Find the virtual audio device
        self._device_index = self._find_device()
        if self._device_index is None:
            available = self._list_output_devices()
            logger.error(
                f"Virtual audio device '{self.device_name}' not found.\n"
                f"Available output devices: {available}\n"
                f"On macOS, install BlackHole: brew install blackhole-2ch"
            )
            raise RuntimeError(f"Audio device not found: {self.device_name}")

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=self.sample_rate,
            output=True,
            output_device_index=self._device_index,
            frames_per_buffer=AUDIO_CHUNK_SAMPLES,
        )
        self._running = True

        device_info = self._pa.get_device_info_by_index(self._device_index)
        logger.info(f"Virtual microphone started: {device_info['name']}")

    def stop(self) -> None:
        """Stop the virtual microphone."""
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        logger.info("Virtual microphone stopped")

    def write(self, pcm_data: bytes) -> None:
        """Write PCM audio data to the virtual device."""
        if not self._stream or not self._running:
            return
        try:
            self._stream.write(pcm_data)
        except Exception as e:
            logger.warning(f"Audio write error: {e}")

    @property
    def is_running(self) -> bool:
        return self._running

    def _find_device(self) -> int | None:
        """Find the virtual audio device index by name."""
        if not self._pa:
            return None
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if (
                self.device_name.lower() in info["name"].lower()
                and info["maxOutputChannels"] > 0
            ):
                return i
        return None

    def _list_output_devices(self) -> list[str]:
        """List all available output devices."""
        if not self._pa:
            return []
        devices = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                devices.append(f"[{i}] {info['name']}")
        return devices
