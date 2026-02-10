"""Voice change engine using RVC (Retrieval-based Voice Conversion).

Pipeline:
    1. Receive PCM audio chunks from microphone (s16le mono 16kHz)
    2. Buffer to minimum processing window (100ms for quality)
    3. Extract HuBERT features from audio
    4. Extract pitch using RMVPE
    5. Apply pitch shift
    6. Run RVC synthesis (TextEncoder + Flow + NSF-HiFi-GAN)
    7. Resample back to 16kHz if model uses different sample rate
    8. Output converted PCM audio

Model architecture (RVC v2 with f0):
    - HuBERT: fairseq checkpoint (hubert_base.pt) - 768-dim feature extraction
    - RMVPE: U-Net + BiGRU (rmvpe.pt) - 360-bin pitch estimation
    - RVC Synthesizer: TextEncoder (enc_p) + ResidualCouplingFlow (flow) +
      NSF-HiFi-GAN Generator (dec) - voice conversion with pitch control
"""

from __future__ import annotations

import logging
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _detect_torch_device() -> str:
    """Auto-detect the best available torch device.

    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed -- voice change will use CPU")
        return "cpu"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"Detected NVIDIA GPU: {name} -- using CUDA")
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Detected Apple Silicon -- using MPS")
        return "mps"

    logger.warning("No GPU detected -- voice change will run on CPU (slower)")
    return "cpu"


def _ensure_fairseq_stub() -> None:
    """Install a minimal fairseq stub so torch.load can unpickle hubert_base.pt.

    The hubert checkpoint only references fairseq.data.dictionary.Dictionary
    during unpickling. We provide a lightweight stand-in so the checkpoint
    loads without the full fairseq package installed.
    """
    if "fairseq" in sys.modules:
        return

    fairseq_mod = types.ModuleType("fairseq")
    fairseq_data = types.ModuleType("fairseq.data")
    fairseq_dict = types.ModuleType("fairseq.data.dictionary")

    class _StubDictionary:
        """Minimal stand-in for fairseq.data.dictionary.Dictionary."""

        def __init__(self, *args, **kwargs):
            pass

    fairseq_dict.Dictionary = _StubDictionary
    fairseq_data.dictionary = fairseq_dict
    fairseq_mod.data = fairseq_data

    sys.modules["fairseq"] = fairseq_mod
    sys.modules["fairseq.data"] = fairseq_data
    sys.modules["fairseq.data.dictionary"] = fairseq_dict


# ---------------------------------------------------------------------------
# HuBERT feature extractor (lightweight reimplementation)
# Architecture: CNN feature extractor -> Transformer encoder (12 layers)
# Input: raw waveform at 16kHz
# Output: 768-dim features at ~50Hz (320 samples per frame)
# ---------------------------------------------------------------------------


class _HuBERTModel:
    """Lightweight HuBERT inference using raw PyTorch (no fairseq runtime)."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        import torch
        import torch.nn.functional as F  # noqa: N812

        self._device = device
        self._F = F

        _ensure_fairseq_stub()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]["model"]
        weights = ckpt["model"]

        self._embed_dim = cfg["encoder_embed_dim"]  # 768
        self._num_heads = cfg["encoder_attention_heads"]  # 12
        self._num_layers = cfg["encoder_layers"]  # 12
        self._ffn_dim = cfg["encoder_ffn_embed_dim"]  # 3072

        # Parse CNN feature extractor config
        # Format: "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"
        self._cnn_layers = self._build_cnn(weights, device)
        self._layer_norm = self._load_layer_norm(
            weights, "layer_norm", 512, device
        )
        self._post_extract_proj = self._load_linear(
            weights, "post_extract_proj", device
        )
        self._pos_conv = self._load_pos_conv(weights, device)
        self._encoder_layer_norm = self._load_layer_norm(
            weights, "encoder.layer_norm", self._embed_dim, device
        )
        self._encoder_layers = []
        for i in range(self._num_layers):
            layer = self._load_transformer_layer(weights, i, device)
            self._encoder_layers.append(layer)

    def _load_param(self, weights, key, device):
        import torch

        return weights[key].float().to(device)

    def _load_layer_norm(self, weights, prefix, dim, device):
        w = self._load_param(weights, f"{prefix}.weight", device)
        b = self._load_param(weights, f"{prefix}.bias", device)
        return (w, b)

    def _load_linear(self, weights, prefix, device):
        w = self._load_param(weights, f"{prefix}.weight", device)
        b = self._load_param(weights, f"{prefix}.bias", device)
        return (w, b)

    def _build_cnn(self, weights, device):
        """Build CNN feature extractor layers.

        Architecture: 7 conv layers
          Layer 0: Conv1d(1, 512, 10, stride=5) + GroupNorm
          Layers 1-4: Conv1d(512, 512, 3, stride=2) + GELU
          Layers 5-6: Conv1d(512, 512, 2, stride=2) + GELU
        """
        layers = []
        for i in range(7):
            conv_w = self._load_param(
                weights, f"feature_extractor.conv_layers.{i}.0.weight", device
            )
            # Layer 0 has GroupNorm (weight + bias at index 2)
            if i == 0:
                gn_w = self._load_param(
                    weights, f"feature_extractor.conv_layers.0.2.weight", device
                )
                gn_b = self._load_param(
                    weights, f"feature_extractor.conv_layers.0.2.bias", device
                )
                layers.append(("conv_gn", conv_w, gn_w, gn_b))
            else:
                layers.append(("conv_gelu", conv_w))
        return layers

    def _load_pos_conv(self, weights, device):
        """Load the positional convolutional embedding.

        Uses weight-normalized Conv1d(768, 768, 128, padding=64, groups=16).
        """
        weight_g = self._load_param(weights, "encoder.pos_conv.0.weight_g", device)
        weight_v = self._load_param(weights, "encoder.pos_conv.0.weight_v", device)
        bias = self._load_param(weights, "encoder.pos_conv.0.bias", device)
        return (weight_g, weight_v, bias)

    def _load_transformer_layer(self, weights, layer_idx, device):
        prefix = f"encoder.layers.{layer_idx}"
        return {
            "self_attn_q": self._load_linear(weights, f"{prefix}.self_attn.q_proj", device),
            "self_attn_k": self._load_linear(weights, f"{prefix}.self_attn.k_proj", device),
            "self_attn_v": self._load_linear(weights, f"{prefix}.self_attn.v_proj", device),
            "self_attn_out": self._load_linear(weights, f"{prefix}.self_attn.out_proj", device),
            "self_attn_ln": self._load_layer_norm(
                weights, f"{prefix}.self_attn_layer_norm", self._embed_dim, device
            ),
            "fc1": self._load_linear(weights, f"{prefix}.fc1", device),
            "fc2": self._load_linear(weights, f"{prefix}.fc2", device),
            "final_ln": self._load_layer_norm(
                weights, f"{prefix}.final_layer_norm", self._embed_dim, device
            ),
        }

    def _apply_layer_norm(self, x, params, eps=1e-5):
        """Apply layer normalization: (x - mean) / std * gamma + beta."""
        gamma, beta = params
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return gamma * (x - mean) / (var + eps).sqrt() + beta

    def _apply_linear(self, x, params):
        w, b = params
        return self._F.linear(x, w, b)

    def _apply_weight_norm_conv1d(self, x, weight_g, weight_v, bias, stride=1, padding=0, groups=1):
        """Apply weight-normalized Conv1d."""
        import torch

        # Compute normalized weight: w = g * v / ||v||
        norm = weight_v.norm(dim=(1, 2), keepdim=True)
        weight = weight_g * weight_v / (norm + 1e-7)
        return self._F.conv1d(x, weight, bias, stride=stride, padding=padding, groups=groups)

    def _apply_cnn(self, waveform):
        """Run through CNN feature extractor.

        Input: (1, 1, T) waveform
        Output: (1, 512, T') feature sequence
        """
        import torch

        x = waveform
        for layer in self._cnn_layers:
            if layer[0] == "conv_gn":
                _, conv_w, gn_w, gn_b = layer
                x = self._F.conv1d(x, conv_w, stride=5 if x.shape[1] == 1 else None)
                # Determine stride from conv weight shape
                x = self._F.group_norm(x, 512, gn_w, gn_b)
                x = self._F.gelu(x)
            else:
                _, conv_w = layer
                stride = conv_w.shape[2]
                if stride == 3:
                    stride = 2
                x = self._F.conv1d(x, conv_w, stride=stride)
                x = self._F.gelu(x)
        return x

    def _apply_pos_conv(self, x):
        """Apply positional convolutional embedding.

        Input/Output: (B, T, C) where C=768
        """
        # Transpose to (B, C, T) for conv1d
        x_t = x.transpose(1, 2)
        weight_g, weight_v, bias = self._pos_conv
        # Conv1d with padding=64, groups=16 -> remove 1 extra on right
        out = self._apply_weight_norm_conv1d(x_t, weight_g, weight_v, bias, padding=64, groups=16)
        # Remove extra sample from right side (kernel_size=128, padding=64 -> 1 extra)
        out = out[:, :, :x_t.shape[2]]
        out = self._F.gelu(out)
        return out.transpose(1, 2)

    def _apply_self_attention(self, x, layer_params):
        """Multi-head self-attention.

        Input: (B, T, C) where C=768
        Output: (B, T, C)
        """
        import torch

        B, T, C = x.shape
        head_dim = C // self._num_heads

        q = self._apply_linear(x, layer_params["self_attn_q"])
        k = self._apply_linear(x, layer_params["self_attn_k"])
        v = self._apply_linear(x, layer_params["self_attn_v"])

        # Reshape to (B, heads, T, head_dim)
        q = q.view(B, T, self._num_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, self._num_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, self._num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = self._F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self._apply_linear(out, layer_params["self_attn_out"])
        return out

    def _apply_transformer_layer(self, x, layer_params):
        """Post-LN transformer layer: x + SA(LN(x)), x + FFN(LN(x))."""
        # Self-attention with residual
        residual = x
        x = self._apply_self_attention(x, layer_params)
        x = residual + x
        x = self._apply_layer_norm(x, layer_params["self_attn_ln"])

        # FFN with residual
        residual = x
        h = self._apply_linear(x, layer_params["fc1"])
        h = self._F.gelu(h)
        h = self._apply_linear(h, layer_params["fc2"])
        x = residual + h
        x = self._apply_layer_norm(x, layer_params["final_ln"])
        return x

    @staticmethod
    def _get_cnn_strides():
        """Return stride for each CNN layer: [5, 2, 2, 2, 2, 2, 2] -> total stride = 320."""
        return [5, 2, 2, 2, 2, 2, 2]

    def extract_features(self, waveform_16k):
        """Extract HuBERT features from 16kHz waveform.

        Args:
            waveform_16k: torch.Tensor of shape (T,) or (1, T), float32, 16kHz

        Returns:
            features: torch.Tensor of shape (1, T', 768) where T' = T // 320
        """
        import torch

        if waveform_16k.dim() == 1:
            waveform_16k = waveform_16k.unsqueeze(0)
        if waveform_16k.dim() == 2:
            waveform_16k = waveform_16k.unsqueeze(0)  # (1, 1, T)

        with torch.no_grad():
            # CNN feature extraction -> (1, 512, T')
            features = self._apply_cnn(waveform_16k)

            # Transpose to (1, T', 512)
            features = features.transpose(1, 2)

            # Layer norm on CNN output
            features = self._apply_layer_norm(features, self._layer_norm)

            # Project to embed_dim: (1, T', 512) -> (1, T', 768)
            features = self._apply_linear(features, self._post_extract_proj)

            # Add positional embedding
            pos_emb = self._apply_pos_conv(features)
            features = features + pos_emb

            # Run through transformer layers
            for layer_params in self._encoder_layers:
                features = self._apply_transformer_layer(features, layer_params)

            # Final layer norm
            features = self._apply_layer_norm(features, self._encoder_layer_norm)

        return features  # (1, T', 768)


# ---------------------------------------------------------------------------
# RMVPE pitch extractor
# Architecture: U-Net (5 encoder + 5 decoder + intermediate) -> BiGRU -> FC
# Input: mel spectrogram (n_mels=128, 16kHz audio)
# Output: pitch in Hz per frame (10ms hop)
# ---------------------------------------------------------------------------


class _RMVPEModel:
    """RMVPE (Robust Model for Voice Pitch Estimation) using raw PyTorch."""

    # RMVPE operates on 16kHz audio with 160-sample hop (10ms)
    SAMPLE_RATE = 16000
    HOP_LENGTH = 160
    N_MELS = 128
    N_FFT = 1024
    WIN_LENGTH = 1024
    N_PITCH_BINS = 360  # pitch bins covering 32.7 Hz to ~1975 Hz
    PITCH_BINS_PER_OCTAVE = 60  # 5 cents per bin
    FMIN = 32.70  # C1

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        import torch
        import torch.nn as nn

        self._device = device
        weights = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Build the U-Net model
        self._model = _RMVPENet()
        self._model.load_state_dict(weights)
        self._model.to(device)
        self._model.eval()

        # Pre-compute mel filterbank
        self._mel_basis = self._build_mel_basis(device)

        # Pre-compute pitch bin center frequencies
        self._pitch_freqs = torch.tensor(
            [self.FMIN * 2 ** (i / self.PITCH_BINS_PER_OCTAVE) for i in range(self.N_PITCH_BINS)],
            dtype=torch.float32,
            device=device,
        )

    def _build_mel_basis(self, device):
        """Build mel filterbank matrix for spectrogram computation."""
        import torch

        n_fft = self.N_FFT
        n_mels = self.N_MELS
        sr = self.SAMPLE_RATE

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595.0 * math.log10(1.0 + hz / 700.0)

        fmin_mel = hz_to_mel(0)
        fmax_mel = hz_to_mel(sr / 2)
        mel_points = torch.linspace(fmin_mel, fmax_mel, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        # FFT bin frequencies
        fft_bins = torch.linspace(0, sr / 2, n_fft // 2 + 1)

        # Build filterbank
        filterbank = torch.zeros(n_mels, n_fft // 2 + 1)
        for m in range(n_mels):
            f_left = hz_points[m]
            f_center = hz_points[m + 1]
            f_right = hz_points[m + 2]

            # Rising slope
            mask_left = (fft_bins >= f_left) & (fft_bins <= f_center)
            if f_center > f_left:
                filterbank[m][mask_left] = (fft_bins[mask_left] - f_left) / (f_center - f_left)

            # Falling slope
            mask_right = (fft_bins >= f_center) & (fft_bins <= f_right)
            if f_right > f_center:
                filterbank[m][mask_right] = (f_right - fft_bins[mask_right]) / (f_right - f_center)

        return filterbank.to(device)

    def _compute_mel(self, waveform):
        """Compute log mel spectrogram.

        Args:
            waveform: (T,) float32 tensor at 16kHz

        Returns:
            mel: (1, 1, n_mels, T') tensor
        """
        import torch

        # STFT
        window = torch.hann_window(self.WIN_LENGTH, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            win_length=self.WIN_LENGTH,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitude = stft.abs()  # (n_fft//2+1, T')

        # Apply mel filterbank
        mel = torch.matmul(self._mel_basis, magnitude)  # (n_mels, T')

        # Log scale
        mel = torch.log(torch.clamp(mel, min=1e-7))

        # Add batch and channel dims: (1, 1, n_mels, T')
        return mel.unsqueeze(0).unsqueeze(0)

    def extract_pitch(self, waveform_16k):
        """Extract pitch (f0) from 16kHz waveform.

        Args:
            waveform_16k: torch.Tensor of shape (T,), float32, 16kHz

        Returns:
            f0: numpy array of pitch values in Hz, shape (T',)
                where T' = ceil(T / hop_length)
                Unvoiced frames have f0 = 0.
        """
        import torch

        with torch.no_grad():
            mel = self._compute_mel(waveform_16k)

            # Pad mel time dimension to multiple of 32 (required for 5 MaxPool2d(2) layers)
            orig_t = mel.shape[3]
            pad_t = (32 - orig_t % 32) % 32
            if pad_t > 0:
                mel = torch.nn.functional.pad(mel, (0, pad_t))

            # Run through RMVPE network -> (1, T', 360) probabilities
            pitch_probs = self._model(mel)

            # Trim padding frames from output
            if pad_t > 0:
                pitch_probs = pitch_probs[:, :orig_t, :]

            # Decode pitch from probability distribution
            f0 = self._decode_pitch(pitch_probs.squeeze(0))

        return f0.cpu().numpy()

    def _decode_pitch(self, probs):
        """Decode pitch from RMVPE probability output.

        Args:
            probs: (T', 360) tensor of pitch bin probabilities

        Returns:
            f0: (T',) tensor of pitch in Hz (0 = unvoiced)
        """
        import torch

        # Threshold for voiced/unvoiced decision
        max_probs, _ = probs.max(dim=-1)
        voiced_mask = max_probs > 0.3

        # Weighted average of nearby bins for sub-bin precision
        # Use top bins within a local window
        center_bins = probs.argmax(dim=-1)  # (T',)

        f0 = torch.zeros(probs.shape[0], device=probs.device)

        for t in range(probs.shape[0]):
            if not voiced_mask[t]:
                continue
            center = center_bins[t].item()
            # Take a window of +/- 4 bins around the peak
            lo = max(0, center - 4)
            hi = min(self.N_PITCH_BINS, center + 5)
            local_probs = probs[t, lo:hi]
            local_freqs = self._pitch_freqs[lo:hi]

            # Softmax-weighted average for precise frequency
            weights = torch.softmax(local_probs * 10.0, dim=0)
            f0[t] = (weights * local_freqs).sum()

        return f0


class _RMVPENet:
    """RMVPE network (U-Net + BiGRU + FC).

    Since we need to match the exact state_dict structure, we build this
    as a proper nn.Module.
    """

    def __new__(cls):
        """Build the RMVPE network as an nn.Module."""
        import torch
        import torch.nn as nn

        # Channel configuration for encoder/decoder
        # Encoder: 1 -> 16 -> 32 -> 64 -> 128 -> 256
        # Decoder: 512 -> 256 -> 128 -> 64 -> 32 -> 16
        enc_channels = [1, 16, 32, 64, 128, 256]
        dec_channels = [256, 128, 64, 32, 16]

        class ConvBlock(nn.Module):
            """Conv-BN-ReLU block with optional shortcut."""

            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
                if in_ch != out_ch:
                    self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
                else:
                    self.shortcut = nn.Identity()

            def forward(self, x):
                return nn.functional.relu(self.conv(x) + self.shortcut(x))

        class EncoderLayer(nn.Module):
            def __init__(self, in_ch, out_ch, n_blocks=4):
                super().__init__()
                blocks = [ConvBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(n_blocks)]
                self.conv = nn.ModuleList(blocks)
                self.pool = nn.MaxPool2d(2)

            def forward(self, x):
                for block in self.conv:
                    x = block(x)
                return self.pool(x), x  # pooled, skip

        class DecoderLayer(nn.Module):
            def __init__(self, in_ch, out_ch, n_blocks=4):
                super().__init__()
                # ConvTranspose2d without bias (matches state dict)
                self.conv1 = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
                # First block gets out_ch*2 input (cat with skip connection)
                blocks = [ConvBlock(out_ch * 2 if i == 0 else out_ch, out_ch)
                          for i in range(n_blocks)]
                self.conv2 = nn.ModuleList(blocks)

            def forward(self, x, skip):
                x = self.conv1(x)
                # Handle size mismatch from pooling
                if x.shape[2:] != skip.shape[2:]:
                    x = nn.functional.interpolate(x, size=skip.shape[2:])
                x = torch.cat([x, skip], dim=1)
                for block in self.conv2:
                    x = block(x)
                return x

        class IntermediateSubLayer(nn.Module):
            """A single intermediate sub-layer containing 4 ConvBlocks.

            State dict keys: intermediate.layers.{i}.conv.{j}.*
            """

            def __init__(self, in_ch, out_ch, n_blocks=4):
                super().__init__()
                blocks = [ConvBlock(in_ch if i == 0 else out_ch, out_ch)
                          for i in range(n_blocks)]
                self.conv = nn.ModuleList(blocks)

            def forward(self, x):
                for block in self.conv:
                    x = block(x)
                return x

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.bn = nn.BatchNorm2d(1)
                enc_layers = []
                for i in range(5):
                    enc_layers.append(EncoderLayer(enc_channels[i], enc_channels[i + 1]))
                self.encoder.layers = nn.ModuleList(enc_layers)

                # Intermediate: 4 sub-layers, each with 4 ConvBlocks
                # First sub-layer: 256 -> 512, rest: 512 -> 512
                self.intermediate = nn.Module()
                int_layers = []
                for i in range(4):
                    in_c = 256 if i == 0 else 512
                    int_layers.append(IntermediateSubLayer(in_c, 512))
                self.intermediate.layers = nn.ModuleList(int_layers)

                dec_layers = []
                for i in range(5):
                    in_c = 512 if i == 0 else dec_channels[i - 1]
                    out_c = dec_channels[i]
                    dec_layers.append(DecoderLayer(in_c, out_c))
                self.decoder = nn.Module()
                self.decoder.layers = nn.ModuleList(dec_layers)

            def forward(self, x):
                x = self.encoder.bn(x)
                skips = []
                for layer in self.encoder.layers:
                    x, skip = layer(x)
                    skips.append(skip)

                for sub_layer in self.intermediate.layers:
                    x = sub_layer(x)

                for i, layer in enumerate(self.decoder.layers):
                    x = layer(x, skips[-(i + 1)])
                return x

        class RMVPEFullNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = UNet()
                # CNN: reduce channels from 16 to 3 (for 3 octave ranges)
                self.cnn = nn.Conv2d(16, 3, 3, padding=1)
                # BiGRU: input = 3 * n_mels = 384, hidden = 256
                self.fc = nn.ModuleList([
                    nn.Module(),  # placeholder for GRU wrapper
                    nn.Linear(512, 360),
                ])
                # Set up GRU as a submodule
                self.fc[0].gru = nn.GRU(
                    input_size=384,
                    hidden_size=256,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )

            def forward(self, mel):
                # mel: (B, 1, 128, T')
                x = self.unet(mel)  # (B, 16, 128, T')
                x = self.cnn(x)  # (B, 3, 128, T')

                # Reshape for GRU: (B, T', 3*128)
                B, C, H, T = x.shape
                x = x.permute(0, 3, 1, 2).reshape(B, T, C * H)  # (B, T', 384)

                # BiGRU
                x, _ = self.fc[0].gru(x)  # (B, T', 512)

                # FC to pitch bins
                x = self.fc[1](x)  # (B, T', 360)

                return torch.sigmoid(x)

        return RMVPEFullNet()


# ---------------------------------------------------------------------------
# RVC v2 Synthesizer (TextEncoder + Flow + NSF-HiFi-GAN Generator)
# ---------------------------------------------------------------------------


class _RVCSynthesizer:
    """RVC v2 synthesis model.

    Architecture:
      - enc_p (TextEncoder): HuBERT features + pitch -> latent z
      - flow (ResidualCouplingBlock): transform latent distribution
      - dec (GeneratorNSF): generate waveform from z + pitch

    The model operates at its native sample rate (e.g., 40kHz for "40k" models).
    """

    def __new__(cls, config, f0_enabled=True, version="v2"):
        """Build and return the synthesizer nn.Module."""
        import torch
        import torch.nn as nn
        from torch.nn.utils.parametrizations import weight_norm

        (
            spec_channels, segment_size, inter_channels, hidden_channels,
            filter_channels, n_heads, n_layers, kernel_size, p_dropout,
            resblock, resblock_kernel_sizes, resblock_dilation_sizes,
            upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
            spk_embed_dim, gin_channels, sr
        ) = config

        # ------- Layer Norm used in RVC (with gamma/beta naming) -------
        class RVCLayerNorm(nn.Module):
            def __init__(self, channels, eps=1e-5):
                super().__init__()
                self.gamma = nn.Parameter(torch.ones(channels))
                self.beta = nn.Parameter(torch.zeros(channels))
                self.eps = eps

            def forward(self, x):
                # x: (B, C, T)
                mean = x.mean(dim=1, keepdim=True)
                var = x.var(dim=1, keepdim=True, unbiased=False)
                return self.gamma.unsqueeze(-1) * (x - mean) / (var + self.eps).sqrt() + self.beta.unsqueeze(-1)

        # ------- Multi-Head Attention (RVC style, 1D conv-based) -------
        class MultiHeadAttention(nn.Module):
            def __init__(self, channels, out_channels, n_heads, window_size=10):
                super().__init__()
                self.channels = channels
                self.out_channels = out_channels
                self.n_heads = n_heads
                self.k_channels = channels // n_heads
                self.window_size = window_size

                self.conv_q = nn.Conv1d(channels, channels, 1)
                self.conv_k = nn.Conv1d(channels, channels, 1)
                self.conv_v = nn.Conv1d(channels, channels, 1)
                self.conv_o = nn.Conv1d(channels, out_channels, 1)

                # Relative position embedding
                self.emb_rel_k = nn.Parameter(
                    torch.randn(1, 2 * window_size + 1, self.k_channels) * (self.k_channels ** -0.5)
                )
                self.emb_rel_v = nn.Parameter(
                    torch.randn(1, 2 * window_size + 1, self.k_channels) * (self.k_channels ** -0.5)
                )

            def forward(self, x):
                B, C, T = x.shape
                q = self.conv_q(x).view(B, self.n_heads, self.k_channels, T)
                k = self.conv_k(x).view(B, self.n_heads, self.k_channels, T)
                v = self.conv_v(x).view(B, self.n_heads, self.k_channels, T)

                # Attention scores: (B, heads, T, T)
                scale = self.k_channels ** -0.5
                scores = torch.matmul(q.transpose(-2, -1), k) * scale

                # Add relative position bias for keys
                # emb_rel_k: (1, 2w+1, k_ch) -- relative position embeddings
                # Build a (T, T) relative position bias matrix
                rel_k_bias = self._compute_relative_bias(q, self.emb_rel_k, T)
                scores = scores + rel_k_bias * scale

                attn = torch.softmax(scores, dim=-1)

                # Apply attention to values
                out = torch.matmul(attn, v.transpose(-2, -1))  # (B, heads, T, k_channels)

                # Add relative position for values
                rel_v_out = self._compute_relative_values(attn, self.emb_rel_v, T)
                out = out + rel_v_out

                out = out.transpose(-2, -1).contiguous().view(B, C, T)
                return self.conv_o(out)

            def _compute_relative_bias(self, q, emb, T):
                """Compute relative position attention bias.

                For each pair (i, j), compute: q[i] . emb_rel[clip(i-j+w, 0, 2w)]
                Returns: (B, heads, T, T) bias matrix

                Args:
                    q: (B, heads, k_channels, T)
                    emb: (1, 2w+1, k_channels)
                    T: sequence length
                """
                w = self.window_size
                # Build relative position index matrix
                # rel_idx[i,j] = i - j + w, clamped to [0, 2w]
                positions = torch.arange(T, device=q.device)
                rel_idx = positions.unsqueeze(1) - positions.unsqueeze(0) + w  # (T, T)
                rel_idx = rel_idx.clamp(0, 2 * w)

                # Gather embeddings: (T, T, k_channels)
                emb_flat = emb.squeeze(0)  # (2w+1, k_channels)
                rel_embs = emb_flat[rel_idx]  # (T, T, k_channels)

                # Compute bias: q[b, h, :, i] . rel_embs[i, j, :]
                # q: (B, heads, k_ch, T) -> (B, heads, T, k_ch)
                q_t = q.transpose(-2, -1)  # (B, heads, T, k_ch)
                # rel_embs: (T, T, k_ch) -> (1, 1, T, T, k_ch)
                rel_embs = rel_embs.unsqueeze(0).unsqueeze(0)
                # Dot product: (B, heads, T, k_ch) x (1, 1, T, T, k_ch) -> sum over k_ch
                bias = torch.einsum("bhtc,ijc->bhtj", q_t, emb_flat[rel_idx])

                return bias

            def _compute_relative_values(self, attn, emb, T):
                """Compute relative position contribution to values.

                For each position i, sum: attn[i,j] * emb_rel_v[clip(i-j+w, 0, 2w)]
                Returns: (B, heads, T, k_channels)

                Args:
                    attn: (B, heads, T, T) attention weights
                    emb: (1, 2w+1, k_channels) relative value embeddings
                    T: sequence length
                """
                w = self.window_size
                positions = torch.arange(T, device=attn.device)
                rel_idx = positions.unsqueeze(1) - positions.unsqueeze(0) + w  # (T, T)
                rel_idx = rel_idx.clamp(0, 2 * w)

                emb_flat = emb.squeeze(0)  # (2w+1, k_channels)
                rel_embs = emb_flat[rel_idx]  # (T, T, k_channels)

                # attn: (B, heads, T, T), rel_embs: (T, T, k_ch)
                # output[b,h,i,c] = sum_j attn[b,h,i,j] * rel_embs[i,j,c]
                out = torch.einsum("bhtj,tjc->bhtc", attn, rel_embs)

                return out

        # ------- FFN for Transformer Encoder -------
        class FFN(nn.Module):
            def __init__(self, in_channels, out_channels, filter_channels, kernel_size):
                super().__init__()
                self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
                self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)

            def forward(self, x):
                x = self.conv_1(x)
                x = torch.relu(x)
                x = self.conv_2(x)
                return x

        # ------- Transformer Encoder (used in enc_p) -------
        class Encoder(nn.Module):
            def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size):
                super().__init__()
                self.attn_layers = nn.ModuleList()
                self.norm_layers_1 = nn.ModuleList()
                self.ffn_layers = nn.ModuleList()
                self.norm_layers_2 = nn.ModuleList()

                for _ in range(n_layers):
                    self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads))
                    self.norm_layers_1.append(RVCLayerNorm(hidden_channels))
                    self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size))
                    self.norm_layers_2.append(RVCLayerNorm(hidden_channels))

            def forward(self, x):
                # x: (B, C, T)
                for attn, norm1, ffn, norm2 in zip(
                    self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
                ):
                    residual = x
                    x = attn(x)
                    x = norm1(x + residual)
                    residual = x
                    x = ffn(x)
                    x = norm2(x + residual)
                return x

        # ------- TextEncoder (enc_p) -------
        class TextEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb_phone = nn.Linear(768, hidden_channels)
                if f0_enabled:
                    self.emb_pitch = nn.Embedding(256, hidden_channels)
                self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size)
                # Project to mean and log-variance
                self.proj = nn.Conv1d(hidden_channels, inter_channels * 2, 1)

            def forward(self, phone, pitch=None):
                # phone: (B, T, 768) HuBERT features
                # pitch: (B, T) quantized pitch indices
                x = self.emb_phone(phone)  # (B, T, hidden)
                if f0_enabled and pitch is not None:
                    x = x + self.emb_pitch(pitch)  # (B, T, hidden)
                x = x.transpose(1, 2)  # (B, hidden, T)
                x = self.encoder(x)
                stats = self.proj(x)  # (B, inter*2, T)
                m, logs = stats.split(inter_channels, dim=1)
                return m, logs

        # ------- WaveNet layer for Flow -------
        class WN(nn.Module):
            """WaveNet-like module used in residual coupling layers."""

            def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels):
                super().__init__()
                self.hidden_channels = hidden_channels
                self.n_layers = n_layers

                self.in_layers = nn.ModuleList()
                self.res_skip_layers = nn.ModuleList()
                self.cond_layer = weight_norm(nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1))

                for i in range(n_layers):
                    dilation = dilation_rate ** i
                    padding = (kernel_size * dilation - dilation) // 2
                    self.in_layers.append(
                        weight_norm(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                              dilation=dilation, padding=padding))
                    )
                    if i < n_layers - 1:
                        res_skip_ch = 2 * hidden_channels
                    else:
                        res_skip_ch = hidden_channels
                    self.res_skip_layers.append(
                        weight_norm(nn.Conv1d(hidden_channels, res_skip_ch, 1))
                    )

            def forward(self, x, g=None):
                # x: (B, hidden, T)
                # g: (B, gin_channels, 1) speaker embedding
                output = torch.zeros_like(x)

                if g is not None:
                    g = self.cond_layer(g)

                for i in range(self.n_layers):
                    x_in = self.in_layers[i](x)
                    if g is not None:
                        cond = g[:, i * 2 * self.hidden_channels:(i + 1) * 2 * self.hidden_channels, :]
                        x_in = x_in + cond

                    # Gated activation
                    t_act = torch.tanh(x_in[:, :self.hidden_channels, :])
                    s_act = torch.sigmoid(x_in[:, self.hidden_channels:, :])
                    acts = t_act * s_act

                    res_skip = self.res_skip_layers[i](acts)
                    if i < self.n_layers - 1:
                        x = x + res_skip[:, :self.hidden_channels, :]
                        output = output + res_skip[:, self.hidden_channels:, :]
                    else:
                        output = output + res_skip

                return output

        # ------- Residual Coupling Layer -------
        class ResidualCouplingLayer(nn.Module):
            def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels):
                super().__init__()
                self.half_channels = channels // 2
                self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
                self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
                self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
                self.post.weight.data.zero_()
                self.post.bias.data.zero_()

            def forward(self, x, g=None, reverse=False):
                x0, x1 = x.split(self.half_channels, dim=1)
                h = self.pre(x0)
                h = self.enc(h, g=g)
                stats = self.post(h)
                m = stats
                if not reverse:
                    x1 = m + x1
                else:
                    x1 = x1 - m
                return torch.cat([x0, x1], dim=1)

        # ------- Flip Layer -------
        class Flip(nn.Module):
            def forward(self, x, *args, **kwargs):
                return torch.flip(x, dims=[1])

        # ------- ResidualCouplingBlock (flow) -------
        class ResidualCouplingBlock(nn.Module):
            def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_flow_layers, gin_channels):
                super().__init__()
                self.flows = nn.ModuleList()
                for i in range(n_flow_layers):
                    self.flows.append(
                        ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, 3, gin_channels)
                    )
                    self.flows.append(Flip())

            def forward(self, x, g=None, reverse=False):
                if not reverse:
                    for flow in self.flows:
                        x = flow(x, g=g, reverse=reverse)
                else:
                    for flow in reversed(self.flows):
                        x = flow(x, g=g, reverse=reverse)
                return x

        # ------- Source Module (NSF) -------
        class SourceModule(nn.Module):
            """Neural source filter: generates excitation signal from f0."""

            def __init__(self, sampling_rate):
                super().__init__()
                self.sampling_rate = sampling_rate
                self.l_linear = nn.Linear(1, 1)

            def forward(self, f0, upsample_factor):
                # f0: (B, 1, T_pitch)
                # Generate sine wave excitation
                B = f0.shape[0]
                T = f0.shape[2] * upsample_factor

                # Upsample f0 to target length
                f0_up = torch.nn.functional.interpolate(
                    f0, size=T, mode="nearest"
                )  # (B, 1, T)

                # Generate phase
                omega = 2 * math.pi * f0_up / self.sampling_rate
                phase = torch.cumsum(omega, dim=2)

                # Sine excitation (voiced) + noise (unvoiced)
                sine = torch.sin(phase)
                voiced = (f0_up > 0).float()
                sine = sine * voiced

                # Apply learned transformation
                sine = self.l_linear(sine.transpose(1, 2)).transpose(1, 2)

                return sine  # (B, 1, T)

        # ------- NSF-HiFi-GAN Generator (dec) -------
        class GeneratorNSF(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_kernels = len(resblock_kernel_sizes)
                self.num_upsamples = len(upsample_rates)
                self.upsample_rates = upsample_rates

                self.m_source = SourceModule(sr)
                self.conv_pre = nn.Conv1d(inter_channels, upsample_initial_channel, 7, padding=3)

                # Speaker conditioning
                self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

                # Upsample layers
                self.ups = nn.ModuleList()
                self.noise_convs = nn.ModuleList()

                total_stride = 1
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
                    ch = upsample_initial_channel // (2 ** (i + 1))
                    self.ups.append(
                        weight_norm(nn.ConvTranspose1d(
                            upsample_initial_channel // (2 ** i),
                            ch,
                            k,
                            stride=u,
                            padding=(k - u) // 2,
                        ))
                    )
                    total_stride *= u
                    # Noise convolution for source module
                    if i + 1 < len(upsample_rates):
                        stride_prod = 1
                        for s in upsample_rates[i + 1:]:
                            stride_prod *= s
                        self.noise_convs.append(
                            nn.Conv1d(1, ch, kernel_size=stride_prod * 2, stride=stride_prod, padding=stride_prod // 2)
                        )
                    else:
                        self.noise_convs.append(nn.Conv1d(1, ch, kernel_size=1))

                # Residual blocks
                self.resblocks = nn.ModuleList()
                for i in range(len(self.ups)):
                    ch = upsample_initial_channel // (2 ** (i + 1))
                    for j, (k_rb, d_rb) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                        self.resblocks.append(self._make_resblock(ch, k_rb, d_rb))

                self.conv_post = nn.Conv1d(ch, 1, 7, padding=3, bias=False)

            def _make_resblock(self, channels, kernel_size, dilation):
                """Create a ResBlock1 (used in HiFi-GAN)."""
                layers1 = nn.ModuleList()
                layers2 = nn.ModuleList()
                for d in dilation:
                    padding = (kernel_size * d - d) // 2
                    layers1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=padding)))
                    layers2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)))

                block = nn.Module()
                block.convs1 = layers1
                block.convs2 = layers2
                return block

            def forward(self, z, f0, g=None):
                # z: (B, inter_channels, T)
                # f0: (B, 1, T_pitch)
                # g: (B, gin_channels, 1)

                # Generate source excitation
                total_upsample = 1
                for u in self.upsample_rates:
                    total_upsample *= u
                har_source = self.m_source(f0, total_upsample)

                x = self.conv_pre(z)
                if g is not None:
                    x = x + self.cond(g)

                for i in range(self.num_upsamples):
                    x = torch.nn.functional.leaky_relu(x, 0.1)
                    x = self.ups[i](x)

                    # Add noise from source
                    x_source = self.noise_convs[i](har_source)
                    if x_source.shape[2] != x.shape[2]:
                        x_source = torch.nn.functional.interpolate(x_source, size=x.shape[2])
                    x = x + x_source

                    # Apply residual blocks
                    xs = None
                    for j in range(self.num_kernels):
                        rb = self.resblocks[i * self.num_kernels + j]
                        x_rb = x
                        for c1, c2 in zip(rb.convs1, rb.convs2):
                            x_rb = torch.nn.functional.leaky_relu(x_rb, 0.1)
                            x_rb = c1(x_rb)
                            x_rb = torch.nn.functional.leaky_relu(x_rb, 0.1)
                            x_rb = c2(x_rb)
                        if xs is None:
                            xs = x_rb
                        else:
                            xs = xs + x_rb
                    x = xs / self.num_kernels

                x = torch.nn.functional.leaky_relu(x, 0.1)
                x = self.conv_post(x)
                x = torch.tanh(x)
                return x

        # ------- Full Synthesizer -------
        class SynthesizerTrnMs256NSFsid(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc_p = TextEncoder()
                self.flow = ResidualCouplingBlock(
                    inter_channels, hidden_channels, 5, 1, 4, gin_channels
                )
                self.dec = GeneratorNSF()
                self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

            def forward(self, phone, phone_lengths, pitch, pitchf, sid):
                # For inference only
                pass

            def infer(self, phone, pitch, pitchf, sid):
                """Run inference: phone features + pitch -> waveform.

                Args:
                    phone: (B, T, 768) HuBERT features
                    pitch: (B, T) quantized pitch indices (0-255)
                    pitchf: (B, 1, T) continuous pitch in Hz
                    sid: (B,) speaker ID (usually 0)

                Returns:
                    audio: (B, 1, T_audio) generated waveform
                """
                g = self.emb_g(sid).unsqueeze(-1)  # (B, gin_channels, 1)

                m_p, logs_p = self.enc_p(phone, pitch)

                # Inverse flow to get latent z
                z_p = m_p  # Use mean (no sampling for inference)
                z = self.flow(z_p, g=g, reverse=True)

                # Generate audio
                audio = self.dec(z, pitchf, g=g)
                return audio

        return SynthesizerTrnMs256NSFsid()


# ---------------------------------------------------------------------------
# Voice Change Engine (main class)
# ---------------------------------------------------------------------------


class VoiceChangeEngine:
    """Real-time voice conversion engine powered by RVC."""

    def __init__(self, model_dir: Path | str = "models"):
        self.model_dir = Path(model_dir)
        self._model = None
        self._model_path: str = ""
        self._loaded = False
        self._avg_latency_ms = 0.0
        self._latency_alpha = 0.1
        self._pitch_shift: int = 0  # semitones
        self._sample_rate: int = 16000

        # Audio buffer for accumulating small chunks
        self._buffer = np.array([], dtype=np.int16)
        # RMVPE needs at least 32 mel frames after 5 MaxPool2d(2) layers.
        # 32 frames * 160 hop = 5120 samples at 16kHz = 320ms.
        # Use 6400 samples (400ms) for safety margin.
        self._min_samples = 6400

        # Internal model references (loaded in load())
        self._hubert: _HuBERTModel | None = None
        self._rmvpe: _RMVPEModel | None = None
        self._device: str = "cpu"

        # RVC model metadata
        self._rvc_config = None
        self._rvc_sr: int = 40000  # native sample rate of RVC model
        self._rvc_f0: bool = True  # whether model supports f0

        # FAISS index for feature retrieval (optional)
        self._faiss_index = None
        self._index_rate: float = 0.75  # blend ratio for index retrieval

    def load(self, device: str | None = None) -> None:
        """Load HuBERT and RMVPE models.

        Args:
            device: torch device. Auto-detected if None.
                    (cuda:0, mps, cpu)
        """
        if device is None:
            device = _detect_torch_device()
        self._device = device
        logger.info(f"Voice change engine using device: {device}")

        # Load HuBERT feature extractor
        hubert_path = self.model_dir / "hubert_base.pt"
        if not hubert_path.exists():
            raise FileNotFoundError(
                f"HuBERT model not found: {hubert_path}. "
                "Download hubert_base.pt to the models directory."
            )
        logger.info(f"Loading HuBERT model from {hubert_path}...")
        self._hubert = _HuBERTModel(str(hubert_path), device)
        logger.info("HuBERT model loaded successfully")

        # Load RMVPE pitch extractor
        rmvpe_path = self.model_dir / "rmvpe.pt"
        if not rmvpe_path.exists():
            raise FileNotFoundError(
                f"RMVPE model not found: {rmvpe_path}. "
                "Download rmvpe.pt to the models directory."
            )
        logger.info(f"Loading RMVPE model from {rmvpe_path}...")
        self._rmvpe = _RMVPEModel(str(rmvpe_path), device)
        logger.info("RMVPE model loaded successfully")

        self._loaded = True
        logger.info("Voice change engine ready (no voice model set yet)")

    def set_voice_model(self, model_path: str | Path) -> bool:
        """Load a specific RVC voice model (.pth file).

        Returns True if model loaded successfully.
        """
        import torch

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Voice model not found: {model_path}")

        logger.info(f"Loading RVC voice model: {model_path}")

        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)

        # Parse model metadata
        config = ckpt["config"]
        self._rvc_config = config
        self._rvc_f0 = bool(ckpt.get("f0", 1))
        version = ckpt.get("version", "v2")

        # Parse sample rate from config or metadata
        sr_str = ckpt.get("sr", "40k")
        if isinstance(sr_str, str):
            self._rvc_sr = int(sr_str.replace("k", "000"))
        else:
            self._rvc_sr = int(config[-1]) if isinstance(config[-1], int) else 40000

        logger.info(
            f"RVC model: version={version}, sr={self._rvc_sr}, f0={self._rvc_f0}, "
            f"config_len={len(config)}"
        )

        # Build and load the synthesizer model
        model = _RVCSynthesizer(config, f0_enabled=self._rvc_f0, version=version)
        model.load_state_dict(ckpt["weight"], strict=False)
        model.to(self._device)
        model.eval()
        self._model = model

        # Try to load FAISS index if available alongside the model
        self._load_faiss_index(model_path)

        self._model_path = str(model_path)
        logger.info(f"RVC voice model loaded: {model_path.stem}")
        return True

    def _load_faiss_index(self, model_path: Path) -> None:
        """Try to load a FAISS index file for feature retrieval.

        Looks for .index files in the same directory as the model.
        """
        self._faiss_index = None

        try:
            import faiss
        except ImportError:
            logger.debug("FAISS not available, skipping index retrieval")
            return

        # Search for matching .index file
        model_dir = model_path.parent
        model_stem = model_path.stem
        index_files = list(model_dir.glob("*.index"))

        # Try to find an index file that matches the model name
        matching = [f for f in index_files if model_stem.lower() in f.stem.lower()]
        if not matching:
            matching = index_files  # Use any available index

        if matching:
            index_path = matching[0]
            try:
                self._faiss_index = faiss.read_index(str(index_path))
                logger.info(f"Loaded FAISS index: {index_path.name} ({self._faiss_index.ntotal} vectors)")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index {index_path}: {e}")

    def process_audio(self, pcm_data: bytes) -> bytes:
        """Process an audio chunk through voice conversion.

        Input: PCM s16le mono audio bytes
        Output: Converted PCM s16le mono audio bytes

        May buffer input and return empty bytes if not enough data
        has accumulated for processing.
        """
        if not self._loaded or self._model is None:
            return pcm_data

        t0 = time.perf_counter()

        # Convert bytes to numpy
        chunk = np.frombuffer(pcm_data, dtype=np.int16)
        self._buffer = np.concatenate([self._buffer, chunk])

        # Need minimum samples for quality inference
        if len(self._buffer) < self._min_samples:
            return b""  # Buffer more before processing

        # Process the buffered audio
        audio_to_process = self._buffer.copy()
        self._buffer = np.array([], dtype=np.int16)

        result = self._convert(audio_to_process)

        # Update latency
        latency_ms = (time.perf_counter() - t0) * 1000
        self._avg_latency_ms = (
            self._latency_alpha * latency_ms + (1 - self._latency_alpha) * self._avg_latency_ms
        )

        return result.tobytes()

    def _convert(self, audio: np.ndarray) -> np.ndarray:
        """Run RVC inference on audio samples.

        Pipeline:
            1. Convert int16 PCM to float32
            2. Extract HuBERT features
            3. Extract pitch with RMVPE
            4. Apply pitch shift
            5. Optionally blend with FAISS index features
            6. Run RVC synthesis
            7. Resample to 16kHz if needed
            8. Convert back to int16 PCM
        """
        import torch

        if self._hubert is None or self._rmvpe is None or self._model is None:
            return audio

        try:
            # 1. Convert int16 PCM to float32 [-1, 1]
            audio_f32 = audio.astype(np.float32) / 32768.0
            waveform = torch.from_numpy(audio_f32).to(self._device)

            # 2. Extract HuBERT features (768-dim at ~50Hz)
            with torch.no_grad():
                hubert_feats = self._hubert.extract_features(waveform)  # (1, T', 768)

            # 3. Extract pitch using RMVPE
            f0 = self._rmvpe.extract_pitch(waveform)  # (T'',) numpy, in Hz

            # 4. Apply pitch shift (in semitones)
            if self._pitch_shift != 0:
                shift_factor = 2.0 ** (self._pitch_shift / 12.0)
                f0 = f0 * shift_factor

            # 5. Align feature and pitch lengths
            feat_len = hubert_feats.shape[1]
            f0_len = len(f0)

            # HuBERT features are at 50Hz (320 samples per frame at 16kHz)
            # RMVPE f0 is at 100Hz (160 samples per frame at 16kHz)
            # We need to downsample f0 to match HuBERT frame rate
            if f0_len > feat_len:
                # Downsample f0 to match feature length
                indices = np.linspace(0, f0_len - 1, feat_len).astype(int)
                f0 = f0[indices]
            elif f0_len < feat_len:
                # Pad f0
                f0 = np.pad(f0, (0, feat_len - f0_len), mode="edge")

            # 6. Optionally blend with FAISS retrieved features
            if self._faiss_index is not None:
                hubert_feats = self._apply_index_retrieval(hubert_feats)

            # 7. Prepare inputs for RVC synthesis
            # Quantize pitch to bins (0-255) for embedding lookup
            pitch_quant = self._quantize_pitch(f0)
            pitch_quant = torch.from_numpy(pitch_quant).long().unsqueeze(0).to(self._device)  # (1, T)

            # Continuous pitch for source module
            f0_tensor = torch.from_numpy(f0.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self._device)  # (1, 1, T)

            # Speaker ID (always 0 for single-speaker models)
            sid = torch.zeros(1, dtype=torch.long, device=self._device)

            # 8. Run RVC synthesis
            with torch.no_grad():
                audio_out = self._model.infer(
                    hubert_feats,
                    pitch_quant,
                    f0_tensor,
                    sid,
                )  # (1, 1, T_audio) at model's native sample rate

            # 9. Convert to numpy
            audio_out = audio_out.squeeze().cpu().numpy()

            # 10. Resample from model's native rate to 16kHz if needed
            if self._rvc_sr != self._sample_rate:
                audio_out = self._resample(audio_out, self._rvc_sr, self._sample_rate)

            # 11. Match output length to input length for real-time streaming
            target_len = len(audio)
            if len(audio_out) > target_len:
                audio_out = audio_out[:target_len]
            elif len(audio_out) < target_len:
                audio_out = np.pad(audio_out, (0, target_len - len(audio_out)))

            # 12. Convert back to int16 PCM
            audio_out = np.clip(audio_out * 32768.0, -32768, 32767).astype(np.int16)

            return audio_out

        except Exception as e:
            logger.error(f"Voice conversion error: {e}", exc_info=True)
            # Fallback: return original audio on error
            return audio

    def _quantize_pitch(self, f0: np.ndarray) -> np.ndarray:
        """Quantize continuous f0 to discrete pitch bins (0-255).

        Uses logarithmic mapping similar to MIDI note numbers.
        Bin 0 = unvoiced, bins 1-255 = voiced pitch range.
        """
        pitch_quant = np.zeros_like(f0, dtype=np.int64)
        voiced = f0 > 0

        if voiced.any():
            # Map frequency to bin: 256 bins covering reasonable pitch range
            # Using log scale: bin = round(12 * log2(f/fmin) * (255/n_octaves))
            fmin = 50.0  # minimum frequency
            n_octaves = 6.0  # cover ~6 octaves
            bins = np.round(
                12.0 * np.log2(np.maximum(f0[voiced], fmin) / fmin) * (255.0 / (12.0 * n_octaves))
            ).astype(np.int64)
            bins = np.clip(bins, 1, 255)
            pitch_quant[voiced] = bins

        return pitch_quant

    def _apply_index_retrieval(self, features):
        """Blend HuBERT features with FAISS-retrieved features.

        This retrieves the closest training features from the index,
        giving the output a more natural quality matching the target voice.
        """
        import torch

        if self._faiss_index is None:
            return features

        try:
            feats_np = features.squeeze(0).cpu().numpy().astype(np.float32)

            # Search for nearest neighbors
            _, indices = self._faiss_index.search(feats_np, 1)

            # Reconstruct features from index
            retrieved = np.zeros_like(feats_np)
            for i, idx in enumerate(indices[:, 0]):
                if idx >= 0:
                    retrieved[i] = self._faiss_index.reconstruct(int(idx))

            retrieved_tensor = torch.from_numpy(retrieved).unsqueeze(0).to(features.device)

            # Blend original and retrieved features
            blended = (1 - self._index_rate) * features + self._index_rate * retrieved_tensor
            return blended

        except Exception as e:
            logger.debug(f"Index retrieval failed: {e}")
            return features

    @staticmethod
    def _resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        """Resample audio using linear interpolation.

        For better quality, torchaudio.functional.resample could be used,
        but linear interpolation is sufficient for real-time streaming.
        """
        if sr_from == sr_to:
            return audio

        ratio = sr_to / sr_from
        target_len = int(len(audio) * ratio)

        # Use numpy interpolation
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def has_model(self) -> bool:
        return bool(self._model_path)

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def avg_latency_ms(self) -> float:
        return self._avg_latency_ms

    @property
    def model_name(self) -> str:
        if self._model_path:
            return Path(self._model_path).stem
        return ""

    @property
    def pitch_shift(self) -> int:
        return self._pitch_shift

    @pitch_shift.setter
    def pitch_shift(self, value: int) -> None:
        self._pitch_shift = max(-12, min(12, value))
