# 🦆 DuckLive

**实时换脸 + 变声的网络摄像头**

一个整合了 AI 换脸、AI 变声、网络串流的单体应用。Windows 端做 GPU 推理，Mac 端自动发现并作为虚拟摄像头/麦克风使用，直接用于视频通话。

## 架构

```
Windows Machine (RTX 5090)                    Mac (Coco's workstation)
┌─────────────────────────────────┐          ┌──────────────────────────────┐
│  DuckLive Server                │          │  DuckLive Client             │
│                                 │          │                              │
│  🎥 Webcam ──► Face Swap       │          │  Auto-discover (mDNS)        │
│                  (InsightFace)  │  WebSocket│                              │
│                       │         │◄────────►│  ┌─ Virtual Camera ──► Zoom  │
│                       ▼         │          │  │                           │
│              Composited Stream  │          │  └─ Virtual Mic    ──► Zoom  │
│                       ▲         │          │                              │
│                       │         │          │  Tray / Status Bar app       │
│  🎤 Mic ────► Voice Change     │          └──────────────────────────────┘
│                  (RVC)          │
│                                 │           Any Device
│  📊 Dashboard (:8080)          │          ┌──────────────────────────────┐
│  🔍 mDNS Advertisement         │          │  Browser                     │
└─────────────────────────────────┘          │  http://ducklive.local:8080  │
                                             │  Dashboard + Live Preview    │
                                             └──────────────────────────────┘
```

## 核心模块

### Server (Windows)
- **采集层**：OpenCV 读摄像头，PyAudio 读麦克风
- **换脸引擎**：InsightFace `inswapper_128` + ONNX Runtime CUDA
- **变声引擎**：RVC (Retrieval-based Voice Conversion) + CUDA
- **串流层**：WebSocket 二进制流（视频 MJPEG + 音频 PCM），低延迟
- **发现层**：mDNS/Bonjour 广播 `_ducklive._tcp`
- **Dashboard**：FastAPI + Web UI，实时状态/预览/配置

### Client (Mac)
- **发现层**：zeroconf 自动发现 DuckLive 服务器
- **接收层**：WebSocket 客户端接收音视频流
- **虚拟摄像头**：pyvirtualcam → 在 Zoom/Teams 中选择 "DuckLive Camera"
- **虚拟麦克风**：通过 BlackHole 虚拟音频设备路由

## 串流协议

WebSocket 二进制帧格式：

```
┌──────┬───────────┬──────────┬─────────────┐
│ Type │ Timestamp │ Size     │ Payload     │
│ 1B   │ 8B (u64)  │ 4B (u32) │ variable    │
├──────┼───────────┼──────────┼─────────────┤
│ 0x01 │ ...       │ ...      │ JPEG frame  │  Video
│ 0x02 │ ...       │ ...      │ PCM s16le   │  Audio
│ 0x03 │ ...       │ ...      │ JSON        │  Control
└──────┴───────────┴──────────┴─────────────┘
```

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| 语言 | Python 3.11+ | AI 生态完善 |
| 换脸 | InsightFace + ONNX Runtime | inswapper_128 模型 |
| 变声 | RVC + RMVPE | 低数据量高质量 |
| GPU 加速 | CUDA 12.x (RTX 5090) | ONNX Runtime CUDA EP |
| Web 框架 | FastAPI + Jinja2 | Dashboard |
| 串流 | WebSocket (websockets) | 低延迟二进制传输 |
| 服务发现 | zeroconf | mDNS/Bonjour |
| 虚拟摄像头 | pyvirtualcam | macOS/Windows |
| 虚拟音频 | BlackHole (Mac) | 虚拟音频路由 |
| 打包 | PyInstaller | .exe / .app |

## Dashboard 功能

- 🟢 全局状态总览（服务器在线/离线、连接的客户端数、帧率、延迟）
- 📹 实时预览（原始画面 vs 换脸后画面）
- 🎭 人脸管理（上传/切换目标人脸照片）
- 🎤 声音管理（上传/切换 RVC 声音模型）
- ⚙️ 参数调节（换脸强度、变声参数、分辨率、帧率）
- 📈 性能监控（GPU 使用率、推理延迟、网络带宽）

## Installation

```bash
# Install from GitHub (requires Python 3.11+)
pip install git+https://github.com/blingblingdev/ducklive.git

# With CUDA support (recommended for GPU machines)
pip install "ducklive[cuda] @ git+https://github.com/blingblingdev/ducklive.git"

# Check required models
ducklive check-models
```

## 快速开始

```bash
# Server (Windows)
ducklive server

# Server with options
ducklive server --host 0.0.0.0 --port 8080 --dev

# Client (Mac)
ducklive client

# 或直接打开 Dashboard
# http://ducklive.local:8080
```

## 开发

```bash
# Clone and install in editable mode
git clone git@github.com:blingblingdev/ducklive.git
cd ducklive
pip install -e ".[dev]"

# Run in dev mode
ducklive server --dev
ducklive client --dev
```

## 路线图

- [x] v0.1 — 项目结构 + 架构设计
- [ ] v0.2 — 换脸引擎 + 虚拟摄像头 (Server→Client 视频流)
- [ ] v0.3 — 变声引擎 + 虚拟麦克风 (音频流)
- [ ] v0.4 — Dashboard (状态 + 预览 + 配置)
- [ ] v0.5 — mDNS 自动发现
- [ ] v0.6 — Windows .exe 打包
- [ ] v0.7 — Mac .app 打包
- [ ] v1.0 — 稳定版发布
- [ ] v1.x — 手机端支持 (WebRTC)

## License

Private — Coco & Cici 🦆
