# DuckLive Changelog

## v0.4.0 — RVC Voice Change + Head Swap Fix (2026-02-09)

### Head Swap: White Edge Fix
- Added `_soften_mask()` in `head_swap.py` to eliminate visible white border artifacts
- Technique: erode the paste-back mask (3x3 kernel, 3 iterations) then Gaussian blur (51x51)
- Creates a wide, smooth alpha falloff at the face-to-background boundary
- Pre-computed constants (`_ERODE_KERNEL`, `_BLUR_KSIZE`, `_ERODE_ITERS`) for performance

### RVC Voice Change Engine (full implementation)
- Replaced passthrough stub in `voice_change.py` with complete RVC v2 pipeline (~1600 lines)
- **HuBERT feature extractor**: lightweight reimplementation using raw PyTorch tensors
  - Loads `hubert_base.pt` without fairseq dependency (minimal stub for unpickling)
  - 7-layer CNN + 12 post-LN transformer layers, outputs 768-dim features at ~50Hz
- **RMVPE pitch extractor**: full U-Net + BiGRU network
  - Loads `rmvpe.pt`, 5-layer encoder/decoder U-Net + bidirectional GRU
  - 360-bin pitch estimation with sub-bin precision via softmax-weighted interpolation
  - Custom mel spectrogram computation (128 mels, 1024 FFT, 160 hop)
- **RVC v2 Synthesizer**: TextEncoder + ResidualCouplingFlow + NSF-HiFi-GAN
  - Builds model from config in .pth file (supports 32k/40k/48k sample rates)
  - NSF excitation source for pitch-accurate voice generation
  - Speaker embedding conditioning throughout the pipeline
- **FAISS index retrieval**: optional feature blending with training data (if .index file exists)
- **Pitch shifting**: -12 to +12 semitones support
- **Buffering**: accumulates 400ms (6400 samples) before processing for RMVPE compatibility
- **Resampling**: converts from model's native rate (e.g. 40kHz) back to 16kHz output
- Tested: ~243ms inference for 500ms audio on CPU (Apple Silicon)

### Dashboard UI Simplified
- Removed face thumbnail grid and face/voice selection dropdowns from dashboard
- Removed upload buttons and related JavaScript (`uploadFace()`, `uploadVoice()`, `updateSelect()`)
- Added read-only display of currently active face and voice model (green highlight when set)
- Engine toggle switches (enable/disable) and monitoring stats preserved
- Fixed engine toggle handlers to POST to correct `/api/engines/configure` endpoint
- Translated all UI text from Chinese to English
- API endpoints (`/api/faces/select`, etc.) kept intact for client proxy usage
- CSS: replaced `.engine-control` styles with `.engine-info` read-only display styles

---

## v0.3.0 — Client-Driven Engine Control (2026-02-09)

**重大改动：换脸/变声的控制权从服务端移到客户端**

### 设计理念变更
- Server 是纯处理服务：提供可用模型列表、接受客户端指令
- Client 是用户操作中心：选择人脸/声音模型、开关引擎、调节参数
- 模型资产由服务端管理（`faces/` + `voices/` 目录），客户端不上传

### Server API 新增
- `GET /api/faces` — 列出可用人脸图片
- `GET /api/faces/{name}/thumbnail` — 人脸缩略图预览
- `GET /api/voices` — 列出可用声音模型
- `POST /api/faces/select` — 选择/清除目标人脸
- `POST /api/voices/select` — 选择/清除声音模型
- `GET /api/engines` — 获取引擎状态（可用性、开关、当前选择）
- `POST /api/engines/configure` — 配置引擎开关和参数

### Client UI 新增控制面板
- 🎭 换脸：开关 + 人脸缩略图网格选择
- 🎤 变声：开关 + 声音模型列表选择 + 音高调节滑块（-12 ~ +12）
- 所有控制通过 Client Python 代理转发到 Server API
- 引擎状态定时轮询（3 秒），UI 自动同步

### ServerConfig 清理
- 移除 `face_image_path`、`voice_model_path` 启动参数
- 引擎默认关闭（`face_swap_enabled=False`、`voice_change_enabled=False`）
- 引擎启动时预加载但不设目标，运行时由客户端设置
- CLI 移除 `--face`、`--voice` 参数

### Client 代理层
- Client Python 后端代理所有 Server API 调用（避免 CORS）
- 新增 `server_dashboard_url` 属性（从 mDNS 或 WS URL 推导）
- 新增 `httpx` 依赖

### Dashboard 精简
- 移除上传功能（不再允许通过 Dashboard 上传模型）
- 保留监控 + API 服务角色

---

## v0.2.0 — Architecture Refactor (2026-02-09)

**重大改动：Server 不再采集摄像头，改为纯处理节点**

### 架构变更
- **Server 端**：移除本地摄像头/麦克风采集，改为接收 Client 上传的原始帧
- **Client 端**：浏览器通过 `getUserMedia()` 采集摄像头/麦克风，通过 WebSocket `/feed` 上传到 Server
- **数据流**：Client 浏览器 → Server 处理 → Client 浏览器预览 + Python 虚拟设备输出
- **新增 Feed 角色**：WebSocket 三种连接类型（feed/stream/dashboard）

### 协议扩展
- 新增上行帧类型：`RAW_VIDEO (0x10)`, `RAW_AUDIO (0x11)`
- Feed 客户端：发送原始帧，接收处理后的帧
- 保持向后兼容：stream/dashboard 端点不变

### Client Web UI 重写
- 浏览器端摄像头/麦克风采集（`getUserMedia`）
- 设备选择器（摄像头、麦克风下拉菜单）
- 双预览：本地摄像头 vs AI 处理后
- AudioWorklet 音频采集（48kHz → 16kHz 重采样）
- 实时统计：发送/接收帧数、FPS

### Server 改进
- 独立的视频/音频处理循环（asyncio tasks）
- 队列式帧接收（raw_video_queue + raw_audio_queue）
- 仅允许一个 Feed 客户端同时连接
- Test mode 保留为独立合成帧生成器

### Dashboard 更新
- "摄像头"卡片改为"Feed 源"卡片
- 显示 Feed 客户端连接状态和来源地址

### Bug 修复
- 修复 getUserMedia 同时请求视频+音频导致挂起的问题（改为分步请求）
- 移除页面加载时的预授权 getUserMedia 调用（避免阻塞设备枚举）

---

## v0.1.4 — Dashboard Preview (2026-02-09)

- Dashboard 双画面实时预览（原始 vs 换脸后）
- 音频电平表
- 引擎开关控制

## v0.1.3 — Protocol Extension (2026-02-09)

- 新增 ORIGINAL_VIDEO/AUDIO, AUDIO_LEVELS 帧类型
- Dashboard/Client 角色分离

## v0.1.2 — Client Web UI (2026-02-09)

- Client 轻量 Web UI（状态 + 预览）
- 暗色主题

## v0.1.1 — Test Mode (2026-02-09)

- 合成视频/音频测试模式
- 无需真实摄像头即可测试管道

## v0.1.0 — Initial Scaffold (2026-02-09)

- 项目结构搭建（~3200 行，33 个文件）
- Server/Client/Dashboard 基础框架
- WebSocket 二进制串流协议
- mDNS 服务发现
- GPU/CPU 自动检测
