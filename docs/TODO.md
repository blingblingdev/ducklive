# DuckLive TODO

## ✅ 已完成
- [x] 项目结构搭建
- [x] Server/Client/Dashboard 基础框架
- [x] WebSocket 二进制串流协议（6种下行 + 2种上行帧类型）
- [x] mDNS 服务发现
- [x] GPU/CPU 自动检测
- [x] Test mode（合成帧）
- [x] Dashboard 双画面预览 + 音频电平表
- [x] Client Web UI
- [x] **架构重构：Server 纯处理节点，Client 浏览器采集**
- [x] **浏览器 getUserMedia 采集 + WebSocket 上传**
- [x] **端到端管道验证（浏览器 → Server → 浏览器预览）**
- [x] **AI 模型下载（inswapper_128.onnx + buffalo_l）**
- [x] **Client 端引擎控制面板（换脸开关+选脸、变声开关+选声音+音高调节）**
- [x] **Server API：资产列表、选择、引擎配置**
- [x] **Client 代理层：所有 Server API 通过 Client 转发**
- [x] **ServerConfig 清理：移除启动时人脸/声音参数，运行时由客户端设置**
- [x] **Head swap paste-back soft blending (erode + Gaussian blur mask)**
- [x] **RVC voice change engine (HuBERT + RMVPE + RVC Synthesizer)**
- [x] **Dashboard: remove face selection UI, monitoring-only**

## 🔥 当前优先
- [ ] 放入真实人脸图片到 `faces/` 测试换脸效果
- [ ] 端到端测试 RVC 变声效果（需 GPU server）

## 📋 待完成
- [ ] 虚拟摄像头输出（pyvirtualcam + OBS 虚拟摄像头后端）
- [ ] 虚拟麦克风输出（BlackHole）
- [ ] Feed 断连自动重连（Server 端优雅处理）
- [ ] 多分辨率自适应（根据网络质量调整 JPEG quality）
- [ ] 延迟监控（端到端 latency 显示在 Dashboard/Client UI）
- [ ] Windows Server 部署测试
- [ ] HTTPS/WSS 支持（跨网络使用）
- [ ] 手机端支持（WebRTC / 移动浏览器）

## 🐛 已知问题
- macOS 终端进程无法获取摄像头权限（已通过浏览器采集绕过）
- getUserMedia 同时请求 video+audio 可能挂起（已改为分步请求）
- AudioWorklet 在某些浏览器下可能失败（已做 fallback 处理）
