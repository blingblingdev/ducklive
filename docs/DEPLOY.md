# DuckLive 部署架构

## 核心原则

**每个组件都可以独立部署在不同机器上，通过网络 IP 互相访问。**

## 组件

| 组件 | 角色 | 监听端口 | 绑定地址 |
|---|---|---|---|
| **DuckLive Server** | GPU 推理 + 串流 | 8080 (Dashboard) + 8765 (WebSocket) | `0.0.0.0`（接受任意来源） |
| **DuckLive Client** | 虚拟摄像头/麦克风 | 无（纯客户端） | N/A |
| **Dashboard** | Web 管理界面 | 内嵌在 Server 8080 端口 | 随 Server |

## 部署场景

### 场景 A：全部本机（开发/测试）

```
┌─ Mac ──────────────────────────────────┐
│  Server (127.0.0.1:8080 + :8765)       │
│  Client → ws://127.0.0.1:8765          │
│  Dashboard → http://127.0.0.1:8080     │
└────────────────────────────────────────┘
```

```bash
# Terminal 1
ducklive server --host 127.0.0.1

# Terminal 2
ducklive client --server-url ws://127.0.0.1:8765
```

### 场景 B：内网分布式（生产推荐）

```
┌─ Windows (192.168.1.100) ──────────────┐    ┌─ Mac (192.168.1.50) ─────────┐
│  RTX 5090                              │    │                               │
│  Server (0.0.0.0:8080 + :8765)         │◄──►│  Client (auto-discover)       │
│  mDNS 广播 _ducklive._tcp              │    │  → ws://192.168.1.100:8765    │
│  Dashboard                             │    │  虚拟摄像头 → Zoom/Teams      │
└────────────────────────────────────────┘    │  虚拟麦克风 → Zoom/Teams      │
        ▲                                     └───────────────────────────────┘
        │
        │  浏览器访问
        ▼
┌─ 任意设备 ─────────────────────────────┐
│  http://192.168.1.100:8080             │
│  Dashboard (状态/预览/配置)             │
└────────────────────────────────────────┘
```

```bash
# Windows
ducklive server

# Mac (自动发现)
ducklive client

# Mac (手动指定)
ducklive client --server-url ws://192.168.1.100:8765

# 任何设备浏览器
# → http://192.168.1.100:8080
```

### 场景 C：多客户端

Server 支持同时最多 5 个客户端连接（可配置 `--max-clients`）。

```
Windows Server (192.168.1.100)
    ├── Mac Client A (192.168.1.50)  → 虚拟摄像头
    ├── Mac Client B (192.168.1.51)  → 虚拟摄像头
    └── iPad 浏览器 (192.168.1.60)   → Dashboard 预览
```

## 网络要求

- Server 和 Client 在同一局域网内
- 端口 8080 (HTTP) 和 8765 (WebSocket) 未被防火墙阻断
- mDNS 自动发现需要组播流量未被路由器屏蔽（大多数家庭/办公网络默认允许）
- 如果 mDNS 不可用，Client 可以通过 `--server-url` 手动指定服务器地址

## 端口配置

所有端口可通过 CLI 参数或环境变量自定义：

```bash
# CLI
ducklive server --port 9090 --ws-port 9765

# 环境变量
export DUCKLIVE_PORT=9090
export DUCKLIVE_WS_PORT=9765
ducklive server
```

## 安全说明

⚠️ DuckLive 目前不包含认证机制，仅适合在受信任的内网环境中使用。
任何能访问到 Server IP + 端口的设备都可以：
- 查看 Dashboard
- 连接流
- 上传人脸/声音模型

后续版本会考虑加入 token 认证。
