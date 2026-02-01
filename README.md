# Supporter-Agent Web (方案B：最稳妥)

## 运行方式（本地）
1) 安装依赖
```bash
pip install fastapi uvicorn pydantic
```

2) 启动后端
```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

3) 打开页面
- 老人端（主控端）：`http://localhost:8000/master_display.html`
- 采集端（他人端）：从老人端侧边栏复制加入链接（仅带 session_id；单设备双声道模式）

> 说明：此版本完全不依赖 Streamlit。WS 长连接由 FastAPI 承担；前端是纯 HTML/JS，内置 ping/pong 心跳 + 自动重连。

## 目录
- backend/app.py: FastAPI (REST + WebSocket + SQLite)
- frontend/master_display.html: 老人端 SPA
- frontend/client_display.html: 采集端（兼容你原来的页面思路）
- data/supporter.db: SQLite 数据库（会自动创建）


## 后端流式ASR（GPU）
安装依赖：
```bash
pip install numpy webrtcvad faster-whisper
```

启动（含HTTPS）：
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --ssl-certfile localhost+3.pem --ssl-keyfile localhost+3-key.pem
```

环境变量（可选）：
- ASR_MODEL=small|medium|large-v3（默认 small）
- ASR_DEVICE=cuda|cpu（默认 cuda）
- ASR_LANGUAGE=zh（默认 zh）
- ASR_PARTIAL_INTERVAL=0.4（默认 0.4s）
- ASR_PARTIAL_WINDOW=6（默认 6s）
- ASR_SILENCE_FINALIZE=0.55（默认 0.55s）
- ASR_MAX_UTTERANCE_SEC=12（默认 12s）
