"""Click Run in VS Code to launch the local application."""

import os
import socket
import threading
import time
import webbrowser
from pathlib import Path

import httpx
import uvicorn


def open_when_ready():
    for _ in range(60):
        try:
            with socket.create_connection(("127.0.0.1", 8008), timeout=0.5):
                webbrowser.open("http://127.0.0.1:8008")
                return
        except OSError:
            time.sleep(0.5)


if __name__ == "__main__":
    try:
        existing = httpx.get("http://127.0.0.1:8008/api/health", timeout=1)
        if existing.is_success and existing.json().get("app") == "ai-docs-agent":
            print("AI Docs Agent is already running: http://127.0.0.1:8008")
            webbrowser.open("http://127.0.0.1:8008")
            raise SystemExit(0)
    except (httpx.HTTPError, ValueError):
        pass
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "server.pid").write_text(str(os.getpid()), encoding="utf-8")
    print("AI Docs Agent: http://127.0.0.1:8008")
    print("Stop: Ctrl+C in this terminal.")
    threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run("ai_docs_agent.api:app", host="127.0.0.1", port=8008, workers=1)
