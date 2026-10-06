"""Sobe o Ollama e a página no mesmo servidor. A página abre antes do modelo terminar de baixar."""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _pull_models() -> None:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    env["OLLAMA_MODELS"] = "/data/models"
    for _ in range(90):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        return
    for name in (os.environ.get("FAST_MODEL", "llama3.1:8b"), os.environ.get("EMBED_MODEL", "nomic-embed-text")):
        subprocess.call(["ollama", "pull", name], env=env)


def main() -> None:
    if "--pull" in sys.argv:
        _pull_models()
        return
    Path("/data/models").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    env["OLLAMA_MODELS"] = "/data/models"
    subprocess.Popen(["ollama", "serve"], env=env)
    # Processo separado: o exec abaixo substitui este programa e mataria uma thread.
    subprocess.Popen([sys.executable, __file__, "--pull"], env=env, start_new_session=True)
    port = os.environ.get("PORT", "8080")
    os.execvp(
        "uvicorn",
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", port],
    )


if __name__ == "__main__":
    main()
