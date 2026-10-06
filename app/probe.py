"""Registra se o Ollama está no ar e se a Radeon aparece via Vulkan."""

import asyncio
import json

from app.compute import detect_compute
from app.config import ROOT
from app.ollama_client import installed_models, ollama_online, running_models


async def collect() -> dict:
    online = await ollama_online()
    compute = detect_compute()
    installed = await installed_models() if online else []
    running = await running_models() if online else []
    vram = sum(int(item.get("size_vram") or 0) for item in running)
    return {
        "ollama_online": online,
        "installed": installed,
        "size_vram": vram,
        "using_gpu": vram > 0,
        "compute": compute,
    }


def main() -> None:
    report = asyncio.run(collect())
    target = ROOT / "data" / "runtime.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compute = report["compute"]
    print(f"online={report['ollama_online']}")
    print(f"library={compute.get('library')} description={compute.get('description')}")
    print(f"vulkan={compute.get('vulkan')} radeon_860m={compute.get('radeon_860m')} using_gpu={report['using_gpu']}")
    print(f"installed={', '.join(report['installed']) or '-'}")
    if not report["ollama_online"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
