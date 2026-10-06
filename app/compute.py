"""Lê o log do Ollama para saber se a Radeon entrou via Vulkan."""

import os
import re
from pathlib import Path


def _log_path() -> Path | None:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    path = Path(local) / "Ollama" / "server.log"
    return path if path.is_file() else None


def _parse_line(line: str) -> dict:
    library = re.search(r"library=(\S+)", line)
    description = re.search(r'description="([^"]*)"', line)
    kind = re.search(r"\btype=(\S+)", line)
    return {
        "library": library.group(1) if library else "",
        "description": description.group(1) if description else "",
        "type": kind.group(1) if kind else "",
    }


def detect_compute() -> dict:
    path = _log_path()
    if path is None:
        return {
            "library": None,
            "description": None,
            "type": None,
            "radeon_860m": False,
            "vulkan": False,
            "devices": [],
        }
    text = path.read_bytes()[-400_000:].decode("utf-8", errors="replace")
    start = text.rfind("discovering available GPUs")
    section = text[start:] if start >= 0 else text
    devices = []
    dropped = None
    for line in section.splitlines():
        if "inference compute" in line:
            parsed = _parse_line(line)
            if parsed["library"]:
                devices.append(parsed)
        if "dropping integrated GPU" in line and "library=Vulkan" in line:
            dropped = _parse_line(line)
    vulkan = next((item for item in reversed(devices) if item["library"].lower() == "vulkan"), None)
    radeon_name = ""
    if vulkan:
        radeon_name = vulkan["description"]
    elif dropped:
        radeon_name = dropped["description"]
    chosen = vulkan or (devices[-1] if devices else dropped)
    return {
        "library": chosen["library"] if chosen else None,
        "description": chosen["description"] if chosen else None,
        "type": chosen["type"] if chosen else None,
        "radeon_860m": "860" in radeon_name,
        "vulkan": vulkan is not None,
        "igpu_dropped": dropped is not None and vulkan is None,
        "devices": devices[-8:],
    }
