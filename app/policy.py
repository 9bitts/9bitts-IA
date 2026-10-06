"""Regras que separam o uso neste notebook do atendimento a clientes."""

from pathlib import Path
from urllib.parse import urlparse


def folder_access_enabled(mode: str) -> bool:
    """A pasta do computador só existe no modo pessoal."""
    return mode == "personal"


def inference_location(ollama_base_url: str) -> str:
    host = (urlparse(ollama_base_url).hostname or "").lower()
    if host in {"127.0.0.1", "localhost", "::1"}:
        return "local"
    return "remote"


def database_kind(database_url: str) -> str:
    if database_url.startswith("postgres"):
        return "postgres"
    return "sqlite"


def resolve_inside(root: Path, relative: str) -> Path:
    """Devolve um arquivo dentro da pasta marcada, ou recusa o caminho."""
    base = root.resolve()
    raw = Path(relative)
    if raw.is_absolute():
        raise PermissionError("fora da pasta indicada")
    candidate = (base / raw).resolve()
    if not candidate.is_relative_to(base):
        raise PermissionError("fora da pasta indicada")
    return candidate
