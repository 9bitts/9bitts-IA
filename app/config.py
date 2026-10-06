"""Decisões fechadas da fase 1 e o que muda quando o assistente vira produto."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

# Modelo do dia a dia e modelo quando a resposta precisa ser melhor.
# Os dois cabem na faixa pedida (cerca de 8B e cerca de 14B) e aceitam ferramentas.
FAST_MODEL = "llama3.1:8b"
QUALITY_MODEL = "qwen2.5:14b"
EMBED_MODEL = "nomic-embed-text"

# A fase 1 está pronta quando estes quatro pontos funcionam no navegador.
PHASE1_ACCEPTANCE = (
    "Abre no navegador e conversa em streaming",
    "O histórico continua depois de fechar e abrir",
    "Dá para ter mais de uma conversa",
    "Dá para trocar entre o modelo rápido e o modelo melhor",
)


class Settings(BaseSettings):
    deploy_mode: str = "personal"
    ollama_base_url: str = "http://127.0.0.1:11434"
    database_url: str = "sqlite:///" + (ROOT / "data" / "app.db").as_posix()
    fast_model: str = FAST_MODEL
    quality_model: str = QUALITY_MODEL
    embed_model: str = EMBED_MODEL
    searxng_url: str = ""
    product_daily_limit: int = 50
    product_admin_email: str = ""
    product_admin_password: str = ""
    host: str = "127.0.0.1"
    port: int = 8787
    persona_path: Path = ROOT / "persona.txt"

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    mode = settings.deploy_mode.strip().lower()
    if mode not in {"personal", "product"}:
        raise RuntimeError("DEPLOY_MODE precisa ser personal ou product.")
    settings.deploy_mode = mode
    settings.ollama_base_url = settings.ollama_base_url.rstrip("/")
    return settings
