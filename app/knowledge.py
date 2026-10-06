"""Índice local dos documentos. Cada bloco fica preso ao usuário que o enviou."""

import asyncio
import json
import math
import threading
from pathlib import Path

from app.config import get_settings
from app import db as database
from app.db import Chunk, KnowledgeBase
from app.ollama_client import embed_texts
from app.policy import resolve_inside

TEXT_EXT = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".rst",
}
SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", "data"}
MAX_FILE_CHARS = 200_000
MAX_FILES = 200
MAX_CHUNKS = 500
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

_jobs: dict[int, dict] = {}
_lock = threading.Lock()


def job_status(user_id: int) -> dict:
    with _lock:
        current = _jobs.get(user_id) or {"state": "idle", "files": 0, "chunks": 0, "error": None}
        return dict(current)


def _set_job(user_id: int, **values) -> None:
    with _lock:
        current = _jobs.setdefault(
            user_id, {"state": "idle", "files": 0, "chunks": 0, "error": None}
        )
        current.update(values)


def ensure_base(db, user_id: int) -> KnowledgeBase:
    base = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user_id).one_or_none()
    if base is None:
        base = KnowledgeBase(user_id=user_id, name="Documentos")
        db.add(base)
        db.flush()
    return base


def sources_for(db, user_id: int) -> list[str]:
    rows = (
        db.query(Chunk.source)
        .filter(Chunk.user_id == user_id)
        .distinct()
        .order_by(Chunk.source)
        .all()
    )
    return [row[0] for row in rows]


def chunk_count(db, user_id: int) -> int:
    return db.query(Chunk).filter(Chunk.user_id == user_id).count()


def split_text(text: str) -> list[str]:
    clean = text.replace("\r\n", "\n").strip()
    if not clean:
        return []
    pieces = []
    start = 0
    while start < len(clean) and len(pieces) < MAX_CHUNKS:
        end = min(len(clean), start + CHUNK_SIZE)
        pieces.append(clean[start:end])
        if end >= len(clean):
            break
        start = end - CHUNK_OVERLAP
    return pieces


def iter_folder_files(root: Path):
    found = 0
    for path in root.rglob("*"):
        if found >= MAX_FILES:
            break
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_EXT:
            continue
        try:
            if path.stat().st_size > MAX_FILE_CHARS:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "\x00" in text[:1024]:
            continue
        found += 1
        yield str(path.relative_to(root)), text


def start_reindex(user_id: int, folder: str) -> None:
    with _lock:
        if _jobs.get(user_id, {}).get("state") == "running":
            return
        _jobs[user_id] = {"state": "running", "files": 0, "chunks": 0, "error": None}
    threading.Thread(target=_index_folder, args=(user_id, folder), daemon=True).start()


def _index_folder(user_id: int, folder: str) -> None:
    try:
        asyncio.run(_index_folder_async(user_id, folder))
    except Exception as exc:
        _set_job(user_id, state="error", error=str(exc))


async def _index_folder_async(user_id: int, folder: str) -> None:
    if database.SessionLocal is None:
        database.init_db()
    root = Path(folder)
    documents = list(iter_folder_files(root))
    db = database.SessionLocal()
    try:
        base = ensure_base(db, user_id)
        base.folder_path = str(root)
        db.query(Chunk).filter(Chunk.user_id == user_id, Chunk.origin == "folder").delete()
        stored = await _store_documents(db, user_id, base.id, documents, origin="folder")
        db.commit()
        _set_job(user_id, state="done", files=len(documents), chunks=stored, error=None)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def add_upload(db, user_id: int, filename: str, text: str) -> int:
    base = ensure_base(db, user_id)
    db.query(Chunk).filter(
        Chunk.user_id == user_id, Chunk.origin == "upload", Chunk.source == filename
    ).delete()
    stored = await _store_documents(db, user_id, base.id, [(filename, text)], origin="upload")
    db.commit()
    return stored


async def _store_documents(db, user_id: int, base_id: int, documents: list[tuple[str, str]], origin: str) -> int:
    pieces: list[tuple[str, str]] = []
    for source, text in documents:
        for piece in split_text(text):
            pieces.append((source, piece))
            if len(pieces) >= MAX_CHUNKS:
                break
        if len(pieces) >= MAX_CHUNKS:
            break
    vectors: list[list[float] | None] = [None] * len(pieces)
    batch = 16
    for start in range(0, len(pieces), batch):
        group = pieces[start : start + batch]
        embedded = await embed_texts([item[1] for item in group])
        if embedded and len(embedded) == len(group):
            for offset, vector in enumerate(embedded):
                vectors[start + offset] = vector
    for (source, content), vector in zip(pieces, vectors):
        db.add(
            Chunk(
                user_id=user_id,
                knowledge_base_id=base_id,
                source=source,
                origin=origin,
                content=content,
                embedding=json.dumps(vector) if vector else None,
            )
        )
    db.flush()
    return len(pieces)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def _keyword_score(query: str, content: str) -> float:
    terms = [term for term in query.lower().split() if len(term) > 2]
    if not terms:
        return 0.0
    haystack = content.lower()
    return float(sum(haystack.count(term) for term in terms))


async def search_documents(db, user_id: int, query: str, limit: int = 4) -> list[Chunk]:
    rows = db.query(Chunk).filter(Chunk.user_id == user_id).all()
    if not rows:
        return []
    query_vector = None
    if any(row.embedding for row in rows):
        embedded = await embed_texts([query])
        if embedded and embedded[0]:
            query_vector = embedded[0]
    scored = []
    for row in rows:
        if query_vector and row.embedding:
            try:
                score = _cosine(query_vector, json.loads(row.embedding))
            except json.JSONDecodeError:
                score = _keyword_score(query, row.content)
        else:
            score = _keyword_score(query, row.content)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:limit]]


def read_marked_file(db, user_id: int, relative: str) -> str:
    base = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user_id).one_or_none()
    if base is None or not base.folder_path:
        raise FileNotFoundError("Nenhuma pasta indicada.")
    path = resolve_inside(Path(base.folder_path), relative)
    if not path.is_file():
        raise FileNotFoundError("Arquivo não encontrado na pasta indicada.")
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:12_000]
