"""Assistente local. O mesmo processo atende o modo pessoal e o modo produto."""

import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import db as database
from app.auth import (
    clear_session_cookie,
    hash_password,
    new_session,
    require_admin,
    require_user,
    set_session_cookie,
    verify_password,
)
from app.chat import (
    assert_can_send,
    documents_ready,
    history_for_model,
    persona_text,
    recent_memories,
    stream_reply,
)
from app.compute import detect_compute
from app.config import PHASE1_ACCEPTANCE, ROOT, get_settings
from app.db import Conversation, Message, User, set_setting, usage_today
from app.knowledge import (
    add_upload,
    chunk_count,
    ensure_base,
    job_status,
    sources_for,
    start_reindex,
)
from app.ollama_client import installed_models, model_is_installed, ollama_online, running_models
from app.policy import database_kind, folder_access_enabled, inference_location
from app.tools import recent_memories as load_memories

STATIC = ROOT / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    database.init_db()
    yield


app = FastAPI(title="9bitts IA", lifespan=lifespan)


def get_db():
    if database.SessionLocal is None:
        database.init_db()
    db = database.SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def current_user(request: Request, response: Response, db: Session = Depends(get_db)) -> User:
    user, token = require_user(db, request)
    if token:
        db.commit()
        set_session_cookie(response, token)
    return user


class LoginBody(BaseModel):
    email: str
    password: str


class UserBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=200)


class ConversationBody(BaseModel):
    model: str | None = None


class ConversationPatch(BaseModel):
    title: str | None = Field(default=None, max_length=160)
    model: str | None = None


class MessageBody(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    model: str | None = None


class SettingsBody(BaseModel):
    persona: str | None = Field(default=None, max_length=8000)
    folder_path: str | None = None


def _model_name(requested: str | None, fallback: str) -> str:
    settings = get_settings()
    chosen = requested or fallback
    if chosen not in {settings.fast_model, settings.quality_model}:
        raise HTTPException(status_code=400, detail="Escolha o modelo rápido ou o modelo melhor.")
    return chosen


def _owned_conversation(db: Session, user: User, conversation_id: int) -> Conversation:
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .one_or_none()
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")
    return conversation


def _public_user(user: User) -> dict:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "is_admin": user.is_admin,
        "deploy_mode": get_settings().deploy_mode,
    }


def _clip_title(text: str) -> str:
    clean = " ".join(text.split())
    return (clean[:80] or "Nova conversa")


@app.get("/api/me")
def me(user: User = Depends(current_user)):
    return _public_user(user)


@app.post("/api/login")
def login(body: LoginBody, response: Response, db: Session = Depends(get_db)):
    settings = get_settings()
    if settings.deploy_mode != "product":
        raise HTTPException(status_code=400, detail="O modo pessoal entra direto, sem senha.")
    email = body.email.strip().lower()
    user = db.query(User).filter(User.email == email).one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="E-mail ou senha inválidos.")
    set_session_cookie(response, new_session(db, user))
    return _public_user(user)


@app.post("/api/logout")
def logout(response: Response, user: User = Depends(current_user)):
    del user
    clear_session_cookie(response)
    return {"ok": True}


@app.get("/api/status")
async def status(user: User = Depends(current_user), db: Session = Depends(get_db)):
    settings = get_settings()
    installed = await installed_models()
    running = await running_models()
    vram = sum(int(item.get("size_vram") or 0) for item in running)
    compute = detect_compute()
    compute["using_gpu"] = vram > 0
    compute["size_vram"] = vram
    limit = settings.product_daily_limit if settings.deploy_mode == "product" else None
    return {
        "deploy_mode": settings.deploy_mode,
        "inference_location": inference_location(settings.ollama_base_url),
        "ollama_base_url": settings.ollama_base_url if settings.deploy_mode == "personal" or user.is_admin else None,
        "ollama_online": await ollama_online(),
        "database": database_kind(settings.database_url),
        "folder_access": folder_access_enabled(settings.deploy_mode),
        "models": {
            "fast": settings.fast_model,
            "quality": settings.quality_model,
            "embed": settings.embed_model,
            "installed": installed,
            "fast_ready": model_is_installed(settings.fast_model, installed),
            "quality_ready": model_is_installed(settings.quality_model, installed),
        },
        "compute": compute,
        "daily_limit": limit,
        "daily_used": usage_today(db, user.id) if limit else 0,
        "phase1_acceptance": list(PHASE1_ACCEPTANCE),
    }


@app.get("/api/conversations")
def list_conversations(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .all()
    )
    return [
        {
            "id": row.id,
            "title": row.title,
            "model": row.model,
            "updated_at": row.updated_at.isoformat(),
        }
        for row in rows
    ]


@app.post("/api/conversations")
def create_conversation(
    body: ConversationBody,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    model = _model_name(body.model, settings.fast_model)
    conversation = Conversation(user_id=user.id, title="Nova conversa", model=model)
    db.add(conversation)
    db.flush()
    return {"id": conversation.id, "title": conversation.title, "model": conversation.model}


@app.patch("/api/conversations/{conversation_id}")
def patch_conversation(
    conversation_id: int,
    body: ConversationPatch,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    conversation = _owned_conversation(db, user, conversation_id)
    if body.title is not None:
        conversation.title = _clip_title(body.title)
    if body.model is not None:
        conversation.model = _model_name(body.model, conversation.model)
    conversation.updated_at = datetime.utcnow()
    return {"id": conversation.id, "title": conversation.title, "model": conversation.model}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    conversation = _owned_conversation(db, user, conversation_id)
    db.delete(conversation)
    return Response(status_code=204)


@app.get("/api/conversations/{conversation_id}/messages")
def list_messages(
    conversation_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    conversation = _owned_conversation(db, user, conversation_id)
    rows = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.id)
        .all()
    )
    payload = []
    for row in rows:
        trace = []
        if row.tool_trace:
            try:
                trace = json.loads(row.tool_trace)
            except json.JSONDecodeError:
                trace = []
        payload.append(
            {
                "id": row.id,
                "role": row.role,
                "content": row.content,
                "tool_trace": trace,
                "model": row.model,
            }
        )
    return payload


@app.post("/api/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: int,
    body: MessageBody,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    conversation = _owned_conversation(db, user, conversation_id)
    model = _model_name(body.model, conversation.model)
    installed = await installed_models()
    if not model_is_installed(model, installed):
        raise HTTPException(
            status_code=409,
            detail=f"O modelo {model} ainda não está neste computador.",
        )
    assert_can_send(db, user.id)
    conversation.model = model
    text = body.content.strip()
    db.add(Message(conversation_id=conversation.id, role="user", content=text))
    if conversation.title in {"", "Nova conversa"}:
        conversation.title = _clip_title(text)
    conversation.updated_at = datetime.utcnow()
    db.flush()
    history = history_for_model(
        db.query(Message).filter(Message.conversation_id == conversation.id).order_by(Message.id).all()
    )
    persona = persona_text(db, user.id)
    memories = load_memories(db, user.id)
    has_documents = documents_ready(db, user.id)
    title = conversation.title
    user_id = user.id
    conversation_pk = conversation.id
    db.commit()

    async def generate():
        yield _sse_comment(title)
        async for event in stream_reply(
            request,
            user_id,
            conversation_pk,
            model,
            history,
            persona,
            memories,
            has_documents,
        ):
            yield event

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse_comment(title: str) -> str:
    return f"event: meta\ndata: {json.dumps({'title': title}, ensure_ascii=False)}\n\n"


@app.get("/api/settings")
def read_settings(user: User = Depends(current_user), db: Session = Depends(get_db)):
    settings = get_settings()
    base = ensure_base(db, user.id)
    allow_folder = folder_access_enabled(settings.deploy_mode)
    return {
        "persona": persona_text(db, user.id),
        "folder_path": base.folder_path if allow_folder else None,
        "folder_access": allow_folder,
        "folder_locked_reason": (
            None
            if allow_folder
            else "No modo produto a pasta deste computador fica desligada. O conhecimento entra por documento, separado por cliente."
        ),
        "knowledge": {
            "chunks": chunk_count(db, user.id),
            "sources": sources_for(db, user.id),
            **job_status(user.id),
        },
        "deploy": {
            "mode": settings.deploy_mode,
            "inference_location": inference_location(settings.ollama_base_url),
            "ollama_base_url": settings.ollama_base_url if allow_folder or user.is_admin else None,
            "database": database_kind(settings.database_url),
            "daily_limit": settings.product_daily_limit if settings.deploy_mode == "product" else None,
        },
    }


@app.put("/api/settings")
def write_settings(
    body: SettingsBody,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    if body.persona is not None:
        text = body.persona.strip()
        if not text:
            raise HTTPException(status_code=400, detail="A persona não pode ficar vazia.")
        set_setting(db, user.id, "persona", text)
    if body.folder_path is not None:
        if not folder_access_enabled(get_settings().deploy_mode):
            raise HTTPException(
                status_code=403,
                detail="No modo produto a pasta deste computador fica desligada.",
            )
        base = ensure_base(db, user.id)
        raw = body.folder_path.strip()
        if raw == "":
            base.folder_path = None
            from app.db import Chunk

            db.query(Chunk).filter(Chunk.user_id == user.id, Chunk.origin == "folder").delete()
        else:
            folder = Path(raw).expanduser()
            if not folder.is_dir():
                raise HTTPException(status_code=400, detail="Essa pasta não existe neste computador.")
            base.folder_path = str(folder.resolve())
            db.commit()
            start_reindex(user.id, base.folder_path)
    db.flush()
    return read_settings(user, db)


@app.post("/api/knowledge/reindex")
def reindex(user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not folder_access_enabled(get_settings().deploy_mode):
        raise HTTPException(status_code=403, detail="No modo produto a pasta deste computador fica desligada.")
    base = ensure_base(db, user.id)
    if not base.folder_path:
        raise HTTPException(status_code=400, detail="Indique uma pasta antes de indexar.")
    start_reindex(user.id, base.folder_path)
    return job_status(user.id)


@app.get("/api/knowledge")
def knowledge(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return {
        "chunks": chunk_count(db, user.id),
        "sources": sources_for(db, user.id),
        **job_status(user.id),
    }


@app.post("/api/knowledge/documents")
async def upload_document(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
):
    name = Path(file.filename or "documento.txt").name
    suffix = Path(name).suffix.lower()
    if suffix not in {".txt", ".md", ".markdown", ".csv", ".json"}:
        raise HTTPException(status_code=400, detail="Envie um arquivo de texto (.txt, .md, .csv ou .json).")
    raw = await file.read()
    if len(raw) > 1_000_000:
        raise HTTPException(status_code=400, detail="O arquivo passa de 1 MB.")
    text = raw.decode("utf-8", errors="replace")
    stored = await add_upload(db, user.id, name, text)
    return {"stored": stored, "sources": sources_for(db, user.id), "chunks": chunk_count(db, user.id)}


@app.get("/api/admin/users")
def admin_users(user: User = Depends(current_user), db: Session = Depends(get_db)):
    require_admin(user)
    rows = db.query(User).order_by(User.id).all()
    return [
        {"id": row.id, "name": row.name, "email": row.email, "is_admin": row.is_admin}
        for row in rows
    ]


@app.post("/api/admin/users")
def admin_create_user(body: UserBody, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require_admin(user)
    if get_settings().deploy_mode != "product":
        raise HTTPException(status_code=403, detail="Contas de cliente só existem no modo produto.")
    email = body.email.strip().lower()
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise HTTPException(status_code=400, detail="E-mail inválido.")
    if db.query(User).filter(User.email == email).one_or_none():
        raise HTTPException(status_code=409, detail="Já existe uma conta com esse e-mail.")
    created = User(
        email=email,
        name=body.name.strip(),
        password_hash=hash_password(body.password),
        is_admin=False,
    )
    db.add(created)
    db.flush()
    return {"id": created.id, "name": created.name, "email": created.email, "is_admin": False}


@app.get("/api/memories")
def memories(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return {"items": recent_memories(db, user.id)}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
