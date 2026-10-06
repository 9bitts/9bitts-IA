"""Loop de conversa: persona, memória, ferramentas e streaming."""

import json
from collections.abc import AsyncIterator
from datetime import datetime

import httpx
from fastapi import HTTPException, Request

from app.config import get_settings
from app import db as database
from app.db import Conversation, Message, get_setting, increment_usage, usage_today
from app.knowledge import chunk_count
from app.ollama_client import chat_stream, merge_tool_calls, normalize_tool_calls
from app.policy import folder_access_enabled
from app.tools import REMEMBER_RE, execute_tool, planned_tools, recent_memories


def assert_can_send(db, user_id: int) -> None:
    settings = get_settings()
    if settings.deploy_mode != "product":
        return
    if usage_today(db, user_id) >= settings.product_daily_limit:
        raise HTTPException(status_code=429, detail="Limite diário de mensagens atingido.")
    increment_usage(db, user_id)


def persona_text(db, user_id: int) -> str:
    custom = get_setting(db, user_id, "persona")
    if custom:
        return custom
    path = get_settings().persona_path
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return "Você é o assistente da 9bitts. Fale de forma direta e resolva a pergunta."


def build_system(persona: str, memories: list[str], allow_files: bool) -> str:
    lines = [
        persona.strip(),
        "",
        f"Data de hoje: {datetime.now().strftime('%d/%m/%Y')}.",
        "Responda no idioma da pessoa. O padrão é português.",
    ]
    if memories:
        lines.append("")
        lines.append("Fatos que esta pessoa pediu para lembrar:")
        lines.extend(f"- {item}" for item in memories)
    else:
        lines.append("")
        lines.append("Ainda não há fatos gravados na memória.")
    lines.append("")
    if allow_files:
        lines.append("Se a pergunta falar de arquivo, pasta ou documento, use a busca nos documentos.")
    else:
        lines.append("Você não tem acesso ao disco deste computador. Use só os documentos enviados por esta conta.")
    lines.append("Se precisar de fato atual, use a busca na web e cite o que ela devolver.")
    lines.append("Não invente o resultado de uma ferramenta.")
    return "\n".join(lines)


def history_for_model(rows: list[Message]) -> list[dict]:
    messages = []
    for row in rows:
        if row.role not in {"user", "assistant"}:
            continue
        content = (row.content or "").strip()
        if not content:
            continue
        messages.append({"role": row.role, "content": content})
    return messages[-30:]


def _preview(text: str) -> str:
    clean = " ".join(text.split())
    return clean[:180]


async def _one_round(model: str, messages: list[dict], tools: list[dict] | None):
    content_parts: list[str] = []
    accumulated: list[dict] = []
    final_calls = None
    async for chunk in chat_stream(model, messages, tools):
        message = chunk.get("message") or {}
        piece = message.get("content") or ""
        if piece:
            content_parts.append(piece)
            yield ("token", piece)
        incoming = message.get("tool_calls") or []
        if incoming:
            merge_tool_calls(accumulated, incoming)
        if chunk.get("done") and incoming:
            final_calls = incoming
    raw = final_calls if final_calls else accumulated
    yield ("calls", normalize_tool_calls(raw))
    yield ("content", "".join(content_parts))


async def stream_reply(
    request: Request,
    user_id: int,
    conversation_id: int,
    model: str,
    history: list[dict],
    persona: str,
    memories: list[str],
    has_documents: bool,
) -> AsyncIterator[str]:
    settings = get_settings()
    allow_files = folder_access_enabled(settings.deploy_mode) and bool(
        _folder_for(user_id)
    )
    messages = [{"role": "system", "content": build_system(persona, memories, allow_files)}, *history]
    user_text = next((item["content"] for item in reversed(history) if item["role"] == "user"), "")
    traces: list[dict] = []
    parts: list[str] = []

    preplanned = planned_tools(user_text, has_documents, allow_files)
    if preplanned:
        notes = []
        for call in preplanned:
            async for event in _run_call(user_id, call, traces):
                yield event
            notes.append(f"{call['name']}:\n{traces[-1]['result']}")
        if messages and messages[-1]["role"] == "user":
            messages[-1] = {
                "role": "user",
                "content": messages[-1]["content"]
                + "\n\nResponda só com o resultado abaixo. Copie códigos e números como estão. Não invente.\n"
                + "\n\n".join(notes),
            }

    try:
        for _round in range(3):
            if await request.is_disconnected():
                break
            round_content: list[str] = []
            calls: list[dict] = []
            use_tools = None
            async for kind, payload in _one_round(model, messages, use_tools):
                if kind == "token":
                    round_content.append(payload)
                    parts.append(payload)
                    yield _sse("token", {"text": payload})
                elif kind == "calls":
                    calls = payload
            if not calls:
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(round_content),
                    "tool_calls": [
                        {"function": {"name": call["name"], "arguments": call["arguments"]}}
                        for call in calls
                    ],
                }
            )
            for call in calls:
                if await request.is_disconnected():
                    break
                if call["name"] == "remember" and not REMEMBER_RE.search(user_text):
                    traces.append(
                        {
                            "name": "remember",
                            "input": call.get("arguments") or {},
                            "preview": "Não gravei: ninguém pediu para lembrar.",
                            "result": "Não gravei nada. Só grave quando a pessoa pedir para lembrar.",
                        }
                    )
                    yield _sse(
                        "tool",
                        {
                            "name": "remember",
                            "status": "done",
                            "preview": traces[-1]["preview"],
                        },
                    )
                    messages.append(
                        {"role": "tool", "name": "remember", "content": traces[-1]["result"]}
                    )
                    continue
                async for event in _run_call(user_id, call, traces):
                    yield event
                messages.append(
                    {
                        "role": "tool",
                        "name": call["name"],
                        "content": traces[-1]["result"][:6000],
                    }
                )
        content = "".join(parts).strip()
    except httpx.HTTPError as exc:
        yield _sse("error", {"message": f"O modelo não respondeu: {exc}"})
        content = "".join(parts).strip() or f"O modelo não respondeu: {exc}"

    if not content.strip():
        content = "Não consegui formular a resposta."
        yield _sse("token", {"text": content})

    message_id = _save_assistant(conversation_id, model, content, traces)
    yield _sse("done", {"message_id": message_id, "content": content})


async def _run_call(user_id: int, call: dict, traces: list[dict]):
    yield _sse(
        "tool",
        {"name": call["name"], "status": "start", "input": call.get("arguments") or {}},
    )
    db = database.SessionLocal()
    try:
        result = await execute_tool(db, user_id, call["name"], call.get("arguments") or {})
    finally:
        db.close()
    traces.append(
        {
            "name": call["name"],
            "input": call.get("arguments") or {},
            "preview": _preview(result),
            "result": result,
        }
    )
    yield _sse("tool", {"name": call["name"], "status": "done", "preview": _preview(result)})


def _folder_for(user_id: int) -> str | None:
    if not folder_access_enabled(get_settings().deploy_mode):
        return None
    db = database.SessionLocal()
    try:
        from app.db import KnowledgeBase

        base = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user_id).one_or_none()
        return base.folder_path if base else None
    finally:
        db.close()


def _save_assistant(conversation_id: int, model: str, content: str, traces: list[dict]) -> int:
    db = database.SessionLocal()
    try:
        conversation = db.get(Conversation, conversation_id)
        public = [
            {"name": item["name"], "input": item["input"], "preview": item["preview"]} for item in traces
        ]
        message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            model=model,
            tool_trace=json.dumps(public, ensure_ascii=False) if public else None,
        )
        db.add(message)
        if conversation is not None:
            conversation.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(message)
        return message.id
    finally:
        db.close()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def documents_ready(db, user_id: int) -> bool:
    return chunk_count(db, user_id) > 0
