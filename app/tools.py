"""Ferramentas do orquestrador: web, documentos e memória."""

import asyncio
import re
from html import unescape
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.config import get_settings
from app.db import Memory
from app.knowledge import read_marked_file, search_documents
from app.policy import folder_access_enabled

WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Busca informação atual na web. Use para notícias, cotações, cargos, clima e qualquer fato que mude.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "O que buscar"}},
            "required": ["query"],
        },
    },
}

DOCS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Busca nos documentos e na pasta que a pessoa indicou.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "O que procurar nos documentos"}},
            "required": ["query"],
        },
    },
}

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Lê um arquivo dentro da pasta indicada. O caminho é relativo a essa pasta.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Caminho relativo do arquivo"}},
            "required": ["path"],
        },
    },
}

REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": "Grava um fato que a pessoa pediu para lembrar em conversas futuras.",
        "parameters": {
            "type": "object",
            "properties": {"fact": {"type": "string", "description": "O fato, em uma frase"}},
            "required": ["fact"],
        },
    },
}

REMEMBER_RE = re.compile(
    r"^\s*(?:por favor,?\s*)?(?:lembre(?:-se)?|guarda|guarde|memorize)(?: que)?\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
WEB_RE = re.compile(
    r"\b(busque|busca na web|pesquise|na web|na internet|cotação|preço atual|notícias de hoje|noticias de hoje)\b",
    re.IGNORECASE,
)
FILE_RE = re.compile(r"\b(arquivo|pasta|documento|documentos)\b", re.IGNORECASE)


def tool_schemas(has_documents: bool, allow_files: bool, allow_remember: bool) -> list[dict]:
    tools = [WEB_TOOL]
    if allow_remember:
        tools.append(REMEMBER_TOOL)
    if has_documents:
        tools.append(DOCS_TOOL)
    if allow_files:
        tools.append(READ_TOOL)
    return tools


def planned_tools(user_text: str, has_documents: bool, allow_files: bool) -> list[dict]:
    """Cobre o pedido quando o modelo local não chama a ferramenta sozinho."""
    planned = []
    remembered = REMEMBER_RE.match(user_text.strip())
    if remembered:
        planned.append({"name": "remember", "arguments": {"fact": remembered.group(1).strip()}})
    if WEB_RE.search(user_text):
        planned.append({"name": "search_web", "arguments": {"query": user_text.strip()[:300]}})
    if has_documents and FILE_RE.search(user_text):
        planned.append({"name": "search_documents", "arguments": {"query": user_text.strip()[:300]}})
    if not allow_files:
        planned = [item for item in planned if item["name"] != "read_file"]
    return planned


def remember_fact(db, user_id: int, fact: str) -> str:
    clean = " ".join(fact.split())
    if not clean:
        return "Nada para gravar."
    clean = clean[:500]
    existing = (
        db.query(Memory).filter(Memory.user_id == user_id, Memory.content == clean).one_or_none()
    )
    if existing is None:
        db.add(Memory(user_id=user_id, content=clean))
        db.commit()
    return "Memória gravada."


def recent_memories(db, user_id: int, limit: int = 30) -> list[str]:
    rows = (
        db.query(Memory)
        .filter(Memory.user_id == user_id)
        .order_by(Memory.id.desc())
        .limit(limit)
        .all()
    )
    return [row.content for row in reversed(rows)]


def _format_hits(results: list[dict]) -> str:
    if not results:
        return "A busca não devolveu resultados."
    lines = []
    for index, item in enumerate(results[:5], start=1):
        title = item.get("title") or "Sem título"
        body = (item.get("body") or "").strip()
        href = item.get("href") or ""
        lines.append(f"{index}. {title}\n{body}\n{href}")
    return "\n\n".join(lines)


def _decode_ddg_href(href: str) -> str:
    if "uddg=" in href:
        parsed = parse_qs(urlparse(href).query)
        target = parsed.get("uddg", [""])[0]
        if target:
            return unquote(target)
    return href


def _strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    return " ".join(unescape(text).split())


async def _search_searxng(query: str) -> list[dict]:
    base = get_settings().searxng_url.strip().rstrip("/")
    if not base:
        return []
    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
        response = await client.get(f"{base}/search", params={"q": query, "format": "json"})
        response.raise_for_status()
        payload = response.json()
    results = []
    for item in payload.get("results") or []:
        results.append(
            {
                "title": item.get("title") or "",
                "body": item.get("content") or "",
                "href": item.get("url") or "",
            }
        )
        if len(results) >= 5:
            break
    return results


async def _search_html(query: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
        response = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        html = response.text
    blocks = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>',
        html,
        flags=re.DOTALL,
    )
    results = []
    for href, title, snippet in blocks[:5]:
        results.append(
            {
                "title": _strip_tags(title),
                "body": _strip_tags(snippet),
                "href": _decode_ddg_href(unescape(href)),
            }
        )
    return results


def _search_ddgs(query: str) -> list[dict]:
    from ddgs import DDGS

    rows = DDGS().text(query, max_results=5)
    results = []
    for item in rows:
        results.append(
            {
                "title": item.get("title") or "",
                "body": item.get("body") or "",
                "href": item.get("href") or item.get("url") or "",
            }
        )
    return results


async def search_web(query: str) -> str:
    query = query.strip()
    if not query:
        return "Informe o que buscar."
    if get_settings().searxng_url.strip():
        try:
            return _format_hits(await _search_searxng(query))
        except (httpx.HTTPError, ValueError) as exc:
            searx_error = str(exc)
    else:
        searx_error = ""
    try:
        hits = await asyncio.to_thread(_search_ddgs, query)
        if hits:
            return _format_hits(hits)
    except Exception:
        hits = []
    try:
        hits = await _search_html(query)
        if hits:
            return _format_hits(hits)
    except httpx.HTTPError as exc:
        detail = searx_error or str(exc)
        return f"A busca na web falhou: {detail}"
    if searx_error:
        return f"A busca na web falhou: {searx_error}"
    return "A busca não devolveu resultados."


async def execute_tool(db, user_id: int, name: str, arguments: dict) -> str:
    try:
        if name == "search_web":
            return await search_web(str(arguments.get("query") or ""))
        if name == "search_documents":
            rows = await search_documents(db, user_id, str(arguments.get("query") or ""))
            if not rows:
                return "Nenhum trecho encontrado nos documentos desta pessoa."
            parts = [f"Fonte: {row.source}\n{row.content}" for row in rows]
            return "\n\n".join(parts)[:6000]
        if name == "read_file":
            if not folder_access_enabled(get_settings().deploy_mode):
                return "A leitura da pasta local está desligada neste modo."
            return read_marked_file(db, user_id, str(arguments.get("path") or ""))
        if name == "remember":
            return remember_fact(db, user_id, str(arguments.get("fact") or ""))
        return f"Ferramenta desconhecida: {name}"
    except PermissionError:
        return "Esse caminho está fora da pasta indicada."
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"A ferramenta {name} falhou: {exc}"
