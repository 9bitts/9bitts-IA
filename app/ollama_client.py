"""Fala com o Ollama. A URL vem da configuração, então o mesmo app aponta para outro servidor."""

import json
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings

TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=60.0, pool=5.0)


def _url(path: str) -> str:
    return get_settings().ollama_base_url + path


async def ollama_online() -> bool:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            response = await client.get(_url("/api/tags"))
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def installed_models() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get(_url("/api/tags"))
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError:
        return []
    names = []
    for item in payload.get("models") or []:
        name = item.get("name") or item.get("model") or ""
        if name:
            names.append(name)
    return names


def model_is_installed(want: str, installed: list[str]) -> bool:
    for name in installed:
        if name == want or name.startswith(want + ":"):
            return True
    return False


async def running_models() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            response = await client.get(_url("/api/ps"))
            response.raise_for_status()
            return list(response.json().get("models") or [])
    except httpx.HTTPError:
        return []


async def embed_texts(texts: list[str]) -> list[list[float]] | None:
    if not texts:
        return []
    payload = {"model": get_settings().embed_model, "input": texts}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=4.0, read=60.0, write=30.0, pool=4.0)) as client:
            response = await client.post(_url("/api/embed"), json=payload)
            if response.status_code == 404:
                vectors = []
                for text in texts:
                    one = await client.post(
                        _url("/api/embeddings"),
                        json={"model": get_settings().embed_model, "prompt": text},
                    )
                    one.raise_for_status()
                    vectors.append(list(one.json().get("embedding") or []))
                return vectors
            response.raise_for_status()
            vectors = response.json().get("embeddings") or []
            return [list(item) for item in vectors]
    except httpx.HTTPError:
        return None


async def chat_stream(model: str, messages: list[dict], tools: list[dict] | None) -> AsyncIterator[dict]:
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": 0.6, "num_ctx": 4096},
    }
    if tools:
        body["tools"] = tools
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream("POST", _url("/api/chat"), json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                yield json.loads(line)


def merge_tool_calls(existing: list[dict], incoming: list[dict]) -> list[dict]:
    for index, call in enumerate(incoming):
        function = call.get("function") or {}
        name = function.get("name") or ""
        arguments = function.get("arguments")
        if index >= len(existing):
            existing.append({"name": name, "arguments": "" if arguments is None else arguments})
            continue
        slot = existing[index]
        if name and len(name) >= len(slot.get("name") or ""):
            slot["name"] = name
        if isinstance(arguments, dict):
            slot["arguments"] = arguments
        elif isinstance(arguments, str) and arguments:
            previous = slot.get("arguments")
            if isinstance(previous, dict):
                slot["arguments"] = arguments
            elif not previous:
                slot["arguments"] = arguments
            elif arguments.startswith(str(previous)):
                slot["arguments"] = arguments
            elif not str(previous).endswith(arguments):
                slot["arguments"] = str(previous) + arguments
    return existing


def normalize_tool_calls(raw: list[dict]) -> list[dict]:
    calls = []
    for index, call in enumerate(raw):
        name = (call.get("name") or "").strip()
        if not name and isinstance(call.get("function"), dict):
            name = (call["function"].get("name") or "").strip()
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        if name:
            calls.append({"name": name, "arguments": arguments})
        elif arguments:
            calls.append({"name": "", "arguments": arguments, "id": index})
    return [call for call in calls if call.get("name")]
