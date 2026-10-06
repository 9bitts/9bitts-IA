"""Confere contas, isolamento e o desenho do modo produto, sem precisar do modelo."""

import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp())
os.environ["DEPLOY_MODE"] = "product"
os.environ["DATABASE_URL"] = "sqlite:///" + (_tmp / "checks.db").as_posix()
os.environ["OLLAMA_BASE_URL"] = "http://gpu.internal:11434"
os.environ["PRODUCT_ADMIN_EMAIL"] = "admin@9bitts.dev"
os.environ["PRODUCT_ADMIN_PASSWORD"] = "uma-senha-forte"
os.environ["PRODUCT_DAILY_LIMIT"] = "2"
os.environ["SEARXNG_URL"] = ""

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.chat import assert_can_send
from app.config import PHASE1_ACCEPTANCE, get_settings
from app.db import Chunk, User
from app.knowledge import ensure_base
from app.main import app
from app.policy import database_kind, folder_access_enabled, inference_location, resolve_inside
from app.tools import planned_tools


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    settings = get_settings()
    if "8b" not in settings.fast_model or "14b" not in settings.quality_model:
        fail("os modelos da fase 1 precisam ser o de 8B e o de 14B")
    if len(PHASE1_ACCEPTANCE) != 4:
        fail("o critério da fase 1 precisa ter os quatro pontos")
    persona = settings.persona_path.read_text(encoding="utf-8")
    if "direta" not in persona or "resolver" not in persona:
        fail("a persona precisa ser direta e existir para resolver")
    if folder_access_enabled("personal") is not True or folder_access_enabled("product") is not False:
        fail("a pasta local só pode existir no modo pessoal")
    if inference_location("http://127.0.0.1:11434") != "local":
        fail("localhost precisa contar como inferência local")
    if inference_location("http://gpu.internal:11434") != "remote":
        fail("um host remoto precisa contar como servidor de inferência")
    if database_kind("postgresql+psycopg://user:pass@localhost/novebitts") != "postgres":
        fail("Postgres precisa ser reconhecido pela URL")
    if database_kind(settings.database_url) != "sqlite":
        fail("o teste usa sqlite")

    root = _tmp / "pasta"
    root.mkdir()
    (root / "nota.txt").write_text("oi", encoding="utf-8")
    assert resolve_inside(root, "nota.txt").name == "nota.txt"
    try:
        resolve_inside(root, "../segredo.txt")
        fail("caminho fora da pasta deveria ser recusado")
    except PermissionError:
        pass

    remembered = planned_tools("Lembre que meu editor favorito é o Cursor.", False, False)
    if not any(item["name"] == "remember" for item in remembered):
        fail("pedido de memória não foi reconhecido")
    if not any(item["name"] == "search_web" for item in planned_tools("Busque na web a cotação do dólar.", False, False)):
        fail("pedido de busca não foi reconhecido")

    with TestClient(app) as client:
        me = client.get("/api/me")
        if me.status_code != 401:
            fail("modo produto precisa pedir login")
        bad = client.post("/api/login", json={"email": "admin@9bitts.dev", "password": "errada"})
        if bad.status_code != 401:
            fail("senha errada precisa falhar")
        ok = client.post("/api/login", json={"email": "admin@9bitts.dev", "password": "uma-senha-forte"})
        if ok.status_code != 200 or not ok.json()["is_admin"]:
            fail("admin não entrou")
        folder = client.put("/api/settings", json={"folder_path": str(root)})
        if folder.status_code != 403:
            fail("modo produto precisa recusar a pasta deste computador")
        status = client.get("/api/status")
        body = status.json()
        if body["inference_location"] != "remote" or body["folder_access"] is not False:
            fail("status do produto não reflete servidor remoto e pasta desligada")
        if body["database"] != "sqlite":
            fail("status do banco divergiu")
        created = client.post("/api/conversations", json={})
        if created.status_code != 200:
            fail("admin não criou conversa")
        conversation_id = created.json()["id"]
        second = client.post(
            "/api/admin/users",
            json={"name": "Cliente", "email": "cliente@9bitts.dev", "password": "senha-do-cliente"},
        )
        if second.status_code != 200:
            fail("não criou a conta do cliente")
        client.cookies.clear()
        client.post("/api/login", json={"email": "cliente@9bitts.dev", "password": "senha-do-cliente"})
        listing = client.get("/api/conversations")
        if listing.json() != []:
            fail("cliente viu conversa de outra conta")
        hidden = client.get(f"/api/conversations/{conversation_id}/messages")
        if hidden.status_code != 404:
            fail("cliente abriu conversa de outra conta")
        upload = client.post(
            "/api/knowledge/documents",
            files={"file": ("nota.txt", b"segredo do cliente", "text/plain")},
        )
        if upload.status_code != 200:
            fail("cliente não guardou documento")

        from app import db as database

        db = database.SessionLocal()
        try:
            admin = db.query(User).filter(User.email == "admin@9bitts.dev").one()
            client_user = db.query(User).filter(User.email == "cliente@9bitts.dev").one()
            base = ensure_base(db, admin.id)
            db.add(
                Chunk(
                    user_id=admin.id,
                    knowledge_base_id=base.id,
                    source="admin.txt",
                    origin="upload",
                    content="segredo do admin que o cliente nao pode ver",
                )
            )
            db.commit()
            visible = db.query(Chunk).filter(Chunk.user_id == client_user.id).all()
            leaked = [row for row in visible if "nao pode ver" in row.content]
            if leaked:
                fail("documento de outra conta apareceu na conta do cliente")
            import asyncio

            from app.knowledge import search_documents

            found = asyncio.run(search_documents(db, client_user.id, "segredo do admin que o cliente nao pode ver"))
            if any("nao pode ver" in row.content for row in found):
                fail("a busca devolveu documento de outra conta")
            assert_can_send(db, client_user.id)
            db.commit()
            assert_can_send(db, client_user.id)
            db.commit()
            try:
                assert_can_send(db, client_user.id)
                fail("limite diário não segurou a terceira mensagem")
            except HTTPException as exc:
                if exc.status_code != 429:
                    fail("limite diário devolveu outro erro")
        finally:
            db.close()

    print("ok: persona, modelos 8B/14B, critério da fase 1")
    print("ok: contas separadas, pasta local desligada, limite diário")
    print("ok: inferência remota e Postgres reconhecidos na configuração")


if __name__ == "__main__":
    main()
