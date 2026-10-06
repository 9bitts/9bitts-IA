import hashlib
import secrets
from datetime import datetime

from fastapi import HTTPException, Request, Response

from app.config import get_settings
from app.db import SessionToken, User, purge_expired_sessions, session_ttl

COOKIE = "session"


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored or "$" not in stored:
        return False
    salt, digest = stored.split("$", 1)
    check = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return secrets.compare_digest(check.hex(), digest)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_session(db, user: User) -> str:
    purge_expired_sessions(db)
    token = secrets.token_urlsafe(32)
    db.add(SessionToken(token_hash=hash_token(token), user_id=user.id, expires_at=session_ttl()))
    db.flush()
    return token


def user_from_request(db, request: Request) -> User | None:
    raw = request.cookies.get(COOKIE)
    if not raw:
        return None
    row = (
        db.query(SessionToken)
        .filter(SessionToken.token_hash == hash_token(raw), SessionToken.expires_at >= datetime.utcnow())
        .one_or_none()
    )
    if row is None:
        return None
    return db.get(User, row.user_id)


def set_session_cookie(response: Response, token: str, secure: bool = False) -> None:
    response.set_cookie(
        COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=secure,
        max_age=60 * 60 * 24 * 30,
        path="/",
    )


def https_request(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "")
    return forwarded.split(",")[0].strip().lower() == "https"


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE, path="/")


def require_user(db, request: Request) -> tuple[User, str | None]:
    """Devolve o usuário e, no modo pessoal, um token novo quando ainda não há sessão."""
    user = user_from_request(db, request)
    if user is not None:
        return user, None
    settings = get_settings()
    if settings.deploy_mode == "personal":
        user = db.query(User).filter(User.email == "diego@localhost").one()
        return user, new_session(db, user)
    raise HTTPException(status_code=401, detail="Entre com e-mail e senha.")


def require_admin(user: User) -> None:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Só quem administra as contas pode fazer isso.")
