"""SQLite neste notebook. Postgres entra quando DATABASE_URL aponta para ele."""

from datetime import datetime, timedelta

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from app.config import ROOT, get_settings

_engine = None
SessionLocal = None


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SessionToken(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(160), default="Nova conversa")
    model: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text, default="")
    tool_trace: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Memory(Base):
    __tablename__ = "memories"
    __table_args__ = (UniqueConstraint("user_id", "content", name="uq_memory_user_content"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    name: Mapped[str] = mapped_column(String(160), default="Documentos")
    folder_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    knowledge_base_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    source: Mapped[str] = mapped_column(String(500))
    origin: Mapped[str] = mapped_column(String(20), default="folder")
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[str | None] = mapped_column(Text, nullable=True)


class UsageDay(Base):
    __tablename__ = "usage_days"
    __table_args__ = (UniqueConstraint("user_id", "day", name="uq_usage_user_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    day: Mapped[str] = mapped_column(String(10))
    message_count: Mapped[int] = mapped_column(Integer, default=0)


class Setting(Base):
    __tablename__ = "settings"
    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_setting_user_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(80))
    value: Mapped[str] = mapped_column(Text, default="")


def _sqlite_pragmas(dbapi_connection, connection_record) -> None:
    del connection_record
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def init_db() -> None:
    global _engine, SessionLocal
    settings = get_settings()
    url = settings.database_url
    if url.startswith("sqlite"):
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False}
    else:
        connect_args = {}
    _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    if url.startswith("sqlite"):
        event.listen(_engine, "connect", _sqlite_pragmas)
    SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    seed()


def session_scope():
    if SessionLocal is None:
        init_db()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def seed() -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        if settings.deploy_mode == "personal":
            user = db.query(User).filter(User.email == "diego@localhost").one_or_none()
            if user is None:
                db.add(User(email="diego@localhost", name="Diego", password_hash=None, is_admin=False))
                db.commit()
            return
        if db.query(User).count() > 0:
            return
        email = settings.product_admin_email.strip()
        password = settings.product_admin_password
        if not email or len(password) < 8:
            raise RuntimeError(
                "No modo produto, defina PRODUCT_ADMIN_EMAIL e PRODUCT_ADMIN_PASSWORD (mínimo 8 caracteres)."
            )
        from app.auth import hash_password

        db.add(
            User(
                email=email.lower(),
                name="Admin",
                password_hash=hash_password(password),
                is_admin=True,
            )
        )
        db.commit()
    finally:
        db.close()


def get_setting(db, user_id: int, key: str) -> str | None:
    row = db.query(Setting).filter(Setting.user_id == user_id, Setting.key == key).one_or_none()
    if row is None or row.value == "":
        return None
    return row.value


def set_setting(db, user_id: int, key: str, value: str) -> None:
    row = db.query(Setting).filter(Setting.user_id == user_id, Setting.key == key).one_or_none()
    if row is None:
        db.add(Setting(user_id=user_id, key=key, value=value))
    else:
        row.value = value


def today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def usage_today(db, user_id: int) -> int:
    row = (
        db.query(UsageDay)
        .filter(UsageDay.user_id == user_id, UsageDay.day == today_key())
        .one_or_none()
    )
    return 0 if row is None else row.message_count


def increment_usage(db, user_id: int) -> int:
    day = today_key()
    row = db.query(UsageDay).filter(UsageDay.user_id == user_id, UsageDay.day == day).one_or_none()
    if row is None:
        row = UsageDay(user_id=user_id, day=day, message_count=1)
        db.add(row)
        db.flush()
        return 1
    row.message_count += 1
    db.flush()
    return row.message_count


def purge_expired_sessions(db) -> None:
    db.query(SessionToken).filter(SessionToken.expires_at < datetime.utcnow()).delete()


def session_ttl() -> datetime:
    return datetime.utcnow() + timedelta(days=30)


def ensure_engine():
    if _engine is None:
        init_db()
    return _engine


def ping_database() -> str:
    ensure_engine()
    with _engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return database_kind_safe()


def database_kind_safe() -> str:
    from app.policy import database_kind

    return database_kind(get_settings().database_url)
