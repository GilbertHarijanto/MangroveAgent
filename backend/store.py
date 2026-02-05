import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


def get_db_path() -> str:
    override = os.getenv("MANGROVE_DB_PATH")
    if override:
        return override
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "chat_history.sqlite")


def connect():
    return sqlite3.connect(get_db_path(), check_same_thread=False)


def init_db() -> None:
    try:
        with connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    title TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    graph_data TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            _ensure_sessions_title_column(cur)
            _ensure_graph_data_column(cur)
            conn.commit()
    except sqlite3.DatabaseError:
        db_path = get_db_path()
        if os.path.exists(db_path):
            os.rename(db_path, f"{db_path}.corrupt")
        with connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    title TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    graph_data TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            _ensure_sessions_title_column(cur)
            _ensure_graph_data_column(cur)
            conn.commit()


def _ensure_sessions_title_column(cur: sqlite3.Cursor) -> None:
    cur.execute("PRAGMA table_info(sessions)")
    columns = {row[1] for row in cur.fetchall()}
    if "title" not in columns:
        cur.execute("ALTER TABLE sessions ADD COLUMN title TEXT")


def _ensure_graph_data_column(cur: sqlite3.Cursor) -> None:
    cur.execute("PRAGMA table_info(messages)")
    columns = {row[1] for row in cur.fetchall()}
    if "graph_data" not in columns:
        cur.execute("ALTER TABLE messages ADD COLUMN graph_data TEXT")


def create_session(session_id: str | None = None) -> str:
    session_id = session_id or str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, created_at),
        )
        conn.commit()
    return session_id


def add_message(session_id: str, role: str, content: str, graph_data: dict | None = None) -> None:
    created_at = datetime.utcnow().isoformat()
    graph_data_json = json.dumps(graph_data) if graph_data else None
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, graph_data, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, graph_data_json, created_at),
        )
        if role == "user":
            conn.execute(
                """
                UPDATE sessions
                SET title = COALESCE(title, ?)
                WHERE id = ?
                """,
                (content[:80], session_id),
            )
        conn.commit()


def get_messages(session_id: str) -> list[dict]:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT role, content, graph_data, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        )
        rows = cur.fetchall()
    return [
        {
            "role": role,
            "content": content,
            "graph_data": json.loads(graph_data) if graph_data else None,
            "created_at": created_at,
        }
        for role, content, graph_data, created_at in rows
    ]


def get_sessions() -> list[dict]:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, created_at, title
            FROM sessions
            ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()
    return [
        {"session_id": session_id, "created_at": created_at, "title": title}
        for session_id, created_at, title in rows
    ]

def delete_session(session_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()


def rename_session(session_id: str, title: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title, session_id),
        )
        conn.commit()


def get_latest_graph_data(session_id: str) -> dict | None:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT graph_data
            FROM messages
            WHERE session_id = ? AND graph_data IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (session_id,),
        )
        row = cur.fetchone()
    if not row or not row[0]:
        return None
    return json.loads(row[0])
