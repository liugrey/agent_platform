"""
storage/db.py
SQLite persistence for sessions, messages and knowledge document metadata.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


class Database:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        agent_id   TEXT DEFAULT NULL,
        created_at TEXT NOT NULL,
        title      TEXT DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS messages (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role       TEXT NOT NULL,
        content    TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE TABLE IF NOT EXISTS knowledge_documents (
        agent_id        TEXT NOT NULL,
        collection_name TEXT NOT NULL,
        document_id     TEXT NOT NULL,
        title           TEXT NOT NULL,
        source          TEXT NOT NULL,
        content         TEXT NOT NULL,
        metadata        TEXT DEFAULT '{}',
        chunk_count     INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL,
        PRIMARY KEY (agent_id, document_id)
    );
    """

    def __init__(self, db_path: str = "./data/agent.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()
        logger.info("Database ready: %s", self.db_path)

    def _init(self):
        with self._conn() as conn:
            conn.executescript(self._SCHEMA)
            self._ensure_columns(conn)
            self._ensure_indexes(conn)

    def _ensure_columns(self, conn: sqlite3.Connection):
        session_columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "agent_id" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN agent_id TEXT DEFAULT NULL")

        doc_columns = {row["name"] for row in conn.execute("PRAGMA table_info(knowledge_documents)").fetchall()}
        if doc_columns:
            additions = {
                "collection_name": "TEXT NOT NULL DEFAULT ''",
                "title": "TEXT NOT NULL DEFAULT ''",
                "source": "TEXT NOT NULL DEFAULT ''",
                "content": "TEXT NOT NULL DEFAULT ''",
                "metadata": "TEXT DEFAULT '{}'",
                "chunk_count": "INTEGER NOT NULL DEFAULT 0",
                "created_at": "TEXT NOT NULL DEFAULT ''",
                "updated_at": "TEXT NOT NULL DEFAULT ''",
            }
            for column, ddl in additions.items():
                if column not in doc_columns:
                    conn.execute(f"ALTER TABLE knowledge_documents ADD COLUMN {column} {ddl}")

    def _ensure_indexes(self, conn: sqlite3.Connection):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kd_agent ON knowledge_documents(agent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kd_source ON knowledge_documents(source)")

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_session(self, session_id: str, title: str = "", agent_id: Optional[str] = None) -> str:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, agent_id, created_at, title) VALUES (?, ?, ?, ?)",
                (session_id, agent_id, now, title),
            )
            if agent_id is not None:
                conn.execute(
                    "UPDATE sessions SET agent_id = COALESCE(agent_id, ?) WHERE id = ?",
                    (agent_id, session_id),
                )
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT s.id, s.agent_id, s.created_at, s.title,
                       (SELECT content FROM messages WHERE session_id = s.id ORDER BY id DESC LIMIT 1) AS last_msg
                FROM sessions s
                WHERE s.id = ?
                """,
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 20, agent_id: Optional[str] = None) -> list[dict]:
        where_clause = "WHERE s.agent_id = ?" if agent_id is not None else ""
        params: list[object] = [agent_id] if agent_id is not None else []
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT s.id, s.agent_id, s.created_at, s.title,
                       (SELECT content FROM messages WHERE session_id = s.id ORDER BY id DESC LIMIT 1) AS last_msg
                FROM sessions s
                {where_clause}
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def add_message(self, session_id: str, role: str, content: str) -> int:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
        return cur.lastrowid

    def get_messages(self, session_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_history_for_llm(self, session_id: str) -> list[dict]:
        return [
            {"role": message["role"], "content": message["content"]}
            for message in self.get_messages(session_id)
            if message["role"] in ("user", "assistant")
        ]

    def upsert_knowledge_documents(
        self,
        agent_id: str,
        collection_name: str,
        documents: list[dict[str, Any]],
        chunk_counts: dict[str, int],
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            for document in documents:
                document_id = str(
                    document.get("document_id") or document.get("source") or document.get("title") or ""
                ).strip()
                if not document_id:
                    continue
                source = str(document.get("source") or document_id).strip()
                title = str(document.get("title") or source).strip()
                content = str(document.get("content") or "").strip()
                metadata = document.get("metadata") or {}
                chunk_count = int(chunk_counts.get(document_id, 0))
                conn.execute(
                    """
                    INSERT INTO knowledge_documents (
                        agent_id, collection_name, document_id, title, source, content,
                        metadata, chunk_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(agent_id, document_id) DO UPDATE SET
                        collection_name = excluded.collection_name,
                        title = excluded.title,
                        source = excluded.source,
                        content = excluded.content,
                        metadata = excluded.metadata,
                        chunk_count = excluded.chunk_count,
                        updated_at = excluded.updated_at
                    """,
                    (
                        agent_id,
                        collection_name,
                        document_id,
                        title,
                        source,
                        content,
                        json.dumps(metadata, ensure_ascii=False),
                        chunk_count,
                        now,
                        now,
                    ),
                )

    def count_agent_knowledge_documents(self, agent_id: Optional[str] = None) -> int:
        with self._conn() as conn:
            if agent_id:
                row = conn.execute(
                    """
                    SELECT COUNT(1) AS total
                    FROM knowledge_documents
                    WHERE agent_id = ?
                    """,
                    (agent_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT COUNT(1) AS total
                    FROM knowledge_documents
                    """
                ).fetchone()
        return int(row["total"]) if row else 0

    def list_agent_knowledge_documents(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        with self._conn() as conn:
            if agent_id:
                rows = conn.execute(
                    """
                    SELECT agent_id, collection_name, document_id, title, source, chunk_count, created_at, updated_at
                    FROM knowledge_documents
                    WHERE agent_id = ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (agent_id, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT agent_id, collection_name, document_id, title, source, chunk_count, created_at, updated_at
                    FROM knowledge_documents
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_agent_knowledge_document(self, agent_id: str, document_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT agent_id, collection_name, document_id, title, source, content,
                       metadata, chunk_count, created_at, updated_at
                FROM knowledge_documents
                WHERE agent_id = ? AND document_id = ?
                """,
                (agent_id, document_id),
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["metadata"] = json.loads(data.get("metadata") or "{}")
        return data
