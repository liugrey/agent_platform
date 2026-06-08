"""
storage/agent_store.py
Agent 配置的持久化 CRUD。
"""

import json
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)


ROLE_PRESETS: dict[str, dict] = {
    "assistant": {
        "label": "通用助手",
        "description": "适合日常问答、信息查询和通用任务处理",
        "system_prompt": "你是一个聪明、专业的 AI 助手，请用简洁清晰的中文回答问题。",
    },
    "customer_service": {
        "label": "客服专员",
        "description": "处理用户咨询、投诉、售后等客服场景",
        "system_prompt": (
            "你是一名专业、耐心的客服专员。请始终保持礼貌友好的态度，"
            "认真倾听用户问题，给出准确、有帮助的解答。"
            "如果遇到无法解决的问题，请告知用户转接人工客服。"
        ),
    },
    "code_reviewer": {
        "label": "代码审查员",
        "description": "审查代码质量、定位 Bug 并给出优化建议",
        "system_prompt": (
            "你是一名资深软件工程师，专注于代码审查。请从正确性、性能、"
            "可读性和安全性等维度分析代码，并给出可操作的修复建议。"
        ),
    },
    "translator": {
        "label": "翻译专家",
        "description": "专业多语言翻译，保持原文语义和风格",
        "system_prompt": (
            "你是一名专业翻译，精通中英日法等多种语言。翻译时请忠实原文语义，"
            "同时保持目标语言自然流畅。"
        ),
    },
    "data_analyst": {
        "label": "数据分析师",
        "description": "分析数据、解读图表、给出洞察和建议",
        "system_prompt": (
            "你是一名专业数据分析师。请清晰说明趋势、指标、异常点，"
            "并给出基于数据的结论和建议。"
        ),
    },
    "writer": {
        "label": "文案写手",
        "description": "撰写营销文案、文章、报告等文本内容",
        "system_prompt": (
            "你是一名经验丰富的文案写手，擅长营销文案、产品描述、报告和文章写作。"
        ),
    },
    "teacher": {
        "label": "知识导师",
        "description": "深入浅出地解释复杂概念，辅助学习",
        "system_prompt": (
            "你是一名耐心、博学的老师，请用通俗易懂的语言解释复杂概念，"
            "善用例子和循序渐进的方式帮助学习。"
        ),
    },
    "custom": {
        "label": "自定义",
        "description": "完全自定义角色，自行编写系统提示词",
        "system_prompt": "",
    },
}


def _default_collection_name(agent_id: str) -> str:
    return f"agent_{agent_id.replace('-', '')}"


def _sanitize_collection_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return value[:128].strip("_") or ""


class AgentStore:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS agents (
        id              TEXT PRIMARY KEY,
        name            TEXT NOT NULL UNIQUE,
        role_type       TEXT NOT NULL DEFAULT 'assistant',
        label           TEXT NOT NULL DEFAULT '通用助手',
        description     TEXT DEFAULT '',
        system_prompt   TEXT NOT NULL,
        model           TEXT NOT NULL,
        provider        TEXT NOT NULL,
        collection_name TEXT NOT NULL DEFAULT '',
        temperature     REAL NOT NULL DEFAULT 0.7,
        max_tokens      INTEGER NOT NULL DEFAULT 2048,
        is_active       INTEGER NOT NULL DEFAULT 1,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL,
        extra           TEXT DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_agents_role ON agents(role_type);
    CREATE INDEX IF NOT EXISTS idx_agents_active ON agents(is_active);
    """

    def __init__(self, db_path: str = "./data/agent.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self):
        with self._conn() as conn:
            conn.executescript(self._SCHEMA)
            self._ensure_columns(conn)

    def _ensure_columns(self, conn: sqlite3.Connection):
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(agents)").fetchall()
        }
        if "collection_name" not in columns:
            conn.execute("ALTER TABLE agents ADD COLUMN collection_name TEXT NOT NULL DEFAULT ''")
            conn.execute(
                "UPDATE agents SET collection_name = ? WHERE collection_name = '' OR collection_name IS NULL",
                (_default_collection_name("placeholder"),),
            )
            rows = conn.execute("SELECT id FROM agents").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE agents SET collection_name = ? WHERE id = ?",
                    (_default_collection_name(row["id"]), row["id"]),
                )

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

    def create(
        self,
        name: str,
        role_type: str,
        model: str,
        provider: str,
        system_prompt: Optional[str] = None,
        description: str = "",
        collection_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        extra: Optional[dict] = None,
    ) -> dict:
        preset = ROLE_PRESETS.get(role_type, ROLE_PRESETS["custom"])
        agent_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        final_prompt = system_prompt if system_prompt is not None else preset["system_prompt"]
        final_desc = description or preset["description"]
        label = preset["label"]
        final_collection_name = (
            _sanitize_collection_name(collection_name or "") or _default_collection_name(agent_id)
        )

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO agents (
                    id, name, role_type, label, description, system_prompt,
                    model, provider, collection_name, temperature, max_tokens,
                    is_active, created_at, updated_at, extra
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    agent_id,
                    name,
                    role_type,
                    label,
                    final_desc,
                    final_prompt,
                    model,
                    provider,
                    final_collection_name,
                    temperature,
                    max_tokens,
                    now,
                    now,
                    json.dumps(extra or {}),
                ),
            )
        logger.info("Agent created: %s (%s)", name, role_type)
        return self.get(agent_id)

    def get(self, agent_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_name(self, name: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM agents WHERE name = ?", (name,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_agents(
        self,
        role_type: Optional[str] = None,
        active_only: bool = True,
        limit: int = 50,
    ) -> list[dict]:
        conditions: list[str] = []
        params: list[object] = []
        if active_only:
            conditions.append("is_active = 1")
        if role_type:
            conditions.append("role_type = ?")
            params.append(role_type)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM agents {where_clause} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update(self, agent_id: str, **fields) -> Optional[dict]:
        allowed = {
            "name",
            "role_type",
            "description",
            "system_prompt",
            "model",
            "provider",
            "collection_name",
            "temperature",
            "max_tokens",
            "is_active",
            "extra",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get(agent_id)

        if "extra" in updates:
            updates["extra"] = json.dumps(updates["extra"])
        if "role_type" in updates:
            preset = ROLE_PRESETS.get(updates["role_type"], ROLE_PRESETS["custom"])
            updates["label"] = preset["label"]
        if "collection_name" in updates:
            updates["collection_name"] = (
                _sanitize_collection_name(updates["collection_name"]) or _default_collection_name(agent_id)
            )

        updates["updated_at"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [agent_id]

        with self._conn() as conn:
            conn.execute(f"UPDATE agents SET {set_clause} WHERE id = ?", values)
        return self.get(agent_id)

    def delete(self, agent_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE agents SET is_active = 0, updated_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), agent_id),
            )
        return cur.rowcount > 0

    def hard_delete(self, agent_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        return cur.rowcount > 0

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["extra"] = json.loads(data.get("extra") or "{}")
        data["is_active"] = bool(data["is_active"])
        if not data.get("collection_name"):
            data["collection_name"] = _default_collection_name(data["id"])
        return data

    @staticmethod
    def list_role_presets() -> list[dict]:
        return [
            {"role_type": key, "label": value["label"], "description": value["description"]}
            for key, value in ROLE_PRESETS.items()
        ]

    @staticmethod
    def get_preset_prompt(role_type: str) -> str:
        return ROLE_PRESETS.get(role_type, ROLE_PRESETS["custom"])["system_prompt"]
