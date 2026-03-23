from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from app.models import IndexedDocument


class StateStore:
    def __init__(self, data_dir: Path) -> None:
        self.db_path = data_dir / "state.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    language TEXT,
                    duration_seconds REAL,
                    tags TEXT,
                    speakers TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)"
            )

    def upsert(self, doc: IndexedDocument) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT INTO documents (doc_id, filename, path, indexed_at, language, duration_seconds, tags, speakers, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename = excluded.filename,
                    path = excluded.path,
                    indexed_at = excluded.indexed_at,
                    language = excluded.language,
                    duration_seconds = excluded.duration_seconds,
                    tags = excluded.tags,
                    speakers = excluded.speakers,
                    status = excluded.status,
                    error = excluded.error
                """,
                (
                    doc.doc_id,
                    doc.filename,
                    doc.path,
                    doc.indexed_at.isoformat() if isinstance(doc.indexed_at, datetime) else str(doc.indexed_at),
                    doc.language,
                    doc.duration_seconds,
                    ",".join(doc.tags),
                    ",".join(doc.speakers),
                    doc.status,
                    doc.error,
                ),
            )

    def seen_success(self, path: Path) -> bool:
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT 1 FROM documents WHERE path = ? AND status = 'indexed' LIMIT 1",
                (str(path),),
            ).fetchone()
            return row is not None

    def recent(self, limit: int = 20) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT doc_id, filename, indexed_at, duration_seconds, status FROM documents ORDER BY indexed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "doc_id": r[0],
                "filename": r[1],
                "indexed_at": r[2],
                "duration_seconds": r[3] or 0.0,
                "status": r[4],
            }
            for r in rows
        ]

    def delete_by_path(self, path: Path) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM documents WHERE path = ?", (str(path),))
