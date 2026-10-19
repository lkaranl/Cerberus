import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from app.core.config import settings
from app.core.logging import logger

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    camera_id TEXT NOT NULL,
    camera_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    tracking_id INTEGER NOT NULL,
    confidence REAL NOT NULL,
    crop_path TEXT NOT NULL,
    full_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_events_camera_id ON events (camera_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp DESC);
"""

class Database:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.db_path

    def _get_connection(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Inicializa as tabelas do banco de dados SQLite."""
        try:
            with self._get_connection() as conn:
                conn.executescript(CREATE_TABLE_SQL)
            logger.info(f"Banco de dados inicializado com sucesso em: {self.db_path}")
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")
            raise

    async def save_event(self, event: Dict[str, Any]) -> int:
        """Salva um novo registro de detecção de forma assíncrona."""
        def _insert():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO events (
                        event_id, camera_id, camera_name, timestamp,
                        tracking_id, confidence, crop_path, full_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["event_id"],
                        event["camera_id"],
                        event["camera_name"],
                        event.get("timestamp", datetime.now().isoformat()),
                        int(event["tracking_id"]),
                        float(event["confidence"]),
                        event["crop_path"],
                        event["full_path"]
                    )
                )
                conn.commit()
                return cursor.lastrowid

        return await asyncio.to_thread(_insert)

    async def get_recent_events(
        self,
        limit: int = 50,
        offset: int = 0,
        camera_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Busca histórico paginado de eventos em ordem cronológica decrescente."""
        def _query():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if camera_id:
                    cursor.execute(
                        """
                        SELECT id, event_id, camera_id, camera_name, timestamp,
                               tracking_id, confidence, crop_path, full_path, created_at
                        FROM events
                        WHERE camera_id = ?
                        ORDER BY id DESC
                        LIMIT ? OFFSET ?
                        """,
                        (camera_id, limit, offset)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, event_id, camera_id, camera_name, timestamp,
                               tracking_id, confidence, crop_path, full_path, created_at
                        FROM events
                        ORDER BY id DESC
                        LIMIT ? OFFSET ?
                        """,
                        (limit, offset)
                    )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        return await asyncio.to_thread(_query)

    async def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Busca um evento específico pelo seu event_id ou id numérico."""
        def _query():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if event_id.isdigit():
                    cursor.execute("SELECT * FROM events WHERE id = ?", (int(event_id),))
                else:
                    cursor.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
                row = cursor.fetchone()
                return dict(row) if row else None

        return await asyncio.to_thread(_query)

    async def get_total_events_count(self, camera_id: Optional[str] = None) -> int:
        def _count():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if camera_id:
                    cursor.execute("SELECT COUNT(*) FROM events WHERE camera_id = ?", (camera_id,))
                else:
                    cursor.execute("SELECT COUNT(*) FROM events")
                return cursor.fetchone()[0]

        return await asyncio.to_thread(_count)


db = Database()
