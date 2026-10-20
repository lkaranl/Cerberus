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
        """Inicializa as tabelas e aplica migrações incrementais de forma segura."""
        try:
            with self._get_connection() as conn:
                # 1. Garante que a tabela básica existe
                conn.executescript(CREATE_TABLE_SQL)

                # 2. Migra colunas adicionais com segurança
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(events)")
                existing_cols = [row["name"] for row in cursor.fetchall()]

                if "person_name" not in existing_cols:
                    conn.execute("ALTER TABLE events ADD COLUMN person_name TEXT DEFAULT 'Desconhecido'")
                if "is_known" not in existing_cols:
                    conn.execute("ALTER TABLE events ADD COLUMN is_known BOOLEAN DEFAULT 0")
                if "face_confidence" not in existing_cols:
                    conn.execute("ALTER TABLE events ADD COLUMN face_confidence REAL DEFAULT 0.0")
                if "entity_type" not in existing_cols:
                    conn.execute("ALTER TABLE events ADD COLUMN entity_type TEXT DEFAULT 'person'")
                if "license_plate" not in existing_cols:
                    conn.execute("ALTER TABLE events ADD COLUMN license_plate TEXT DEFAULT ''")

                conn.commit()

                # 3. Cria índices sobre as colunas migradas
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_person_name ON events (person_name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_is_known ON events (is_known)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_entity_type ON events (entity_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_license_plate ON events (license_plate)")
                conn.commit()

            logger.info(f"Banco de dados SQLite pronto em: {self.db_path}")
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")
            raise

    def initialize(self) -> None:
        """Alias de conveniência para init_db."""
        self.init_db()

    async def save_event(self, event: Dict[str, Any]) -> int:
        """Salva um novo registro de detecção (Pessoa, Veículo ou Pet) no SQLite."""
        def _insert():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO events (
                        event_id, camera_id, camera_name, timestamp,
                        tracking_id, confidence, crop_path, full_path,
                        person_name, is_known, face_confidence,
                        entity_type, license_plate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["event_id"],
                        event["camera_id"],
                        event["camera_name"],
                        event.get("timestamp", datetime.now().isoformat()),
                        int(event["tracking_id"]),
                        float(event["confidence"]),
                        event["crop_path"],
                        event["full_path"],
                        event.get("person_name", "Desconhecido"),
                        1 if event.get("is_known") else 0,
                        float(event.get("face_confidence", 0.0)),
                        event.get("entity_type", "person"),
                        event.get("license_plate", "")
                    )
                )
                conn.commit()
                return cursor.lastrowid

        return await asyncio.to_thread(_insert)

    async def update_event(self, event_id: str, updates: Dict[str, Any]) -> bool:
        """Atualiza campos de um evento existente (ex: upgrade de foto com rosto nítido ou identificação confirmada)."""
        def _update():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                set_clauses = []
                params = []
                for k, v in updates.items():
                    set_clauses.append(f"{k} = ?")
                    params.append(v)
                params.append(event_id)
                query = f"UPDATE events SET {', '.join(set_clauses)} WHERE event_id = ?"
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount > 0

        return await asyncio.to_thread(_update)

    async def get_recent_events(
        self,
        limit: int = 50,
        offset: int = 0,
        camera_id: Optional[str] = None,
        is_known: Optional[bool] = None,
        entity_type: Optional[str] = None,
        person_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Busca histórico paginado com suporte a filtros por tipo de entidade e identidade."""
        def _query():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                conditions = []
                params = []

                if camera_id:
                    conditions.append("camera_id = ?")
                    params.append(camera_id)
                if is_known is not None:
                    conditions.append("is_known = ?")
                    params.append(1 if is_known else 0)
                if entity_type:
                    conditions.append("entity_type = ?")
                    params.append(entity_type)
                if person_name:
                    conditions.append("person_name = ?")
                    params.append(person_name)

                where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                sql = f"""
                    SELECT id, event_id, camera_id, camera_name, timestamp,
                           tracking_id, confidence, crop_path, full_path,
                           person_name, is_known, face_confidence,
                           entity_type, license_plate, created_at
                    FROM events
                    {where_clause}
                    ORDER BY id DESC
                    LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])
                cursor.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]

        return await asyncio.to_thread(_query)

    async def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
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

    async def get_events_for_ai_query(self, hours: int = 24, limit: int = 40) -> List[Dict[str, Any]]:
        """Busca eventos recentes ordenados para alimentar o assistente de IA."""
        def _query():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                sql = """
                    SELECT camera_name, timestamp, entity_type, person_name,
                           is_known, license_plate, confidence
                    FROM events
                    ORDER BY id DESC
                    LIMIT ?
                """
                cursor.execute(sql, (limit,))
                return [dict(row) for row in cursor.fetchall()]

        return await asyncio.to_thread(_query)

    async def get_analytics_data(self) -> Dict[str, Any]:
        """Calcula estatísticas agregadas de presenças, veículos e pets para o Dashboard."""
        def _analytics():
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Totais gerais
                cursor.execute("SELECT COUNT(*) FROM events")
                total_events = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM events WHERE is_known = 1")
                total_known = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM events WHERE is_known = 0")
                total_unknown = cursor.fetchone()[0]

                # Eventos de hoje
                today_prefix = datetime.now().strftime("%Y-%m-%d")
                cursor.execute("SELECT COUNT(*) FROM events WHERE timestamp LIKE ?", (f"{today_prefix}%",))
                events_today = cursor.fetchone()[0]

                # Distribuição por Tipo de Entidade (Pessoas, Veículos, Pets)
                cursor.execute("""
                    SELECT entity_type, COUNT(*) as count
                    FROM events
                    GROUP BY entity_type
                """)
                by_entity = {row["entity_type"]: row["count"] for row in cursor.fetchall()}

                # Distribuição por Câmera
                cursor.execute("""
                    SELECT camera_id, camera_name, COUNT(*) as count
                    FROM events
                    GROUP BY camera_id, camera_name
                    ORDER BY count DESC
                """)
                by_camera = [dict(row) for row in cursor.fetchall()]

                # Distribuição por Horário do Dia (00 a 23h)
                cursor.execute("""
                    SELECT substr(timestamp, 12, 2) as hour, COUNT(*) as count
                    FROM events
                    WHERE length(timestamp) >= 13
                    GROUP BY hour
                    ORDER BY hour ASC
                """)
                by_hour = [dict(row) for row in cursor.fetchall()]

                # Top Pessoas
                cursor.execute("""
                    SELECT person_name, COUNT(*) as count, MAX(timestamp) as last_seen
                    FROM events
                    WHERE entity_type = 'person' AND is_known = 1
                    GROUP BY person_name
                    ORDER BY count DESC
                    LIMIT 10
                """)
                top_persons = [dict(row) for row in cursor.fetchall()]

                # Top Veículos & Placas
                cursor.execute("""
                    SELECT person_name, license_plate, is_known, COUNT(*) as count, MAX(timestamp) as last_seen
                    FROM events
                    WHERE entity_type = 'vehicle'
                    GROUP BY person_name, license_plate
                    ORDER BY count DESC
                    LIMIT 10
                """)
                top_vehicles = [dict(row) for row in cursor.fetchall()]

                # Top Pets
                cursor.execute("""
                    SELECT person_name, is_known, COUNT(*) as count, MAX(timestamp) as last_seen
                    FROM events
                    WHERE entity_type = 'pet'
                    GROUP BY person_name
                    ORDER BY count DESC
                    LIMIT 10
                """)
                top_pets = [dict(row) for row in cursor.fetchall()]

                return {
                    "total_events": total_events,
                    "total_known": total_known,
                    "total_unknown": total_unknown,
                    "events_today": events_today,
                    "by_entity": by_entity,
                    "by_camera": by_camera,
                    "by_hour": by_hour,
                    "top_persons": top_persons,
                    "top_vehicles": top_vehicles,
                    "top_pets": top_pets
                }

        return await asyncio.to_thread(_analytics)

    async def clear_all_events(self, delete_files: bool = True) -> int:
        """Limpa todos os eventos do banco de dados SQLite e recicla os arquivos de evidência em disco."""
        def _clear():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT crop_path, full_path FROM events")
                rows = cursor.fetchall()

                if delete_files:
                    for row in rows:
                        for path_str in (row["crop_path"], row["full_path"]):
                            if path_str:
                                try:
                                    p = Path(path_str)
                                    if p.exists() and p.is_file():
                                        p.unlink()
                                except Exception:
                                    pass

                cursor.execute("DELETE FROM events")
                deleted_count = cursor.rowcount
                conn.commit()
                conn.execute("VACUUM")
                return deleted_count

        count = await asyncio.to_thread(_clear)
        logger.info(f"🧹 Banco de dados de eventos limpo: {count} registros removidos.")
        return count


db = Database()
