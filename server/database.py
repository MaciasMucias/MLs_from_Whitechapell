from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/games.sqlite")


@dataclass
class ParticipantGame:
    game_id: str
    map_name: str
    scenario_order: int  # position in permutation cycle (0 to N-1)
    gaming_habit: str  # 'never' | 'sometimes' | 'regularly' | 'unknown'
    outcome: str  # winner string from engine
    turns_survived: int
    turn_limit: int
    move_sequence: list
    replay: dict  # full ReplayRecord serialised (asdict)


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id        TEXT    NOT NULL,
                map_name       TEXT    NOT NULL,
                scenario_order INTEGER NOT NULL,
                gaming_habit   TEXT    NOT NULL,
                outcome        TEXT    NOT NULL,
                turns_survived INTEGER NOT NULL,
                turn_limit     INTEGER NOT NULL,
                move_sequence  TEXT    NOT NULL,
                replay         TEXT    NOT NULL,
                created_at     TEXT    NOT NULL
            )
        """)


def load_replay_from_db(db_id: int, path: Path = DB_PATH) -> dict | None:
    """Return the raw replay JSON dict for the given row id, or None if not found."""
    if not path.exists():
        return None
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT replay FROM games WHERE id = ?", (db_id,)).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def save_game(record: ParticipantGame, path: Path = DB_PATH) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO games "
            "(game_id, map_name, scenario_order, gaming_habit, outcome, turns_survived, turn_limit, move_sequence, replay, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.game_id,
                record.map_name,
                record.scenario_order,
                record.gaming_habit,
                record.outcome,
                record.turns_survived,
                record.turn_limit,
                json.dumps(record.move_sequence),
                json.dumps(record.replay),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
