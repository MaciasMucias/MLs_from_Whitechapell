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
    gaming_habit: str       # 'never' | 'sometimes' | 'regularly' | 'unknown'
    outcome: str            # winner string from engine
    turns_survived: int
    turn_limit: int
    move_sequence: list


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id        TEXT    NOT NULL,
                map_name       TEXT    NOT NULL,
                gaming_habit   TEXT    NOT NULL,
                outcome        TEXT    NOT NULL,
                turns_survived INTEGER NOT NULL,
                turn_limit     INTEGER NOT NULL,
                move_sequence  TEXT    NOT NULL,
                created_at     TEXT    NOT NULL
            )
        """)


def save_game(record: ParticipantGame, path: Path = DB_PATH) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO games "
            "(game_id, map_name, gaming_habit, outcome, turns_survived, turn_limit, move_sequence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.game_id,
                record.map_name,
                record.gaming_habit,
                record.outcome,
                record.turns_survived,
                record.turn_limit,
                json.dumps(record.move_sequence),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
