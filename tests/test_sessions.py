"""Participant session reconstruction (analysis/sessions.py).

Every fixture is built here into a tmp_path sqlite file. Nothing reads
data/games.sqlite or data/replays/ — those are mutable and a test that depends
on them fails for reasons unrelated to the code.

The cases mirror the real shapes found in the 2026-07-20 snapshot: admin rows
interleaved with genuine play, the pre-fix duplicate-course bug, sessions
abandoned part-way, and two clean complete courses.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

import pytest

from analysis.sessions import (
    ADMIN_ARTIFACT,
    AMBIGUOUS_ORDER,
    DUPLICATE_COURSE_BUG,
    INCOMPLETE_COURSE,
    PRE_COURSE_ERA,
    WHITECHAPEL_IN_COURSE,
    Game,
    Session,
    classify,
    load_games,
    reconstruct_sessions,
)

BASE = dt.datetime(2026, 7, 4, 20, 0, 0, tzinfo=dt.timezone.utc)


def _mkdb(path, rows):
    """rows: (map_name, scenario_order, gaming_habit, outcome, offset_seconds)"""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE games (id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT, "
        "map_name TEXT, scenario_order INTEGER, gaming_habit TEXT, outcome TEXT, "
        "turns_survived INTEGER, turn_limit INTEGER, move_sequence TEXT, "
        "replay TEXT, created_at TEXT)"
    )
    for i, (m, order, habit, outcome, offset) in enumerate(rows):
        conn.execute(
            "INSERT INTO games (game_id, map_name, scenario_order, gaming_habit, "
            "outcome, turns_survived, turn_limit, move_sequence, replay, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                f"g{i}",
                m,
                order,
                habit,
                outcome,
                5,
                15,
                "[]",
                "{}",
                (BASE + dt.timedelta(seconds=offset)).isoformat(),
            ),
        )
    conn.commit()
    conn.close()
    return path


def _game(map_name="course_1", order=0, habit="played_many", outcome="cops", offset=0):
    return Game(
        row_id=1,
        game_id="g",
        map_name=map_name,
        scenario_order=order,
        gaming_habit=habit,
        outcome=outcome,
        turns_survived=5,
        turn_limit=15,
        created_at=BASE + dt.timedelta(seconds=offset),
    )


def _session(*specs):
    return Session(games=[_game(m, o, offset=i * 60) for i, (m, o) in enumerate(specs)])


# --- loading ----------------------------------------------------------------


def test_load_orders_oldest_first(tmp_path):
    db = _mkdb(
        tmp_path / "g.sqlite",
        [
            ("course_1", 0, "played_many", "cops", 500),
            ("course_2", 1, "played_many", "jack", 100),
        ],
    )
    games = load_games(db)
    assert [g.map_name for g in games] == ["course_2", "course_1"]


def test_load_missing_snapshot_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_games(tmp_path / "nope.sqlite")


def test_snapshot_is_opened_read_only(tmp_path):
    """An analysis bug must never be able to mutate the snapshot."""
    db = _mkdb(tmp_path / "g.sqlite", [("course_1", 0, "played_many", "cops", 0)])
    before = db.read_bytes()
    load_games(db)
    assert db.read_bytes() == before


# --- classification ---------------------------------------------------------


def test_complete_distinct_course_is_usable():
    s = _session(("course_1", 0), ("course_2", 1), ("course_3", 2))
    assert classify(s) == []
    s.exclusions = []
    assert s.is_usable


def test_repeated_map_is_the_duplicate_bug():
    s = _session(("course_1", 0), ("course_2", 1), ("course_1", 2))
    assert DUPLICATE_COURSE_BUG in classify(s)


def test_all_whitechapel_is_pre_course_era():
    s = _session(("whitechapel", 0), ("whitechapel", 1), ("whitechapel", 2))
    reasons = classify(s)
    assert PRE_COURSE_ERA in reasons
    assert WHITECHAPEL_IN_COURSE not in reasons


def test_whitechapel_mixed_into_a_course_session():
    """The real shape of rows 35-37: whitechapel/course_1/whitechapel."""
    s = _session(("whitechapel", 0), ("course_1", 1), ("whitechapel", 2))
    reasons = classify(s)
    assert WHITECHAPEL_IN_COURSE in reasons
    assert DUPLICATE_COURSE_BUG in reasons
    assert PRE_COURSE_ERA not in reasons


def test_short_session_is_incomplete():
    assert INCOMPLETE_COURSE in classify(_session(("course_1", 0), ("course_2", 1)))


def test_non_sequential_order_is_flagged_not_guessed():
    s = _session(("course_3", 0), ("course_1", 2))
    assert AMBIGUOUS_ORDER in classify(s)


# --- grouping ---------------------------------------------------------------


def test_admin_rows_are_separated_before_grouping(tmp_path):
    """Admin rows interleave with real play and must not split a session."""
    db = _mkdb(
        tmp_path / "g.sqlite",
        [
            ("course_1", 0, "played_many", "cops", 0),
            ("unknown", -1, "unknown", "jack", 30),  # interleaved admin row
            ("course_2", 1, "played_many", "jack", 60),
            ("course_3", 2, "played_many", "cops", 90),
        ],
    )
    sessions, artifacts = reconstruct_sessions(load_games(db))
    assert len(artifacts) == 1
    assert len(sessions) == 1
    assert sessions[0].is_usable
    assert [r for r in sessions[0].maps] == ["course_1", "course_2", "course_3"]


def test_a_long_gap_starts_a_new_session(tmp_path):
    db = _mkdb(
        tmp_path / "g.sqlite",
        [
            ("course_1", 0, "played_many", "cops", 0),
            ("course_2", 1, "played_many", "cops", 100_000),
        ],
    )
    sessions, _ = reconstruct_sessions(load_games(db))
    assert len(sessions) == 2


def test_changing_gaming_habit_starts_a_new_session(tmp_path):
    db = _mkdb(
        tmp_path / "g.sqlite",
        [
            ("course_1", 0, "played_many", "cops", 0),
            ("course_2", 1, "never_played", "cops", 60),
        ],
    )
    sessions, _ = reconstruct_sessions(load_games(db))
    assert len(sessions) == 2


def test_order_restarting_starts_a_new_session(tmp_path):
    """Two participants back to back: 0,1 then 0,1 again."""
    db = _mkdb(
        tmp_path / "g.sqlite",
        [
            ("course_1", 0, "played_many", "cops", 0),
            ("course_2", 1, "played_many", "cops", 60),
            ("course_3", 0, "played_many", "cops", 120),
            ("course_1", 1, "played_many", "cops", 180),
        ],
    )
    sessions, _ = reconstruct_sessions(load_games(db))
    assert len(sessions) == 2
    assert all(len(s.games) == 2 for s in sessions)


def test_every_row_is_accounted_for(tmp_path):
    """The partition must reconcile: no row silently disappears."""
    rows = [
        ("course_1", 0, "played_many", "cops", 0),
        ("unknown", -1, "unknown", "jack", 30),
        ("course_2", 1, "played_many", "jack", 60),
        ("whitechapel", 0, "played_few", "cops", 100_000),
    ]
    db = _mkdb(tmp_path / "g.sqlite", rows)
    games = load_games(db)
    sessions, artifacts = reconstruct_sessions(games)
    assert len(artifacts) + sum(len(s.games) for s in sessions) == len(rows)


def test_win_rate_counts_jack_wins():
    s = Session(
        games=[
            _game(outcome="jack", offset=0),
            _game(outcome="cops", offset=60),
            _game(outcome="cops", offset=120),
        ]
    )
    assert s.win_rate == pytest.approx(1 / 3)
