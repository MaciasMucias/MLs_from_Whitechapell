"""Reconstruct participant sessions from the games table, and account for every
row that is excluded.

The schema has **no participant or session ID** (`server/database.py`), and that
cannot be fixed retroactively — the study is winding down and production is not
to be modified. Sessions are therefore inferred from three weak signals:

    * a `scenario_order` run that increases 0 -> 1 -> 2
    * `created_at` proximity
    * a constant `gaming_habit`

Where those signals disagree the grouping is **flagged, not guessed**.
Concurrent participants are exactly the case this schema cannot resolve, and a
wrong grouping is worse than a dropped one: it silently mixes two people's play
into one "participant" and inflates N.

Every excluded row carries a recorded reason, and the reasons partition the
table — `report()` reconciles the totals so the thesis can state an honest N.

Usage:
    uv run python -m analysis.sessions
    uv run python -m analysis.sessions --db data/study/games_20260909.sqlite
    uv run python -m analysis.sessions --verbose
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DB = Path("data/games.sqlite")

# Maps served by the course design (`maps/course_participant.json`). Anything
# else in a course-era session is a pre-fix artifact.
COURSE_MAPS = ("course_1", "course_2", "course_3")
COURSE_LENGTH = 3

# Rows further apart than this never belong to the same sitting. Within-session
# gaps in the observed data are 8-180s; the smallest between-session gap is
# ~575s, so 15 minutes separates them with a wide margin without being so loose
# that two back-to-back participants merge.
DEFAULT_MAX_GAP_SECONDS = 900


# --- Exclusion reasons ------------------------------------------------------
# Row-level: the row itself is not participant play.
ADMIN_ARTIFACT = "admin_artifact"
# Session-level: the session is real play but unusable for the comparison.
PRE_COURSE_ERA = "pre_course_era"
DUPLICATE_COURSE_BUG = "duplicate_course_bug"
WHITECHAPEL_IN_COURSE = "whitechapel_in_course_session"
INCOMPLETE_COURSE = "incomplete_course"
AMBIGUOUS_ORDER = "ambiguous_scenario_order"

REASON_TEXT = {
    ADMIN_ARTIFACT: (
        "scenario_order=-1 / map_name='unknown' — admin-panel and replay-fork "
        "sessions call register_session without set_participant_meta, so "
        "save_game still fires with fallback values. Not participant play."
    ),
    PRE_COURSE_ERA: (
        "every game on 'whitechapel' — collected before the course design "
        "existed. Production now serves only course_1/2/3."
    ),
    DUPLICATE_COURSE_BUG: (
        "a map is served twice in one session — the CourseQueue duplicate bug "
        "fixed in 594fba0. The participant never saw the full course."
    ),
    WHITECHAPEL_IN_COURSE: (
        "a course-era session that was served 'whitechapel' — same pre-fix "
        "queue bug, seen from the other side."
    ),
    INCOMPLETE_COURSE: (
        f"fewer than {COURSE_LENGTH} games — abandoned part-way. Nothing is "
        "persisted for an unfinished game, so the course cannot be completed."
    ),
    AMBIGUOUS_ORDER: (
        "scenario_order is not exactly 0,1,2 — the grouping could not be "
        "resolved from the available signals. Flagged rather than guessed."
    ),
}


@dataclass(frozen=True)
class Game:
    """One row of the games table, as analysis sees it."""

    row_id: int
    game_id: str
    map_name: str
    scenario_order: int
    gaming_habit: str
    outcome: str
    turns_survived: int
    turn_limit: int
    created_at: dt.datetime

    @property
    def is_admin_artifact(self) -> bool:
        return self.scenario_order < 0 or self.map_name == "unknown"

    @property
    def jack_won(self) -> bool:
        return self.outcome == "jack"


@dataclass
class Session:
    """A reconstructed participant sitting."""

    games: list[Game] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)

    @property
    def gaming_habit(self) -> str:
        return self.games[0].gaming_habit if self.games else "unknown"

    @property
    def started_at(self) -> dt.datetime:
        return self.games[0].created_at

    @property
    def maps(self) -> list[str]:
        return [g.map_name for g in self.games]

    @property
    def is_usable(self) -> bool:
        return not self.exclusions

    @property
    def win_rate(self) -> float:
        if not self.games:
            return 0.0
        return sum(g.jack_won for g in self.games) / len(self.games)


def load_games(db_path: Path | str = DEFAULT_DB) -> list[Game]:
    """Read the games table, oldest first. Read-only; the file is a snapshot."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"no snapshot at {path}")
    # URI mode=ro so an analysis bug can never write to the snapshot.
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT id, game_id, map_name, scenario_order, gaming_habit, "
            "outcome, turns_survived, turn_limit, created_at FROM games"
        ).fetchall()

    games = [
        Game(
            row_id=r[0],
            game_id=r[1],
            map_name=r[2],
            scenario_order=r[3],
            gaming_habit=r[4],
            outcome=r[5],
            turns_survived=r[6],
            turn_limit=r[7],
            created_at=dt.datetime.fromisoformat(r[8]),
        )
        for r in rows
    ]
    games.sort(key=lambda g: g.created_at)
    return games


def reconstruct_sessions(
    games: list[Game], max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS
) -> tuple[list[Session], list[Game]]:
    """Group participant games into sessions.

    Returns (sessions, admin_artifacts). Admin rows are removed *before*
    grouping: they interleave with real play (see rows 19-20 of the 2026-07-20
    snapshot, which sit inside a run of participant games) and would otherwise
    split a genuine session in two.

    A new session starts when any grouping signal breaks: the participant
    profile changes, scenario_order fails to advance, or the gap is too long.
    """
    artifacts = [g for g in games if g.is_admin_artifact]
    play = [g for g in games if not g.is_admin_artifact]

    sessions: list[Session] = []
    current: Session | None = None

    for game in play:
        if current is not None:
            previous = current.games[-1]
            gap = (game.created_at - previous.created_at).total_seconds()
            starts_new = (
                game.gaming_habit != previous.gaming_habit
                or game.scenario_order <= previous.scenario_order
                or gap > max_gap_seconds
            )
        else:
            starts_new = True

        if starts_new:
            current = Session()
            sessions.append(current)
        current.games.append(game)

    for session in sessions:
        session.exclusions = classify(session)
    return sessions, artifacts


def classify(session: Session) -> list[str]:
    """Every reason this session cannot be used. Empty means usable."""
    reasons: list[str] = []
    maps = session.maps

    if all(m == "whitechapel" for m in maps):
        reasons.append(PRE_COURSE_ERA)
    elif "whitechapel" in maps:
        # Mixed course/whitechapel: the pre-fix queue served a non-course map.
        reasons.append(WHITECHAPEL_IN_COURSE)

    if len(maps) != len(set(maps)):
        reasons.append(DUPLICATE_COURSE_BUG)

    if len(session.games) < COURSE_LENGTH:
        reasons.append(INCOMPLETE_COURSE)

    if [g.scenario_order for g in session.games] != list(range(len(session.games))):
        reasons.append(AMBIGUOUS_ORDER)

    return reasons


def report(
    db_path: Path | str = DEFAULT_DB,
    max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS,
    verbose: bool = False,
) -> dict:
    """Print the reconciliation and return the summary numbers."""
    games = load_games(db_path)
    sessions, artifacts = reconstruct_sessions(games, max_gap_seconds)
    usable = [s for s in sessions if s.is_usable]

    print(f"\nSnapshot: {db_path}")
    if games:
        print(
            f"Collected: {games[0].created_at:%Y-%m-%d} to {games[-1].created_at:%Y-%m-%d}"
        )
    print(f"Rows: {len(games)}\n")

    print(f"{'':4}{len(artifacts):>4} rows  excluded — {ADMIN_ARTIFACT}")
    print(
        f"{'':4}{len(games) - len(artifacts):>4} rows  participant play, in "
        f"{len(sessions)} reconstructed sessions\n"
    )

    counts: Counter[str] = Counter()
    for session in sessions:
        for reason in session.exclusions:
            counts[reason] += 1

    print("Sessions by exclusion reason (a session may have several):")
    for reason, n in counts.most_common():
        print(f"  {n:>3}  {reason}")
    print(f"  {len(usable):>3}  USABLE\n")

    n_usable_games = sum(len(s.games) for s in usable)
    print(f"==> {len(usable)} usable participants, {n_usable_games} usable games\n")

    if usable:
        by_habit = Counter(s.gaming_habit for s in usable)
        print("Usable sessions by gaming_habit:")
        for habit, n in sorted(by_habit.items()):
            wins = [s.win_rate for s in usable if s.gaming_habit == habit]
            print(f"  {habit:<14} n={n}  mean win rate {sum(wins) / len(wins):.1%}")
        print()

    if verbose:
        print("Every reconstructed session:")
        for i, s in enumerate(sessions):
            tag = "USABLE" if s.is_usable else ",".join(s.exclusions)
            rows = ",".join(str(g.row_id) for g in s.games)
            print(
                f"  [{i:>2}] {s.started_at:%m-%d %H:%M}  {s.gaming_habit:<13} "
                f"rows={rows:<12} {'/'.join(s.maps):<34} {tag}"
            )
        print()
        print("Exclusion reasons in full:")
        for reason, text in REASON_TEXT.items():
            print(f"  {reason}\n      {text}")
        print()

    return {
        "rows": len(games),
        "admin_artifacts": len(artifacts),
        "sessions": len(sessions),
        "usable_sessions": len(usable),
        "usable_games": n_usable_games,
        "exclusions": dict(counts),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default=str(DEFAULT_DB), help="snapshot to analyse")
    p.add_argument("--max-gap-seconds", type=int, default=DEFAULT_MAX_GAP_SECONDS)
    p.add_argument("--verbose", action="store_true", help="list every session")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report(args.db, args.max_gap_seconds, args.verbose)
