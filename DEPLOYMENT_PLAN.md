# Deployment Plan: Participant Frontend + Data Recording

## Goal

Get a participant-facing web UI live so dissertation participants can play as Jack against heuristic cops. Every completed game is recorded to SQLite, tagged with the participant's self-reported skill level, for later comparison against the RL agent.

The existing `frontend/` is a **researcher debugging tool** (admin panel, raw replay viewer, PMF overlay). It is **not** the participant UI. A new clean frontend must be built.

---

## Background

### What already exists and works

| Component | Location | Status |
|-----------|----------|--------|
| Game engine | `engine/` | Complete |
| Heuristic cops | `agents/heuristic_cops.py` | Complete |
| FastAPI server | `server/` | Complete |
| Game API endpoints | `server/routes.py` | Complete |
| 5-slot replay JSON storage | `server/replay.py` | Complete |
| Researcher debug frontend | `frontend/` | Complete — **keep untouched** |
| Map data | `maps/whitechapel.json` | Complete |
| SVG board | `Mapa_v5.svg` | Complete (399 KB, served at `/api/map-svg`) |

### What is missing

1. **Participant-facing frontend** (`frontend_participant/`) — skill form, tutorial, clean game UI
2. **SQLite data recording** (`server/database.py`) — stores completed games keyed by skill level
3. **Skill level plumbing** — `POST /api/game/new` must accept `skill_level`; session must carry it
4. **Server routing** — participant frontend at `/`, debug frontend moved to `/debug/`
5. **Deployment artifacts** — `Dockerfile`, `.dockerignore`, `fly.toml`

### Key architectural facts for implementation

- `server/main.py` app variable: `whitechapel_ui`
- Run command: `uvicorn server.main:whitechapel_ui --host 0.0.0.0 --port 8000`
- All paths are **relative to the working directory** (project root). Docker `WORKDIR /app` keeps this working.
- The SVG board is embedded as `<image href="/api/map-svg" ...>` inside an inline SVG element. ViewBox: `0 0 1434.5061 965.09717`.
- Board rendering uses SVG overlay groups injected on top of the base image. Node positions come from `/api/map` (jack nodes have `x`, `y`, `id`, `node_type`, `edges`; cop nodes have `x`, `y`, `id`, `edges`, `jack_neighbours`).
- Cop animation data comes from the `events` array on the `POST /api/game/{id}/jack-move` response. Each event: `{ cop, moved_to, jack_neighbours, action, arrest_target, arrest_success, search_hits }`.
- Game over is detected by `state.terminated === true` and `state.winner` (`"jack"` | `"cops"`).
- `winner` can also be `"timeout"` or `"surrounded"` — check the engine's `end_of_round()` in `engine/env.py` to confirm exact strings used.
- The replay system (`server/replay.py`) already serialises `GameRecord` via `dataclasses.asdict()`. The SQLite `move_sequence` column stores the same JSON.

---

## Implementation Steps

### Step 1 — `server/database.py` (new file)

```python
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/games.db")


@dataclass
class ParticipantGame:
    game_id: str
    skill_level: str          # 'beginner' | 'experienced' | 'advanced' | 'unknown'
    outcome: str              # winner string from engine
    turns_survived: int
    turn_limit: int
    move_sequence: list       # list of serialised RoundRecord dicts


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id        TEXT    NOT NULL,
                skill_level    TEXT    NOT NULL,
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
            "INSERT INTO games (game_id, skill_level, outcome, turns_survived, turn_limit, move_sequence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.game_id,
                record.skill_level,
                record.outcome,
                record.turns_survived,
                record.turn_limit,
                json.dumps(record.move_sequence),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
```

---

### Step 2 — `server/session.py`

Add `skill_level: str = "unknown"` to the `GameSession` dataclass:

```python
@dataclass
class GameSession:
    game_id: str
    game_map: Map
    state: GameState
    terminated: bool
    winner: str | None
    rng: random.Random
    blocking: bool = False
    turn_limit: int | None = None
    skill_level: str = "unknown"          # ← add this
    history: list = field(default_factory=list)
    round_history: list = field(default_factory=list)
```

---

### Step 3 — `server/routes.py`

Two changes:

**a) Accept `skill_level` in `POST /api/game/new`**

```python
class NewGameRequest(BaseModel):
    skill_level: str = "unknown"

@router.post("/game/new")
async def new_game(body: NewGameRequest, request: Request):
    session = new_session(request.app.state.game_map)
    session.skill_level = body.skill_level
    return state_view(session)
```

**b) Save to SQLite on game over**

Inside the `jack_move` handler, replace the existing terminal block:

```python
if terminated:
    from server.replay import build_and_save_replay
    from server.database import ParticipantGame, save_game
    import dataclasses
    build_and_save_replay(session)
    effective_limit = session.turn_limit if session.turn_limit is not None else session.game_map.turn_limit
    save_game(ParticipantGame(
        game_id=session.game_id,
        skill_level=session.skill_level,
        outcome=session.winner or "unknown",
        turns_survived=session.state.turn,
        turn_limit=effective_limit,
        move_sequence=[dataclasses.asdict(r) for r in session.round_history],
    ))
```

---

### Step 4 — `server/main.py`

Two changes:

**a) Call `init_db()` in lifespan**

```python
from server.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.game_map = load_map(Path("maps/whitechapel.json"))
    app.state.cop_agent = HeuristicCops()
    app.state.director = NoOpDirector()
    yield
```

**b) Change frontend mounts**

```python
# Participant frontend at root
whitechapel_ui.mount("/", StaticFiles(directory="frontend_participant", html=True), name="participant")
# Debug frontend at secret path (not linked from participant page)
whitechapel_ui.mount("/debug", StaticFiles(directory="frontend", html=True), name="debug")
```

> **Order matters**: the `/debug` mount must be registered before `/` in FastAPI, or move the participant mount after all API routes. Actually with StaticFiles the catch-all `/` mount must come **last** — keep it as the final `.mount()` call.

---

### Step 5 — `frontend_participant/` (new directory)

Three files: `index.html`, `game.html`, `game.js`. The `api.js` from `frontend/` can be copied verbatim with one addition (the `skill_level` argument to `newGame`).

#### `frontend_participant/api.js`

Copy `frontend/api.js` and change `newGame` to:

```js
async function newGame(skillLevel) {
  const r = await fetch("/api/game/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ skill_level: skillLevel }),
  });
  if (!r.ok) throw new Error("Failed to start game");
  return r.json();
}
```

Keep `fetchMap()` and `jackMove()` identical to `frontend/api.js`.

---

#### `frontend_participant/index.html`

Landing page with:
- Title and one-paragraph description of the task
- Skill level radio buttons (Beginner / Experienced / Advanced)
- "How to play" tutorial toggle (inline, 4 slides — see §9.3 of DESIGN_REQUIREMENTS.md)
- "Start game" button → calls `newGame(skillLevel)`, stores `game_id` + `skill_level` in `sessionStorage`, redirects to `game.html`

Tutorial slides cover:
1. Goal: reach your hideout before the turn limit runs out
2. How to move: click any highlighted (orange-ringed) node
3. What cops do: after your move, each cop moves then searches or attempts an arrest; blue-ringed nodes were confirmed visited
4. Arrest: if a cop arrests the node you're on, the cops win

No links to `/debug/` anywhere on this page.

---

#### `frontend_participant/game.html` + `game.js`

The game JS is a **stripped-down adaptation of `frontend/board.js`**. Keep:
- SVG overlay rendering (cop nodes as red squares, Jack node as green ring, hideout as yellow ring, legal moves as orange rings, visited nodes as blue rings)
- Zone layer (`renderZoneLayer`)
- Cop animation after Jack's move
- `updateStatus` / game-over detection

Remove entirely:
- `window.adminPickMode` and pick layer
- `window.adminOnStateUpdate`
- `window.renderForReplay`
- `window.setPmfData` / PMF layer
- `window.startGameFromState`
- `logEntry` event log (replace with a simpler turn counter in the sidebar)
- `mode-banner` (admin pick mode indicator)

**Game flow:**
1. On page load: read `game_id` and `skill_level` from `sessionStorage`. If missing, redirect back to `index.html`.
2. Fetch `/api/map` for node positions, then fetch `/api/game/{game_id}` for initial state, then render.
3. Jack clicks a legal move → `jackMove(gameId, destination)` → animate cop events → render new state.
4. When `state.terminated`, show end-of-game overlay:
   - Win: "You escaped!" + turns survived
   - Loss (arrested): "Caught! The police arrested you."
   - Loss (timeout): "Time's up! The police patrol the area."
   - Loss (surrounded): "Surrounded! No escape route."
   - "Play again" button → redirect to `index.html`

**Cop animation** (adapted from `frontend/board.js` `handleJackMove`):
The `events` array on the move response contains one entry per cop per sub-step. Replay them sequentially with a short `setTimeout` delay between each to show cop movement visually. After the animation completes, call `render(state)`.

**Sidebar content** (minimal):
- Turn counter: "Round X of Y"
- Hideout node number
- Visited count
- Legend (same swatches as debug frontend, minus pick-mode colour)

---

### Step 6 — Deployment artifacts

#### `Dockerfile`

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra server --no-dev

COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "server.main:whitechapel_ui", \
     "--host", "0.0.0.0", "--port", "8000"]
```

#### `.dockerignore`

```
.git
.venv
__pycache__
*.pyc
*.pyo
.pytest_cache
data/
*.map
```

> `data/` is excluded — it lives on a persistent volume at runtime. `*.map` files (`jack.map`, `cops.map`) are legacy prototyping assets not loaded by the current server.

#### `fly.toml`

```toml
app = "mls-from-whitechapel"
primary_region = "lhr"

[build]

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = "off"
  auto_start_machines = true
  min_machines_running = 1

[[vm]]
  memory = "256mb"
  cpu_kind = "shared"
  cpus = 1

[mounts]
  source = "replay_data"
  destination = "/app/data"
```

`auto_stop_machines = "off"` is required — the server holds active game sessions in memory and must not be paused mid-game.

---

## Deployment Steps (Fly.io)

Prerequisites: [Fly.io CLI](https://fly.io/docs/flyctl/install/) installed, account created (credit card required for verification but free tier will not charge).

```bash
# 1. Authenticate
fly auth login

# 2. Create the app (first time only — uses fly.toml)
fly launch --no-deploy

# 3. Create the persistent volume (1 GB is enough for SQLite + replays)
fly volumes create replay_data --region lhr --size 1

# 4. Deploy
fly deploy

# 5. Check logs
fly logs
```

The app will be live at `https://mls-from-whitechapel.fly.dev` (or whatever name Fly assigns).

**Subsequent deploys:** just `fly deploy` after pushing changes.

**Back up the database:**
```bash
fly ssh console -C "cat /app/data/games.db" > games_backup.db
```

---

## Railway (fallback)

If Fly.io free tier RAM proves insufficient (256 MB can be tight for Python 3.13 + game state):

1. Push code to GitHub
2. Create a Railway project → "Deploy from GitHub repo"
3. Add a Volume at `/app/data`
4. Set start command: `uv run uvicorn server.main:whitechapel_ui --host 0.0.0.0 --port $PORT`
5. Railway auto-detects Python and will use `pyproject.toml`

Cost: ~$5/month on the Hobby plan.

---

## Verification Checklist

After deploying (or testing locally with `uv run uvicorn server.main:whitechapel_ui --reload`):

- [ ] `/` loads the participant landing page — skill form visible, no admin panel
- [ ] Selecting a skill level and clicking Start redirects to `game.html` with the board rendered
- [ ] Jack's starting node is highlighted green, hideout is yellow, legal moves are orange
- [ ] Clicking a legal move triggers cop animation, then updates the board
- [ ] Game ends with the correct overlay message (win / arrested / timeout / surrounded)
- [ ] `data/games.db` contains a row for the completed game with correct `skill_level` and `outcome`
- [ ] `/debug/` loads the original researcher debug frontend and still works
- [ ] Restarting the server does not lose the SQLite rows (volume is working)
- [ ] Board renders acceptably on a mobile phone browser

---

## Files Changed / Created Summary

| File | Action |
|------|--------|
| `server/database.py` | Create |
| `server/session.py` | Add `skill_level` field |
| `server/routes.py` | Accept `skill_level` in new-game; call `save_game` on terminal |
| `server/main.py` | Call `init_db()`; change frontend mounts |
| `frontend_participant/index.html` | Create |
| `frontend_participant/game.html` | Create |
| `frontend_participant/game.js` | Create (adapted from `frontend/board.js`) |
| `frontend_participant/api.js` | Create (adapted from `frontend/api.js`) |
| `Dockerfile` | Create |
| `.dockerignore` | Create |
| `fly.toml` | Create |
| `frontend/` | Untouched |
