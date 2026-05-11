# MLs from Whitechapel

A reinforcement learning agent trained to play the fugitive role in an asymmetric board game based on *Letters from Whitechapel*. Heuristic cops hunt Jack using a probabilistic position model; a curriculum Director adjusts difficulty to keep training productive. Human participants play via a web interface, providing performance baselines for RL evaluation.

The thesis trains Jack using PPO, evaluates win rate and survival time, and ablates the curriculum system against a baseline without it.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra server
```

## Running locally

### Participant UI

The human-play interface served to dissertation participants.

```bash
uv run uvicorn server.main:whitechapel_ui --reload
```

Open http://localhost:8000

### Debug UI

Researcher tool with an admin panel, cop PMF overlay, and replay viewer.

```bash
uv run uvicorn server.debug_main:debug_ui --reload --port 8001
```

Open http://localhost:8001

Both apps share the same SQLite database (`data/games.sqlite`) and can run simultaneously on different ports.
