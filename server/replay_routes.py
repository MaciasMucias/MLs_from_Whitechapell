from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from server.replay import list_replays, load_replay

replay_router = APIRouter()


@replay_router.get("")
async def get_replays():
    """Return metadata list for all saved replay slots."""
    return list_replays()


@replay_router.get("/{slot}")
async def get_replay(slot: int):
    """Return the full replay record for a given slot."""
    record = load_replay(slot)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No replay in slot {slot}")
    return asdict(record)
