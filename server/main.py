import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from engine.graph import load_map
from server.admin_routes import admin_router
from server.database import init_db
from server.replay_routes import replay_router
from server.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    course = json.loads(Path("maps/course.json").read_text())
    app.state.course = course
    app.state.game_maps = {
        entry["name"]: load_map(Path("maps") / entry["file"]) for entry in course
    }
    yield


whitechapel_ui = FastAPI(lifespan=lifespan)
whitechapel_ui.include_router(router, prefix="/api")
whitechapel_ui.include_router(admin_router, prefix="/api/admin")
whitechapel_ui.include_router(replay_router, prefix="/api/replays")

whitechapel_ui.mount(
    "/", StaticFiles(directory="frontend_participant", html=True), name="participant"
)
