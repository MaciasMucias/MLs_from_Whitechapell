from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from agents import HeuristicCops, NoOpDirector
from engine.graph import load_map
from server.admin_routes import admin_router
from server.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.game_map = load_map(Path("maps/whitechapel.json"))
    app.state.cop_agent = HeuristicCops()
    app.state.director = NoOpDirector()
    yield


whitechapel_ui = FastAPI(lifespan=lifespan)
whitechapel_ui.include_router(router, prefix="/api")
whitechapel_ui.include_router(admin_router, prefix="/api/admin")
whitechapel_ui.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
