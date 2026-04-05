from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from agents import NoOpDirector, RandomCops
from engine.graph import load_map
from server.admin_routes import admin_router
from server.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.game_map = load_map(Path("maps/whitechapel.json"))
    app.state.cop_agent = RandomCops()
    app.state.director = NoOpDirector()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router, prefix="/api")
app.include_router(admin_router, prefix="/api/admin")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
