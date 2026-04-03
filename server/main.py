from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from engine.graph import load_map
from server.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.game_map = load_map(Path("maps/whitechapel.json"))
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
