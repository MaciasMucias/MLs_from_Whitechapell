import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from engine.graph import load_map
from server.course_queue import CourseQueue
from server.database import init_db
from server.routes import limiter, router
from server.session import cleanup_old_sessions


async def _session_cleanup_loop():
    while True:
        await asyncio.sleep(30 * 60)
        cleanup_old_sessions(4 * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    course = json.loads(Path("maps/course_participant.json").read_text())
    app.state.course = course
    app.state.game_maps = {
        entry["name"]: load_map(Path("maps") / entry["file"]) for entry in course
    }
    app.state.course_queue = CourseQueue(map_names=[entry["name"] for entry in course])
    task = asyncio.create_task(_session_cleanup_loop())
    yield
    task.cancel()


whitechapel_ui = FastAPI(lifespan=lifespan)
whitechapel_ui.state.limiter = limiter
whitechapel_ui.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
whitechapel_ui.add_middleware(SlowAPIMiddleware)

whitechapel_ui.include_router(router, prefix="/api")

whitechapel_ui.mount(
    "/", StaticFiles(directory="frontend_participant", html=True), name="participant"
)
