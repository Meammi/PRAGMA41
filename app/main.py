from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes.api import router

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"

app = FastAPI()
app.include_router(router)

app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")
