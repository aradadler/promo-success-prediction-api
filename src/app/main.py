"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-style API skeleton for promo success prediction inference.",
)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Basic health endpoint placeholder."""
    return {"status": "ok"}

