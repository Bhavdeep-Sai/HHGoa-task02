from backend.app.api.endpoints import router as api_router
from backend.app.api.websocket import router as ws_router

__all__ = ["api_router", "ws_router"]
