"""FastAPI routers for the cybersecurity RAG assistant."""

from app.routers.chat import router as chat_router
from app.routers.admin import router as admin_router
from app.routers.tools import router as tools_router
from app.routers.triggers import router as triggers_router
from app.routers.agent_chains import router as agent_chains_router
from app.routers.audio import router as audio_router
from app.routers.admin_audio import router as admin_audio_router
from app.routers.admin_emotion import router as admin_emotion_router
from app.routers.backup import router as backup_router

__all__ = ["chat_router", "admin_router", "tools_router", "triggers_router", "agent_chains_router", "audio_router", "admin_audio_router", "admin_emotion_router", "backup_router"]
