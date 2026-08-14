from backend.app.stt.base import BaseSTTProvider
from backend.app.stt.sarvam import SarvamSTTProvider
from backend.app.stt.mock_stt import MockSTTProvider
from backend.app.config import settings


def get_stt_provider() -> BaseSTTProvider:
    if settings.SARVAM_API_KEY:
        return SarvamSTTProvider()
    return MockSTTProvider()
