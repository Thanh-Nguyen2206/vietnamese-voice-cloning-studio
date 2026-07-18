import asyncio

import pytest

import engines


def test_missing_optional_dependency_is_friendly():
    error = engines.friendly_engine_error("XTTS", ModuleNotFoundError("No module named TTS", name="TTS"))
    assert "TTS" in str(error) and "requirements.txt" in str(error)


def test_edge_timeout_is_graceful(monkeypatch):
    class Communicate:
        def __init__(self, *args, **kwargs):
            pass

        async def save(self, path):
            raise asyncio.TimeoutError

    monkeypatch.setattr("edge_tts.Communicate", Communicate)
    with pytest.raises(RuntimeError, match="timed out|không phản hồi"):
        engines.edge_infer(None, "", "xin chào", 1.0, 42)
