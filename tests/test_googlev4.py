"""Tests for the GoogleTranslator class (no network calls)."""

import json

import pytest

from youtube_auto_dub.googlev4 import BATCH_CHUNK, GoogleTranslator


class _Resp:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class _RpcClient:
    """Fake httpx client: token page + a 200 RPC response with valid payload."""

    async def get(self, *args, **kwargs):
        return _Resp(200, "no token here")

    async def post(self, *args, **kwargs):
        parts = [["Hello world"], ["Second line"]]
        data = ["x", [[[None, None, None, None, None, parts]]]]
        inner = json.dumps(data).replace("\\", "\\\\").replace('"', '\\"')
        return _Resp(200, f'["wrb.fr","MkEWBc","{inner}",null,null,null,"generic"]')

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_source_equals_target_skips_translation():
    """When source == target, translate() should return the text unchanged."""
    translator = GoogleTranslator()
    text = "This is a test sentence."
    result = await translator.translate(text, source="en", target="en")
    assert result == text
    await translator.close()


@pytest.mark.asyncio
async def test_empty_text_returns_empty():
    translator = GoogleTranslator()
    assert await translator.translate("", source="en", target="vi") == ""
    await translator.close()


@pytest.mark.asyncio
async def test_batch_source_equals_target():
    translator = GoogleTranslator()
    texts = ["Hello", "World", "Test"]
    result = await translator.translate_batch(texts, source="en", target="en")
    assert result == texts
    await translator.close()


@pytest.mark.asyncio
async def test_rpc_response_is_parsed_text_not_coroutine():
    """Regression: _translate_rpc must await _parse_rpc_response."""
    translator = GoogleTranslator()
    translator.client = _RpcClient()
    result = await translator._translate_rpc("hi", "en", "vi")
    assert isinstance(result, str)
    assert result == "Hello world Second line"
    await translator.close()


@pytest.mark.asyncio
async def test_translate_raises_when_all_methods_fail():
    """Returning input here would render an untranslated video as success."""
    translator = GoogleTranslator()

    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    translator._translate_rpc = boom
    translator._translate_scrape = boom
    with pytest.raises(RuntimeError, match="Translation failed"):
        await translator.translate("hello", source="en", target="vi")
    await translator.close()


@pytest.mark.asyncio
async def test_batch_is_chunked_for_get_fallback():
    """Whole-video batches exceed URL limits (HTTP 400) on the scrape path."""
    translator = GoogleTranslator()
    sizes = []

    async def fake(text, source="auto", target="vi"):
        n = text.count("|||") + 1
        sizes.append(n)
        return "\n\n|||\n\n".join(["x"] * n)

    translator.translate = fake
    texts = [f"seg {i}" for i in range(BATCH_CHUNK * 2 + 5)]
    out = await translator.translate_batch(texts, target="vi")
    assert out == ["x"] * len(texts)
    assert max(sizes) <= BATCH_CHUNK
    await translator.close()
