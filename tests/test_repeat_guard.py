"""Repeat-guard tests for degenerate model repetition."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from src import continuation, main as main_module


def _chunk(delta: dict, finish=None) -> str:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _text_chunk(text: str) -> str:
    return _chunk({"content": text})


async def _stream(chunks):
    for chunk in chunks:
        yield chunk


def _text_of(chunks) -> str:
    text = ""
    for chunk in chunks:
        if not chunk.startswith("data: "):
            continue
        payload = chunk[6:].strip()
        if payload == "[DONE]":
            continue
        data = json.loads(payload)
        for choice in data.get("choices", []):
            text += str((choice.get("delta") or {}).get("content") or "")
    return text


class RepeatGuardUnitTests(unittest.TestCase):
    def test_detects_repeated_short_phrase(self):
        guard = continuation.RepeatGuard(min_repeats=12, max_unit=40)
        hit, repeats, unit = guard.feed("现在添加。" * 20)
        self.assertTrue(hit)
        self.assertEqual(unit, "现在添加。")
        self.assertGreaterEqual(repeats, 12)

    def test_ignores_unique_text(self):
        guard = continuation.RepeatGuard(min_repeats=12, max_unit=40)
        hit, _, _ = guard.feed(
            "第一段不同内容。第二段也不同。第三段继续不同。第四段仍然不同。"
        )
        self.assertFalse(hit)

    def test_ignores_table_separator_rows(self):
        guard = continuation.RepeatGuard(min_repeats=12, max_unit=40)
        hit, _, _ = guard.feed("|---|---|\n" * 30)
        self.assertFalse(hit)


class RepeatGuardStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_is_cut_short_on_repetition(self):
        chunks = [_text_chunk("现在添加。") for _ in range(40)]
        chunks.extend([_chunk({}, finish="stop"), "data: [DONE]\n\n"])
        with patch.dict(
            os.environ,
            {
                "TRAE_REPEAT_GUARD": "1",
                "TRAE_AUTO_CONTINUE": "0",
                "TRAE_COMPLETION_CONTRACT": "0",
            },
        ):
            output = [
                chunk
                async for chunk in main_module._stream_with_continuation(
                    [], "m", {}, _stream(chunks)
                )
            ]
        text = _text_of(output)
        self.assertGreaterEqual(text.count("现在添加"), 3)
        self.assertLess(text.count("现在添加"), 20)
        self.assertEqual(
            sum(1 for chunk in output if '"finish_reason": "stop"' in chunk),
            1,
        )
        self.assertTrue(any("[DONE]" in chunk for chunk in output))

    async def test_normal_text_is_not_cut(self):
        chunks = [_text_chunk(f"第{i}段不同内容。") for i in range(30)]
        chunks.extend([_chunk({}, finish="stop"), "data: [DONE]\n\n"])
        with patch.dict(
            os.environ,
            {
                "TRAE_REPEAT_GUARD": "1",
                "TRAE_AUTO_CONTINUE": "0",
                "TRAE_COMPLETION_CONTRACT": "0",
            },
        ):
            output = [
                chunk
                async for chunk in main_module._stream_with_continuation(
                    [], "m", {}, _stream(chunks)
                )
            ]
        self.assertEqual(_text_of(output).count("段不同内容"), 30)

    async def test_disabled_guard_passes_repetition(self):
        chunks = [_text_chunk("现在添加。") for _ in range(40)]
        chunks.extend([_chunk({}, finish="stop"), "data: [DONE]\n\n"])
        with patch.dict(
            os.environ,
            {
                "TRAE_REPEAT_GUARD": "0",
                "TRAE_AUTO_CONTINUE": "0",
                "TRAE_COMPLETION_CONTRACT": "0",
            },
        ):
            output = [
                chunk
                async for chunk in main_module._stream_with_continuation(
                    [], "m", {}, _stream(chunks)
                )
            ]
        self.assertGreaterEqual(_text_of(output).count("现在添加"), 30)


if __name__ == "__main__":
    unittest.main()
