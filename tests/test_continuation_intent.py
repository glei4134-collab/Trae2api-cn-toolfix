"""Intent-only ending continuation tests for the relay side."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from src import continuation, main as main_module


def _chat_chunk(text: str = "", finish=None) -> str:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {"content": text} if text else {},
                "finish_reason": finish,
            }
        ],
    }
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


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


class IntentDetectionTests(unittest.TestCase):
    def test_observed_endings_are_detected(self):
        cases = [
            "DOM 结构已清楚。修正测试脚本。",
            "所以只改 XAML 没用。现在查这个属性的实现。",
            "现在编写最终文档。",
            "`poster` 和 `loop_motion` 这两个模板各有一个占位符没有默认值。现在给它们补上默认值。",
            "#connLabel 这个 ID 并不存在。先查清真实 ID，再运行测试。",
            "真实端点 Range 完全正常，HEAD 返回 405 需要修。同时修 `probe_video_size` 的 duration 恒为 None 问题。",
            "Now I'll fix the test script.",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(continuation.looks_intent_only(text, "stop"))

    def test_completed_endings_are_not_detected(self):
        cases = [
            "全部完成。",
            "已补上默认值。",
            "现在可以补上了。",
            "两个模板也加了默认值。",
            "模板接口全部正常，404 兜底也正确。",
            "接下来你可以自己运行 npm test。",
            "已修复并部署到中继服务器。",
            "需要你确认模板名称。",
            "上一个任务已完成，报告如下。",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(continuation.looks_intent_only(text, "stop"))

    def test_tool_calls_and_length_are_never_intent_continued(self):
        self.assertFalse(
            continuation.looks_intent_only("修正测试脚本。", "tool_calls")
        )
        self.assertFalse(
            continuation.looks_intent_only(
                "修正测试脚本。", "stop", has_tool_calls=True
            )
        )
        self.assertFalse(
            continuation.looks_intent_only("修正测试脚本。", "length")
        )

    def test_agentic_short_fallback(self):
        self.assertTrue(
            continuation.looks_intent_only(
                "模板占位符还有一个问题", "stop", agentic=True
            )
        )
        self.assertFalse(
            continuation.looks_intent_only(
                "模板占位符还有一个问题", "stop", agentic=False
            )
        )

    def test_intent_instruction_is_used(self):
        messages = continuation.continued_messages(
            [{"role": "user", "content": "go"}],
            "现在补上默认值。",
            intent_only=True,
        )
        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(
            messages[-1]["content"], continuation.INTENT_CONTINUE_INSTRUCTION
        )


class StreamIntentContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_intent_nudge_reissues_once_with_tools(self):
        calls = []

        def fake_dispatch(messages, model, options):
            calls.append(list(messages))
            return _stream(
                [
                    _chat_chunk("已补上默认值。"),
                    _chat_chunk(finish="stop"),
                    "data: [DONE]\n\n",
                ]
            )

        first = _stream(
            [
                _chat_chunk("现在补上默认值。"),
                _chat_chunk(finish="stop"),
                "data: [DONE]\n\n",
            ]
        )
        with patch.dict(
            os.environ,
            {"TRAE_AUTO_CONTINUE_ON_INTENT": "1", "TRAE_AUTO_CONTINUE": "0"},
        ), patch.object(main_module, "_deferred_dispatch_stream", fake_dispatch):
            chunks = [
                chunk
                async for chunk in main_module._stream_with_continuation(
                    [{"role": "user", "content": "go"}],
                    "m",
                    {"tools": [{"type": "function"}]},
                    first,
                )
            ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0][-1]["content"], continuation.INTENT_CONTINUE_INSTRUCTION
        )
        text = _text_of(chunks)
        self.assertIn("现在补上默认值。", text)
        self.assertIn("已补上默认值。", text)
        self.assertEqual(
            sum(
                1
                for chunk in chunks
                if '"finish_reason": "stop"' in chunk
            ),
            1,
        )
        self.assertEqual(chunks.count("data: [DONE]\n\n"), 1)

    async def test_intent_is_ignored_without_tools(self):
        calls = []

        def fake_dispatch(messages, model, options):
            calls.append(messages)
            return _stream([_chat_chunk(finish="stop"), "data: [DONE]\n\n"])

        first = _stream(
            [
                _chat_chunk("现在补上默认值。"),
                _chat_chunk(finish="stop"),
                "data: [DONE]\n\n",
            ]
        )
        with patch.dict(
            os.environ,
            {"TRAE_AUTO_CONTINUE_ON_INTENT": "1", "TRAE_AUTO_CONTINUE": "0"},
        ), patch.object(main_module, "_deferred_dispatch_stream", fake_dispatch):
            chunks = [
                chunk
                async for chunk in main_module._stream_with_continuation(
                    [{"role": "user", "content": "go"}], "m", {}, first
                )
            ]
        self.assertEqual(calls, [])
        self.assertNotIn("已补上默认值。", _text_of(chunks))

    async def test_intent_disabled_by_default(self):
        calls = []

        def fake_dispatch(messages, model, options):
            calls.append(messages)
            return _stream([_chat_chunk(finish="stop"), "data: [DONE]\n\n"])

        first = _stream(
            [
                _chat_chunk("现在补上默认值。"),
                _chat_chunk(finish="stop"),
                "data: [DONE]\n\n",
            ]
        )
        with patch.dict(
            os.environ,
            {"TRAE_AUTO_CONTINUE_ON_INTENT": "0", "TRAE_AUTO_CONTINUE": "0"},
        ), patch.object(main_module, "_deferred_dispatch_stream", fake_dispatch):
            chunks = [
                chunk
                async for chunk in main_module._stream_with_continuation(
                    [{"role": "user", "content": "go"}],
                    "m",
                    {"tools": [{"type": "function"}]},
                    first,
                )
            ]
        self.assertEqual(calls, [])
        self.assertIn("现在补上默认值。", _text_of(chunks))


if __name__ == "__main__":
    unittest.main()
