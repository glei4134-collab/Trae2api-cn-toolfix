"""Completion-contract continuation tests for the relay side."""

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


def _tool_chunk(name: str, arguments: dict, call_id: str = "call_1") -> str:
    return _chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ]
        }
    )


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


def _has_tool_calls(chunks) -> bool:
    for chunk in chunks:
        if not chunk.startswith("data: "):
            continue
        payload = chunk[6:].strip()
        if payload == "[DONE]":
            continue
        data = json.loads(payload)
        for choice in data.get("choices", []):
            if (choice.get("delta") or {}).get("tool_calls"):
                return True
    return False


class CompletionContractTests(unittest.TestCase):
    def test_contract_is_added_only_for_agentic_tool_requests(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "pwsh", "parameters": {"type": "object"}},
            }
        ]
        messages = [
            {"role": "user", "content": "go"},
            {"role": "tool", "content": "done"},
        ]
        with patch.dict(os.environ, {"TRAE_COMPLETION_CONTRACT": "1"}):
            options = main_module._maybe_enable_completion_contract(
                messages, {"tools": tools}
            )
        self.assertTrue(options.get("_completion_contract"))
        names = [
            ((tool.get("function") or {}).get("name") or "")
            for tool in options["tools"]
        ]
        self.assertIn(continuation.COMPLETION_TOOL_NAME, names)
        self.assertIn("pwsh", names)

    def test_contract_is_skipped_without_tool_history(self):
        tools = [{"type": "function", "function": {"name": "pwsh"}}]
        with patch.dict(os.environ, {"TRAE_COMPLETION_CONTRACT": "1"}):
            options = main_module._maybe_enable_completion_contract(
                [{"role": "user", "content": "hi"}], {"tools": tools}
            )
        self.assertNotIn("_completion_contract", options)
        self.assertEqual(options["tools"], tools)

    def test_contract_is_skipped_for_forced_tool_choice(self):
        tools = [{"type": "function", "function": {"name": "pwsh"}}]
        messages = [{"role": "tool", "content": "done"}]
        for choice in (
            "none",
            "required",
            {"type": "function", "function": {"name": "pwsh"}},
        ):
            with self.subTest(choice=choice), patch.dict(
                os.environ, {"TRAE_COMPLETION_CONTRACT": "1"}
            ):
                options = main_module._maybe_enable_completion_contract(
                    messages, {"tools": tools, "tool_choice": choice}
                )
            self.assertNotIn("_completion_contract", options)

    def test_completion_instruction_is_used(self):
        messages = continuation.continued_messages(
            [{"role": "user", "content": "go"}],
            "现在补上默认值。",
            completion_contract=True,
        )
        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(
            messages[-1]["content"], continuation.COMPLETION_INSTRUCTION
        )


class StreamCompletionContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_call_is_stripped_and_finishes_stop(self):
        calls = []

        def fake_dispatch(messages, model, options):
            calls.append(messages)
            return _stream([_chunk({}, finish="stop"), "data: [DONE]\n\n"])

        first = _stream(
            [
                _tool_chunk(continuation.COMPLETION_TOOL_NAME, {"summary": "全部完成"}),
                _chunk({}, finish="tool_calls"),
                "data: [DONE]\n\n",
            ]
        )
        with patch.dict(
            os.environ,
            {"TRAE_COMPLETION_CONTRACT": "1", "TRAE_AUTO_CONTINUE": "0"},
        ), patch.object(main_module, "_deferred_dispatch_stream", fake_dispatch):
            chunks = [
                chunk
                async for chunk in main_module._stream_with_continuation(
                    [{"role": "tool", "content": "x"}],
                    "m",
                    {"tools": [{"type": "function"}], "_completion_contract": True},
                    first,
                )
            ]
        self.assertEqual(calls, [])
        self.assertFalse(_has_tool_calls(chunks))
        self.assertIn("全部完成", _text_of(chunks))
        self.assertEqual(
            sum(1 for chunk in chunks if '"finish_reason": "stop"' in chunk),
            1,
        )
        self.assertEqual(
            sum(1 for chunk in chunks if '"finish_reason": "tool_calls"' in chunk),
            0,
        )

    async def test_missing_completion_triggers_one_nudge(self):
        calls = []

        def fake_dispatch(messages, model, options):
            calls.append(list(messages))
            return _stream(
                [
                    _tool_chunk(
                        continuation.COMPLETION_TOOL_NAME, {"summary": "已补上"}
                    ),
                    _chunk({}, finish="tool_calls"),
                    "data: [DONE]\n\n",
                ]
            )

        first = _stream(
            [
                _text_chunk("现在补上默认值。"),
                _chunk({}, finish="stop"),
                "data: [DONE]\n\n",
            ]
        )
        with patch.dict(
            os.environ,
            {"TRAE_COMPLETION_CONTRACT": "1", "TRAE_AUTO_CONTINUE": "0"},
        ), patch.object(main_module, "_deferred_dispatch_stream", fake_dispatch):
            chunks = [
                chunk
                async for chunk in main_module._stream_with_continuation(
                    [{"role": "tool", "content": "x"}],
                    "m",
                    {"tools": [{"type": "function"}], "_completion_contract": True},
                    first,
                )
            ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0][-1]["content"], continuation.COMPLETION_INSTRUCTION
        )
        text = _text_of(chunks)
        self.assertIn("现在补上默认值。", text)
        self.assertIn("已补上", text)
        self.assertFalse(_has_tool_calls(chunks))

    async def test_real_tool_call_is_forwarded_without_nudge(self):
        calls = []

        def fake_dispatch(messages, model, options):
            calls.append(messages)
            return _stream([_chunk({}, finish="stop"), "data: [DONE]\n\n"])

        first = _stream(
            [
                _tool_chunk("pwsh", {"command": "echo ok"}),
                _chunk({}, finish="tool_calls"),
                "data: [DONE]\n\n",
            ]
        )
        with patch.dict(
            os.environ,
            {"TRAE_COMPLETION_CONTRACT": "1", "TRAE_AUTO_CONTINUE": "0"},
        ), patch.object(main_module, "_deferred_dispatch_stream", fake_dispatch):
            chunks = [
                chunk
                async for chunk in main_module._stream_with_continuation(
                    [{"role": "tool", "content": "x"}],
                    "m",
                    {"tools": [{"type": "function"}], "_completion_contract": True},
                    first,
                )
            ]
        self.assertEqual(calls, [])
        self.assertTrue(_has_tool_calls(chunks))
        self.assertIn("pwsh", "".join(chunks))

    async def test_no_nudge_when_contract_disabled(self):
        calls = []

        def fake_dispatch(messages, model, options):
            calls.append(messages)
            return _stream([_chunk({}, finish="stop"), "data: [DONE]\n\n"])

        first = _stream(
            [
                _text_chunk("现在补上默认值。"),
                _chunk({}, finish="stop"),
                "data: [DONE]\n\n",
            ]
        )
        with patch.dict(
            os.environ,
            {"TRAE_COMPLETION_CONTRACT": "0", "TRAE_AUTO_CONTINUE": "0"},
        ), patch.object(main_module, "_deferred_dispatch_stream", fake_dispatch):
            chunks = [
                chunk
                async for chunk in main_module._stream_with_continuation(
                    [{"role": "tool", "content": "x"}],
                    "m",
                    {"tools": [{"type": "function"}]},
                    first,
                )
            ]
        self.assertEqual(calls, [])
        self.assertIn("现在补上默认值。", _text_of(chunks))


if __name__ == "__main__":
    unittest.main()
