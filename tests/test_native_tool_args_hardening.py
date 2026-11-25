"""Regression coverage for native tool argument sanitizing and truncation guards."""

from __future__ import annotations

import json
import unittest

from src import sse


def _call(name: str, arguments: str, call_id: str = "call_test") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class NativeToolArgumentHardeningTests(unittest.TestCase):
    def test_raw_control_character_is_reencoded(self):
        calls = sse._ensure_native_tool_arguments(
            [_call("read_file", '{"path":"Get-ChildItem\nsrc"}')],
            strict=True,
            finish_reason="tool_calls",
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"])["path"],
            "Get-ChildItem\nsrc",
        )

    def test_single_backslash_windows_path_is_escaped(self):
        calls = sse._ensure_native_tool_arguments(
            [_call("read_file", '{"path":"D:\\src\\lvjiang\\core"}')],
            strict=True,
            finish_reason="tool_calls",
        )
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"])["path"],
            "D:\\src\\lvjiang\\core",
        )

    def test_unescaped_inner_quotes_are_escaped(self):
        calls = sse._ensure_native_tool_arguments(
            [_call("read_file", '{"path":"Select-String -Pattern "batch" src"}')],
            strict=True,
            finish_reason="tool_calls",
        )
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"])["path"],
            'Select-String -Pattern "batch" src',
        )

    def test_truncated_mutating_call_is_rejected_on_length(self):
        with self.assertRaises(sse.InvalidNativeToolArguments):
            sse._ensure_native_tool_arguments(
                [_call("pwsh", '{"command":"Remove-Item -Recurse -Force D:\\tmp\\x')],
                strict=True,
                finish_reason="length",
            )

    def test_truncated_mutating_call_is_repaired_on_normal_finish(self):
        calls = sse._ensure_native_tool_arguments(
            [_call("pwsh", '{"command":"Remove-Item -Recurse -Force D:\\tmp\\x')],
            strict=True,
            finish_reason="tool_calls",
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("Remove-Item", json.loads(calls[0]["function"]["arguments"])["command"])

    def test_orphan_grep_suffix_is_rebuilt_from_schema(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "grep",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                        },
                        "required": ["pattern"],
                    },
                },
            }
        ]
        calls = sse._ensure_native_tool_arguments(
            [
                _call(
                    "grep",
                    'currency|队列|queue|排队|等待|超时|timeout", '
                    '"path": "D:\\\\量化交易\\\\豆包\\\\repos\\\\x.md"}',
                )
            ],
            strict=True,
            finish_reason="tool_calls",
            tools=tools,
        )
        self.assertEqual(len(calls), 1)
        payload = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(payload["pattern"], "currency|队列|queue|排队|等待|超时|timeout")
        self.assertEqual(payload["path"], "D:\\量化交易\\豆包\\repos\\x.md")

    def test_unrepairable_readonly_call_is_dropped_without_failing_turn(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "grep",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            }
        ]
        calls = sse._ensure_native_tool_arguments(
            [
                _call("grep", '{"pattern":"ok"}', "call_ok"),
                _call("grep", ";garbage;", "call_bad"),
            ],
            strict=True,
            finish_reason="tool_calls",
            tools=tools,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_ok")

    def test_unrepairable_mutating_call_is_dropped_on_normal_finish(self):
        calls = sse._ensure_native_tool_arguments(
            [
                _call("pwsh", '{"command":"echo ok"}', "call_ok"),
                _call("pwsh", ";garbage;", "call_bad"),
            ],
            strict=True,
            finish_reason="tool_calls",
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_ok")

    def test_all_unrepairable_readonly_calls_still_raise(self):
        with self.assertRaises(sse.InvalidNativeToolArguments):
            sse._ensure_native_tool_arguments(
                [_call("grep", ";garbage;")],
                strict=True,
                finish_reason="tool_calls",
            )

    def test_overlapping_native_fragments_merge_once(self):
        head = '{"command": "$ErrorActionPreference=\'SilentlyContinue\'\\n$r=\''
        tail = (
            "Preference='SilentlyContinue'\\n$r='D:\\\\SteamLibrary\\\\steamapps\\\\common\\\\Game'"
            '\\nWrite-Output \\"x\\"\\nGet-ChildItem $r'
        )
        merged = sse._merge_split_argument_fragments(
            [_call("pwsh", head, "a"), _call("pwsh", tail, "b")]
        )
        self.assertEqual(len(merged), 1)
        command = json.loads(merged[0]["function"]["arguments"])["command"]
        self.assertTrue(command.startswith("$ErrorActionPreference='SilentlyContinue'"))
        self.assertEqual(command.count("$ErrorActionPreference"), 1)


if __name__ == "__main__":
    unittest.main()
