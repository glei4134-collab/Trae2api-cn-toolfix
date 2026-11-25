"""Optional truncation helpers for explicit caller-managed diagnostics.

Production Chat/Anthropic handlers do not implicitly re-issue a model turn:
the caller owns continuation so a truncated response cannot silently duplicate
billable work or replay a tool turn.  The detection and message-building
helpers remain available for tests and explicitly opted-in internal callers.
"""

from __future__ import annotations

import os
import re
from typing import Any, Mapping, Optional, Sequence

# Every alias the relay's translators and Trae itself use for "the model ran
# out of output budget".  Missing one would hand the caller a half answer.
_LENGTH_FINISH_REASONS = frozenset(
    {
        "length",
        "max_tokens",
        "max_output_tokens",
        "max_token",
        "token_limit",
        "model_context_window_exceeded",
        "context_window_exceeded",
        "context_length_exceeded",
    }
)

_DEFAULT_TEXT_THRESHOLD = 200
_DEFAULT_SIMILARITY_THRESHOLD = 0.5
_DEFAULT_MAX_CONTINUES = 3

CONTINUE_INSTRUCTION = (
    "请继续输出，从你中断的地方继续，不要重复已经写过的内容，不要重新开始。"
)
REASONING_ONLY_INSTRUCTION = (
    "Your previous response contained only thinking/reasoning with no actual "
    "output. Provide the actual response now - either text content or a tool call."
)
INTENT_CONTINUE_INSTRUCTION = (
    "你上一条消息只声明了下一步要做什么，但没有调用任何工具。"
    "请立即实际执行：直接调用相应工具完成该动作，不要再次只回复计划或说明。"
)

_DEFAULT_INTENT_MAX_CONTINUES = 1
_INTENT_ACTION_RE = re.compile(
    r"^(?:现在|接下来|下一步|下面|然后|接着|马上|立即|这就|我先|我来|先|再|随后|"
    r"同时|另外|还要|还需|最后|之后|顺手|顺便|并)"
    r"[^。！？!?\n]{0,80}?"
    r"(?:查|查看|检查|看|读|读取|写|编写|改|修改|修|修复|补|填|加上|添加|设置|"
    r"配置|调整|替换|移除|删除|运行|跑|执行|验证|测试|创建|新建|更新|生成|整理|"
    r"搜索|查找|实现|定位|分析|处理|继续|重跑|试|保存|提交|推送|部署|安装|重启|"
    r"启动|拉取|克隆|下载|上传|同步|刷新|合并|拆分|重构|优化|格式化|导入|导出|"
    r"绑定|注册|登录|连接|打开|关闭|切换|编译|打包|发布|回滚)"
)
_INTENT_BARE_ACTION_RE = re.compile(
    r"^(?:修正|修复|编写|更新|修改|运行|执行|创建|生成|添加|删除|检查|验证|测试|"
    r"继续|重新|补全|补齐|补上|填入|填写|设置|配置|调整|替换|移除|保存|提交|推送|"
    r"部署|安装|重启|启动|重跑|实现|重写|完善|整理|同步|刷新|合并|拆分|重构|优化|"
    r"格式化|导入|导出|编译|打包|发布|绑定|注册|登录|连接|打开|关闭|切换)"
    r"[^，,。！？!?\n]{0,60}$"
)
_INTENT_FINISHED_RE = re.compile(
    r"(?:已经|已|完成|搞定|结束|可以|就好了|没问题|不需要|无需|[了啦]\s*$)"
)
_INTENT_USER_RE = re.compile(r"你(?:可以|需要|应该|要|自己)")
_INTENT_EN_RE = re.compile(
    r"^(?:now|next|then|let me|i'll|i will)\b[^.!?\n]{0,120}"
    r"(?:check|read|write|edit|fix|run|test|verify|create|update|generate|"
    r"search|find|implement|continue|try|look)",
    re.IGNORECASE,
)
_INTENT_DONE_HINT_RE = re.compile(
    r"(?:已|完成|完毕|正常|成功|通过|失败|错误|报告|总结|等待|稍等|了|啦|"
    r"你(?:可以|需要|应该|要)|done|complete|completed|finished|fixed|updated|"
    r"added|changed|works|passing)",
    re.IGNORECASE,
)

# Trailing characters that prove the model stopped mid-token/structure.
_TRUNCATED_ENDINGS = (
    re.compile(r",\s*$"),
    re.compile(r"[|｜]\s*$"),
    re.compile(r"\.\.\.\s*$"),
    re.compile(r"\\\s*$"),
    re.compile(r"//\s*$"),
    re.compile(r"#\s*$"),
    re.compile(r"[-–—]\s*$"),
    re.compile(r"[:：]\s*$"),
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value


def enabled() -> bool:
    """Whether the relay may transparently continue a truncated answer."""

    # Automatic re-issue was introduced by the Zedex compatibility layer.
    # It can duplicate billable turns and causes providers that require a
    # single tool turn to reject the follow-up.  Keep the helper available for
    # explicit diagnostics, but production defaults to the provider's terminal
    # response; operators must opt in with TRAE_AUTO_CONTINUE=1.
    return _env_flag("TRAE_AUTO_CONTINUE", False)


def max_continues() -> int:
    return _env_int("TRAE_AUTO_CONTINUE_MAX", _DEFAULT_MAX_CONTINUES, 0)


def intent_enabled() -> bool:
    """Whether an intent-only ending may be continued with one nudge."""

    return _env_flag("TRAE_AUTO_CONTINUE_ON_INTENT", False)


def intent_max_continues() -> int:
    return _env_int(
        "TRAE_AUTO_CONTINUE_ON_INTENT_MAX", _DEFAULT_INTENT_MAX_CONTINUES, 0
    )


def text_threshold() -> int:
    return _env_int(
        "TRAE_TRUNCATION_TEXT_THRESHOLD", _DEFAULT_TEXT_THRESHOLD, 1
    )


def similarity_threshold() -> float:
    value = _env_float(
        "TRAE_TRUNCATION_SIMILARITY_THRESHOLD", _DEFAULT_SIMILARITY_THRESHOLD
    )
    return min(max(value, 0.0), 1.0)


def similarity(left: str, right: str) -> float:
    """Character-position similarity used to stop a repeated short answer."""

    left = left or ""
    right = right or ""
    overlap = min(len(left), len(right))
    if overlap <= 0:
        return 0.0
    same = sum(1 for index in range(overlap) if left[index] == right[index])
    return same / overlap


def _unclosed_code_fence(text: str) -> bool:
    return text.count("```") % 2 == 1


def _unbalanced_brackets(tail: str) -> bool:
    """Whether a code-shaped tail is missing closing brackets.

    A plain count difference needs a small tolerance (prose and tables contain
    stray brackets), but a tail that *ends* inside an open bracket group is
    unambiguous: nothing after it could close the group.
    """

    opens = sum(tail.count(char) for char in "[{(")
    closes = sum(tail.count(char) for char in "]})")
    if opens > closes + 2:
        return True
    # Walk the tail and require every closer to match the innermost opener, so
    # ``{a: [1, 2}`` is not mistaken for balanced output.
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for char in tail:
        if char in "[{(":
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                # Mismatched closer: the structure is already broken, which is
                # exactly the mid-write state this detector looks for.
                return True
            stack.pop()
    return bool(stack)


_SENTENCE_END_RE = re.compile(r"[.!?。！？\]\)\}`\"'\u201d\u2019]\s*$")
_FRAGMENT_TEXT_LIMIT = 40
_FRAGMENT_REASONING_MIN = 200
_FRAGMENT_REASONING_RATIO = 4


def _stopped_after_thinking(text: str, reasoning: str) -> bool:
    """A long reasoning trace followed by a sentence fragment.

    The model can burn its whole turn inside the thinking channel and then
    emit only one or two tokens of the answer.  Treating that fragment as the
    final answer hides the fact that nothing was actually written, so it is
    continued once (the caller's similarity guard stops a repeat loop).
    """

    visible = (text or "").strip()
    thinking = (reasoning or "").strip()
    if not visible or len(visible) > _FRAGMENT_TEXT_LIMIT:
        return False
    if len(thinking) < _FRAGMENT_REASONING_MIN:
        return False
    if len(thinking) < len(visible) * _FRAGMENT_REASONING_RATIO:
        return False
    return not _SENTENCE_END_RE.search(visible)


def looks_truncated(
    text: str,
    finish_reason: Optional[str] = None,
    *,
    has_tool_calls: bool = False,
    reasoning: str = "",
) -> bool:
    """Return whether a completed upstream turn looks cut off mid-answer.

    Tool-call turns are never continued: the caller owns execution and must
    receive the call as-is.
    """

    if has_tool_calls:
        return False
    reason = str(finish_reason or "").strip().lower()
    if reason in _LENGTH_FINISH_REASONS:
        return True
    body = text or ""
    if not body.strip():
        # Nothing public was produced.  Reasoning-only output is the one case
        # worth re-asking; a totally empty turn is handled by the existing
        # empty-response retry/fallback path instead.
        return bool((reasoning or "").strip())
    if _unclosed_code_fence(body):
        return True
    if _unbalanced_brackets(body[-100:]):
        return True
    tail = body[-100:].rstrip()
    for pattern in _TRUNCATED_ENDINGS:
        if pattern.search(tail):
            return True
    return _stopped_after_thinking(body, reasoning)


def looks_intent_only(
    text: str,
    finish_reason: Optional[str] = None,
    *,
    has_tool_calls: bool = False,
    agentic: bool = False,
) -> bool:
    """Whether a finished turn only announced the next action.

    Tool turns are never continued.  Length stops belong to the truncation
    helper and must not be double-counted here.
    """

    if has_tool_calls:
        return False
    reason = str(finish_reason or "").strip().lower()
    if reason in _LENGTH_FINISH_REASONS:
        return False
    if reason in {"tool_calls", "tool_use", "function_call"}:
        return False
    body = str(text or "").strip()
    if not body or len(body) > 500:
        return False
    sentences = [
        part.strip() for part in re.split(r"[\n。！？!?]+", body) if part.strip()
    ]
    last = sentences[-1] if sentences else body
    if not last:
        return False
    if _INTENT_FINISHED_RE.search(last) or _INTENT_USER_RE.search(last):
        return False
    if (
        _INTENT_ACTION_RE.search(last)
        or _INTENT_BARE_ACTION_RE.search(last)
        or _INTENT_EN_RE.search(last)
    ):
        return True
    last_clause = re.split(r"[，,；;]", last)[-1].strip()
    if last_clause and (
        _INTENT_ACTION_RE.search(last_clause)
        or _INTENT_BARE_ACTION_RE.search(last_clause)
    ):
        return True
    # Fallback for phrasings the verb list cannot cover: a short ending in a
    # conversation that already used tools, without any completion marker.
    return bool(
        agentic
        and len(body) <= 160
        and not _INTENT_DONE_HINT_RE.search(last)
        and not re.match(r"^(?:请|建议|需要|如果|若)", body)
    )


def looks_reasoning_only(
    text: str, *, has_tool_calls: bool = False, reasoning: str = ""
) -> bool:
    if has_tool_calls:
        return False
    return not (text or "").strip() and bool((reasoning or "").strip())


def continued_messages(
    messages: Sequence[Mapping[str, Any]],
    partial_text: str,
    *,
    reasoning_only: bool = False,
    intent_only: bool = False,
) -> list[dict[str, Any]]:
    """Return the conversation for one continuation round.

    The partial answer is replayed as an assistant turn so the upstream keeps
    its own voice, followed by a user turn that only asks for the remainder.
    """

    continued: list[dict[str, Any]] = [
        dict(message) for message in messages if isinstance(message, Mapping)
    ]
    partial = (partial_text or "").strip()
    if partial:
        continued.append({"role": "assistant", "content": partial_text})
    if intent_only:
        instruction = INTENT_CONTINUE_INSTRUCTION
    elif reasoning_only:
        instruction = REASONING_ONLY_INSTRUCTION
    else:
        instruction = CONTINUE_INSTRUCTION
    continued.append({"role": "user", "content": instruction})
    return continued
