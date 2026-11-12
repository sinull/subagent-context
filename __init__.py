"""subagent-context — make `delegate_task` children inherit the parent agent's
conversation context by default.

Two rendering modes, selected by ``SUBAGENT_CONTEXT_MODE``:

``digest`` (default)
    A bounded digest: last N user/assistant messages, truncated to a char
    budget. Cheap, predictable, ignores tool noise.

``full``
    Everything: every message of the parent conversation (user, assistant and
    tool results), untruncated, no tail limit — optionally clamped by
    ``SUBAGENT_CONTEXT_MAX_CHARS`` (default 200,000; ``0`` = truly unlimited).
    The clamp exists because the child's first message is kept verbatim through
    its context compression, so an oversized injection can wedge the child
    against its model's context window with no way to compact it away.

Mechanism (public, documented plugin surfaces only — no core patching):

  * ``pre_llm_call`` fires before the tool loop of each user turn with the
    parent's ``conversation_history`` (all prior turns + the current user
    message). We render it to text and remember it per ``session_id``.
  * ``pre_tool_call`` fires before each tool dispatch with ``(tool_name,
    args)``. When the tool is ``delegate_task`` we return
    ``{"action": "modify", "args": {...}}`` — the documented arg-rewrite
    return — to prepend the rendered parent context to the call's ``context``
    field (single-task and batch ``tasks`` forms).

The rendered context is placed *before* the model-written task context so all
tasks of one batch share an identical message prefix and siblings can still
read each other's prompt cache.

Configuration (environment variables, read at call time):

  SUBAGENT_CONTEXT_MODE        digest (default) | full
  SUBAGENT_CONTEXT_MAX_CHARS   full-mode safety clamp, chars (default 200000; 0 = unlimited)
  SUBAGENT_CONTEXT_BUDGET      digest-mode char budget (default 6000)
  SUBAGENT_CONTEXT_TAIL        digest-mode parent message count (default 30)
  SUBAGENT_CONTEXT_DISABLE     "1"/"true"/"yes" makes the plugin a no-op

Both hook callbacks are pure string operations (no I/O, no LLM calls) so they
finish far inside the default 30 s ``hook_callback_timeout`` — important
because a timed-out ``pre_tool_call`` fails *closed* (blocks the tool).
"""

from __future__ import annotations

import os
import threading

__all__ = ["register", "on_pre_llm_call", "on_pre_tool_call"]

INJECT_HEADER = "[parent-agent context digest]"

# digest mode
_DEFAULT_BUDGET = 6000
_DEFAULT_TAIL = 30
# full mode: ~50k tokens on a 128k-token child model, leaving room to work.
# The child's first message is protected from its own context compression, so
# this clamp is the only thing keeping an oversized injection from wedging the
# child against the model's context window.
_DEFAULT_FULL_MAX_CHARS = 200_000
_MAX_SESSIONS = 64  # bounded memory; FIFO eviction beyond this

_lock = threading.Lock()
_digests: dict[str, str] = {}  # session_id -> rendered context, built per turn


# --------------------------------------------------------------------------
# configuration helpers
# --------------------------------------------------------------------------

def _env_raw(name: str) -> str:
    return os.environ.get(name, "").strip()


def _env_int(name: str, default: int, floor: int) -> int:
    raw = _env_raw(name)
    if not raw:
        return default
    try:
        return max(floor, int(raw))
    except ValueError:
        return default


def _mode() -> str:
    value = _env_raw("SUBAGENT_CONTEXT_MODE").lower()
    return value if value in ("digest", "full") else "digest"


def _budget() -> int:  # digest mode only
    return _env_int("SUBAGENT_CONTEXT_BUDGET", _DEFAULT_BUDGET, 500)


def _tail() -> int:  # digest mode only
    return _env_int("SUBAGENT_CONTEXT_TAIL", _DEFAULT_TAIL, 1)


def _full_max_chars() -> int:
    """Full-mode clamp. ``0`` = unlimited (explicitly accepted risk)."""
    raw = _env_raw("SUBAGENT_CONTEXT_MAX_CHARS")
    if not raw:
        return _DEFAULT_FULL_MAX_CHARS
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_FULL_MAX_CHARS


def _disabled() -> bool:
    return _env_raw("SUBAGENT_CONTEXT_DISABLE").lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------
# message rendering (defensive: provider messages may be dicts or objects,
# content may be a string or a list of content blocks)
# --------------------------------------------------------------------------

def _role_of(msg) -> str:
    if isinstance(msg, dict):
        role = msg.get("role")
    else:
        role = getattr(msg, "role", None)
    return str(role) if role else "unknown"


def _content_of(msg):
    if isinstance(msg, dict):
        return msg.get("content")
    return getattr(msg, "content", None)


def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content)


def _render_digest(history, budget: int) -> str:
    """Bounded digest: user/assistant messages only, tail-capped, truncated."""
    if not history:
        return ""

    msgs = list(history)[-_tail():]
    per_msg = max(200, budget // max(1, len(msgs)))

    lines: list[str] = []
    used = 0
    for msg in msgs:
        role = _role_of(msg)
        if role not in ("user", "assistant"):
            continue
        text = _text_of(_content_of(msg)).strip()
        if not text:
            continue
        if role == "user" and INJECT_HEADER in text:
            # A digest that boomeranged back as a user message — never re-embed
            # it, or every delegation round would grow the digest.
            continue
        if len(text) > per_msg:
            text = "…(truncated) " + text[-per_msg:]

        line = f"{role}: {text}"
        if used + len(line) > budget:
            room = budget - used
            if room > 200:
                lines.append(line[: room - 1] + "…")
            break
        lines.append(line)
        used += len(line) + 1

    return "\n".join(lines)[:budget]


def _render_full(history, clamp: int) -> str:
    """Everything: every role (user / assistant / tool results), untruncated.

    When the result exceeds ``clamp``, drop from the FRONT (recency matters)
    and say how much was omitted. ``clamp == 0`` disables the clamp entirely.
    """
    if not history:
        return ""

    lines: list[str] = []
    for msg in history:
        role = _role_of(msg)
        text = _text_of(_content_of(msg)).strip()
        if not text:
            continue
        if role == "user" and INJECT_HEADER in text:
            continue  # never re-embed a boomeranged injection
        lines.append(f"{role}: {text}")

    body = "\n".join(lines)
    if clamp and len(body) > clamp:
        omitted = len(body) - clamp
        body = f"(older parent context omitted: {omitted:,} chars)\n" + body[-clamp:]
    return body


def _render(history) -> str:
    if _mode() == "full":
        return _render_full(history, _full_max_chars())
    return _render_digest(history, _budget())


# --------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------

def register(ctx):
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)


def on_pre_llm_call(**kwargs):
    """Snapshot the parent context at the start of each user turn."""
    if _disabled():
        return
    session_id = kwargs.get("session_id")
    history = kwargs.get("conversation_history")
    if not session_id or not history:
        return

    rendered = _render(history)
    with _lock:
        if session_id not in _digests and len(_digests) >= _MAX_SESSIONS:
            _digests.pop(next(iter(_digests)))  # FIFO evict oldest session
        _digests[session_id] = rendered


def _augment(context, digest: str) -> str:
    # The header doubles as an idempotency marker: on_pre_tool_call and the
    # renderers both look for it to avoid re-embedding an existing injection.
    prefix = f"{INJECT_HEADER}\n{digest}"
    if context:
        return f"{prefix}\n\n[task context from parent agent]\n{context}"
    return prefix


def on_pre_tool_call(**kwargs):
    """Rewrite delegate_task args to carry the parent context by default."""
    if _disabled():
        return
    if kwargs.get("tool_name") != "delegate_task":
        return
    args = kwargs.get("args")
    if not isinstance(args, dict):
        return

    session_id = kwargs.get("session_id")
    with _lock:
        digest = _digests.get(session_id or "", "")
    if not digest:
        return

    args = dict(args)  # never mutate the original mapping in place

    tasks = args.get("tasks")
    if isinstance(tasks, list):
        new_tasks = []
        for task in tasks:
            if (
                isinstance(task, dict)
                and INJECT_HEADER not in str(task.get("context") or "")
            ):
                task = dict(task)
                task["context"] = _augment(task.get("context"), digest)
            new_tasks.append(task)
        args["tasks"] = new_tasks
    elif INJECT_HEADER not in str(args.get("context") or ""):
        args["context"] = _augment(args.get("context"), digest)

    return {"action": "modify", "args": args}
