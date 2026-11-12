"""subagent-context — make `delegate_task` children inherit (a bounded digest of)
the parent agent's conversation context by default.

Hermes subagents start with a completely fresh conversation ("Subagents Know
Nothing"): their only task context is the `goal`/`context` fields the parent
model fills in. This plugin closes that gap mechanically instead of hoping the
model writes good context.

Mechanism (public, documented plugin surfaces only — no core patching):

  * ``pre_llm_call`` fires before the tool loop of each user turn with the
    parent's ``conversation_history``. We render a bounded text digest of that
    history and remember it per ``session_id``.
  * ``pre_tool_call`` fires before each tool dispatch with ``(tool_name,
    args)``. When the tool is ``delegate_task`` we return
    ``{"action": "modify", "args": {...}}`` — the documented arg-rewrite
    return — to prepend the parent digest to the call's ``context`` field
    (both the single-task form and every task in the batch ``tasks`` form).

The digest is placed *before* the model-written task context so all tasks of
one batch share an identical message prefix and siblings can still read each
other's prompt cache.

Configuration (environment variables, read at call time):

  SUBAGENT_CONTEXT_BUDGET   max chars of parent digest per task (default 6000)
  SUBAGENT_CONTEXT_TAIL     max parent messages considered     (default 30)
  SUBAGENT_CONTEXT_DISABLE  "1"/"true"/"yes" makes the plugin a no-op

Both hook callbacks are pure string operations (no I/O, no LLM calls) so they
finish far inside the default 30 s ``hook_callback_timeout`` — important
because a timed-out ``pre_tool_call`` fails *closed* (blocks the tool).
"""

from __future__ import annotations

import os
import threading

__all__ = ["register", "on_pre_llm_call", "on_pre_tool_call"]

INJECT_HEADER = "[parent-agent context digest]"

_DEFAULT_BUDGET = 6000
_DEFAULT_TAIL = 30
_MAX_SESSIONS = 64  # bounded memory; FIFO eviction beyond this

_lock = threading.Lock()
_digests: dict[str, str] = {}  # session_id -> digest built at turn start


# --------------------------------------------------------------------------
# configuration helpers
# --------------------------------------------------------------------------

def _env_int(name: str, default: int, floor: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(floor, int(raw.strip()))
    except ValueError:
        return default


def _budget() -> int:
    return _env_int("SUBAGENT_CONTEXT_BUDGET", _DEFAULT_BUDGET, 500)


def _tail() -> int:
    return _env_int("SUBAGENT_CONTEXT_TAIL", _DEFAULT_TAIL, 1)


def _disabled() -> bool:
    return os.environ.get("SUBAGENT_CONTEXT_DISABLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


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


def _build_digest(history, budget: int) -> str:
    """Render the tail of the parent conversation into at most ``budget`` chars.

    Only user / assistant messages are kept — raw tool chatter and results are
    noise for a child and would burn its fresh context for nothing.
    """
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

    digest = _build_digest(history, _budget())
    with _lock:
        if session_id not in _digests and len(_digests) >= _MAX_SESSIONS:
            _digests.pop(next(iter(_digests)))  # FIFO evict oldest session
        _digests[session_id] = digest


def _augment(context, digest: str) -> str:
    # The header doubles as an idempotency marker: on_pre_tool_call and
    # _build_digest both look for it to avoid re-embedding an existing digest.
    prefix = f"{INJECT_HEADER}\n{digest}"
    if context:
        return f"{prefix}\n\n[task context from parent agent]\n{context}"
    return prefix


def on_pre_tool_call(**kwargs):
    """Rewrite delegate_task args to carry the parent digest by default."""
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
