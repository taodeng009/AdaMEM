"""Token-usage helpers for OpenAI-compatible completion responses."""

from collections import defaultdict
from typing import Any, Dict, Iterable


def _usage_value(usage: Any, *names: str) -> int:
    for name in names:
        if isinstance(usage, dict) and name in usage:
            return int(usage[name] or 0)
        value = getattr(usage, name, None)
        if value is not None:
            return int(value or 0)
    return 0


def token_call_from_response(
    response: Any,
    *,
    call_type: str,
    model: str,
    backend: str,
) -> Dict[str, Any]:
    """Extract exact server-reported usage for one completed model call."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
    reported_total = _usage_value(usage, "total_tokens")
    total_tokens = reported_total or input_tokens + output_tokens
    return {
        "call_type": call_type,
        "model": model,
        "backend": backend,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_available": usage is not None,
    }


def summarize_token_calls(calls: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate token calls while retaining action/strategy attribution."""
    calls = list(calls)
    by_call_type = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    totals = {
        "calls": len(calls),
        "usage_available_calls": 0,
        "missing_usage_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for call in calls:
        available = bool(call.get("usage_available", False))
        totals["usage_available_calls" if available else "missing_usage_calls"] += 1
        call_type = str(call.get("call_type", "unknown"))
        group = by_call_type[call_type]
        group["calls"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = int(call.get(key, 0) or 0)
            totals[key] += value
            group[key] += value
    totals["by_call_type"] = dict(by_call_type)
    return totals
