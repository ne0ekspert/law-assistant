from __future__ import annotations

import os
import requests
from typing import List, Dict, Callable, Any
import json


def chat_ollama(model: str, messages: List[Dict], host: str | None = None) -> str:
    base = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{base}/api/chat"
    resp = requests.post(url, json={"model": model, "messages": messages, "stream": False}, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def chat_openai(model: str, messages: List[Dict], api_key: str | None = None) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    res = client.chat.completions.create(model=model, messages=messages)
    return res.choices[0].message.content or ""


def chat_anthropic(model: str, messages: List[Dict], api_key: str | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    # Convert OpenAI-style messages to Anthropic format
    sys = None
    content_msgs: List[Dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            sys = m.get("content")
            continue
        if role == "user":
            content_msgs.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            content_msgs.append({"role": "assistant", "content": m.get("content", "")})

    msg = client.messages.create(
        model=model,
        system=sys,
        max_tokens=800,
        messages=content_msgs,
    )
    return "".join(part.text for part in msg.content)


def chat(provider: str, model: str, messages: List[Dict], *, api_key: str | None = None, host: str | None = None) -> str:
    p = provider.lower()
    if p == "ollama":
        return chat_ollama(model, messages, host=host)
    if p == "openai":
        return chat_openai(model, messages, api_key=api_key)
    if p == "anthropic":
        return chat_anthropic(model, messages, api_key=api_key)
    raise ValueError(f"Unknown chat provider: {provider}")


def chat_with_tools(
    provider: str,
    model: str,
    messages: List[Dict],
    *,
    tools: List[Dict],
    tool_handler: Callable[[str, Dict[str, Any]], str],
    api_key: str | None = None,
    host: str | None = None,
    max_tool_loops: int = 4,
) -> str:
    p = provider.lower()
    if p == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        history = [*messages]
        for _ in range(max_tool_loops):
            res = client.chat.completions.create(model=model, messages=history, tools=tools, tool_choice="auto")
            msg = res.choices[0].message
            tool_calls = msg.tool_calls or []
            if not tool_calls:
                return msg.content or ""
            # Append assistant message with tool_calls
            history.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })
            # Execute each tool call and append results
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {"raw": tc.function.arguments}
                result = tool_handler(name, args)
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        # Fallback: last attempt without tools
        final = client.chat.completions.create(model=model, messages=history)
        return final.choices[0].message.content or ""

    if p == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        history = [*messages]
        for _ in range(max_tool_loops):
            res = client.messages.create(model=model, max_tokens=800, messages=history, tools=tools)
            # Check for tool_use blocks
            tool_uses = [c for c in res.content if getattr(c, "type", None) == "tool_use"]
            texts = [c.text for c in res.content if getattr(c, "type", None) == "text"]
            if tool_uses:
                # Append assistant message with tool_use blocks
                history.append({
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input}
                        for tu in tool_uses
                    ],
                })
                # Execute and append tool_results as a user message
                tool_results = []
                for tu in tool_uses:
                    result = tool_handler(tu.name, tu.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result})
                history.append({"role": "user", "content": tool_results})
                continue
            # No tool use -> return text
            if texts:
                return "".join(texts)
            # Otherwise, return empty
            return ""

    # Fallback for providers without tool support
    return chat(provider, model, messages, api_key=api_key, host=host)
