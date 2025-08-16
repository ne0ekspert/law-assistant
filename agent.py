from __future__ import annotations

import os
import sys
from typing import List, Dict
import argparse
import json

from law_assistant.config import get_settings
from law_assistant.embeddings import resolve_embeddings
from law_assistant.vector_store import VectorStore
from law_assistant.llm import chat, chat_with_tools
from types import SimpleNamespace
import time


# Verbose logging is toggled by --verbose at runtime
VERBOSE = False


def log(msg: str) -> None:
    if VERBOSE:
        print(f"[agent] {msg}")


def build_index_now():
    # Build the vector index using the CLI's implementation
    try:
        from law_assistant.cli import build_index as build_index_cmd
    except Exception as e:
        print(f"Failed to import index builder: {e}")
        return False

    s = get_settings()
    args = SimpleNamespace(
        source=None,
        embeddings_provider=None,
        embeddings_model=None,
        chunk_size=800,
        chunk_overlap=100,
        batch=16,
    )
    try:
        log("Index not found. Building vector index now ...")
        build_index_cmd(args)
        log("Index build complete.\n")
        return True
    except Exception as e:
        log(f"Index build failed: {e}")
        return False


def load_vector_store():
    s = get_settings()
    index_path = os.path.join(s.data_dir, "index.faiss")
    meta_path = os.path.join(s.data_dir, "meta.json")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        if not build_index_now():
            print("Could not build index automatically. Try: law-assistant index")
            sys.exit(1)
    return s, VectorStore.load(index_path=index_path, meta_path=meta_path)


def retrieve(vs: VectorStore, emb, query: str, k: int = 5) -> List[str]:
    t0 = time.perf_counter()
    q_vec = emb.embed([query])[0]
    t_emb = (time.perf_counter() - t0) * 1000
    log(f"Embeddings: encoded query in {t_emb:.1f} ms")

    t1 = time.perf_counter()
    results = vs.search(q_vec, k=k)
    t_search = (time.perf_counter() - t1) * 1000
    log(f"FAISS: searched top-{k} in {t_search:.1f} ms (index ntotal={vs.size})")

    ctxs = []
    for _, score, md in results:
        text = md.get("text", "")
        path = (md.get("metadata") or {}).get("path") if isinstance(md.get("metadata"), dict) else None
        prefix = f"[score={score:.3f}]"
        if path:
            prefix += f" [source={path}]"
        ctxs.append(f"{prefix} {text}")

    if VERBOSE:
        print("--- Retrieved contexts ---")
        for i, c in enumerate(ctxs, 1):
            # Print only the first 240 chars per context to keep logs readable
            preview = c[:240].replace("\n", " ")
            print(f"  {i}. {preview}{'…' if len(c) > 240 else ''}")
        print("-------------------------")
    return ctxs


def format_prompt(contexts: List[str], question: str) -> List[Dict]:
    system = (
        "You are a helpful Korean legal assistant. Use only the provided CONTEXT to answer. "
        "Cite sources by filename when helpful. If the answer is not in the context, say you don't know."
    )
    user_content = "CONTEXT:\n" + "\n\n".join(contexts) + "\n\nQUESTION: " + question
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def main():
    parser = argparse.ArgumentParser(description="Interactive Korean Law Agent")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose tool/LLM logs")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = bool(args.verbose)

    s, vs = load_vector_store()
    emb = resolve_embeddings(s.embeddings_provider, s.embeddings_model)

    log(f"Embeddings provider={s.embeddings_provider} model={s.embeddings_model}")
    log(f"Chat provider={s.chat_provider} model={s.chat_model}")

    print("Korean Law Agent. Type :exit to quit, :reset to clear history.")
    history: List[Dict] = []
    top_k = 5
    memory_turns = 6  # number of prior messages to keep

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.startswith(":"):
            cmd = q[1:].strip().lower()
            if cmd == "exit":
                break
            if cmd == "reset":
                history.clear()
                print("History cleared.")
                continue
            if cmd.startswith("topk"):
                try:
                    _, val = cmd.split()
                    top_k = max(1, int(val))
                    print(f"top_k set to {top_k}")
                except Exception:
                    print("Usage: :topk <int>")
                continue
            print("Commands: :exit, :reset, :topk <int>")
            continue

        # Decide tool-calling path for OpenAI/Anthropic; fallback to inline contexts otherwise
        provider = (s.chat_provider or "").lower()
        system_base = (
            "You are a helpful Korean legal assistant. Cite sources by filename when helpful."
        )

        msgs: List[Dict] = history[-(memory_turns*2):]
        answer = ""

        if provider in {"openai", "anthropic"}:
            def tool_retrieve(name: str, args: Dict) -> str:
                if name != "retrieve":
                    return json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)
                query = args.get("query") or q
                k = int(args.get("top_k") or top_k)
                log(f"ToolCall: retrieve(query={query!r}, top_k={k})")
                t0 = time.perf_counter()
                ctxs = retrieve(vs, emb, query, k=k)
                dt = (time.perf_counter() - t0) * 1000
                log(f"ToolResult: retrieved {len(ctxs)} in {dt:.1f} ms")
                items = []
                for c in ctxs:
                    items.append({"passage": c})
                return json.dumps({"contexts": items}, ensure_ascii=False)

            tools_openai = [
                {
                    "type": "function",
                    "function": {
                        "name": "retrieve",
                        "description": "Retrieve relevant law passages for the user's query.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "User's question to search for."},
                                "top_k": {"type": "integer", "description": "Number of passages to return.", "minimum": 1, "maximum": 50},
                            },
                        },
                    },
                }
            ]

            tools_anthropic = [
                {
                    "name": "retrieve",
                    "description": "Retrieve relevant law passages for the user's query.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                        },
                    },
                }
            ]

            sysmsg = {
                "role": "system",
                "content": system_base + " Use the retrieve tool when you need information from the law corpus.",
            }
            usermsg = {"role": "user", "content": q}
            msgs = [sysmsg] + msgs + [usermsg]

            t2 = time.perf_counter()
            log("LLM: calling chat API with tools…")
            answer = chat_with_tools(
                provider=s.chat_provider,
                model=s.chat_model,
                messages=msgs,
                tools=tools_openai if provider == "openai" else tools_anthropic,
                tool_handler=tool_retrieve,
            )
            t_llm = (time.perf_counter() - t2) * 1000
            log(f"LLM: response received in {t_llm:.1f} ms")
        else:
            contexts = retrieve(vs, emb, q, k=top_k)
            turn_msgs = format_prompt(contexts, q)
            msgs = msgs + turn_msgs
            t2 = time.perf_counter()
            log("LLM: calling chat API…")
            answer = chat(
                provider=s.chat_provider,
                model=s.chat_model,
                messages=msgs,
            )
            t_llm = (time.perf_counter() - t2) * 1000
            log(f"LLM: response received in {t_llm:.1f} ms")
        print(f"Agent: {answer}\n")

        # Update history with the actual conversational content (without context block)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
