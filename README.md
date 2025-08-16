# law-assistant

[한국어](README.ko.md)

---

FAISS-based Retrieval-Augmented Generation (RAG) over the Korean law corpus in `law/`.
Supports embeddings and chat with Ollama, OpenAI, and Anthropic.

## Quick start

- Choose a preset and copy to `.env`:
  - `cp .env.ollama .env` (local models)
  - `cp .env.openai .env` (OpenAI only)
  - `cp .env.anthropic .env` (Anthropic chat + local embeddings)
- Ensure Python 3.10+ is installed.
- Install deps: `pip install -e .` (or `pip install -r requirements.txt`).

## Build the index

- Default uses Ollama embeddings `nomic-embed-text`:
  - `law-assistant index --chunk-size 800 --chunk-overlap 100`
- With OpenAI embeddings:
  - `EMBEDDINGS_PROVIDER=openai EMBEDDINGS_MODEL=text-embedding-3-small law-assistant index`

## Query with RAG

- Default chat via Ollama (`CHAT_MODEL` from `.env`):
  - `law-assistant query "행정절차법의 처분 기준은?"`
- OpenAI chat:
  - `CHAT_PROVIDER=openai CHAT_MODEL=gpt-5-mini law-assistant query "부칙 적용 기준은?"`
- Anthropic chat:
  - `CHAT_PROVIDER=anthropic CHAT_MODEL=claude-3-5-sonnet-20240620 law-assistant query "형법 총칙 요건?"`

## Notes

- Embeddings providers: `ollama`, `openai`, `anthropic` (Anthropic embeddings may require access to `/v1/embeddings`).
- Ollama must have an embedding-capable model pulled, e.g., `ollama pull nomic-embed-text`.
- Index files are under `data/` (`index.faiss`, `meta.json`).

## Agent chat loop

- Build the index first (see above).
- Run: `python agent.py` (auto-builds the index if missing)
- Verbose logs: `python agent.py --verbose` (shows retrieval/LLM timing and tool calls)
- Commands: `:exit`, `:reset`, `:topk <int>`

## Tool calling

- With `CHAT_PROVIDER=openai` or `anthropic`, the agent exposes a `retrieve(query, top_k)` tool the model can call to fetch relevant passages instead of stuffing context into the prompt.
- With other providers (e.g., `ollama`), the agent falls back to injecting retrieved context into the user message.
- Use `--verbose` to see per-step timing and tool calls.

## Disclaimer

- This project and any agent/script/model responses do not constitute legal advice and must not be relied upon for legal decisions.
- Laws, precedents, and administrative rules may change; the accuracy, completeness, and currency of the repository data (including the `law/` submodule) and generated indices are not guaranteed.
- LLMs can hallucinate. Always verify with original texts and official sources.
- No liability is assumed for any damages arising from use; all responsibility lies with the user.
- Consult a qualified attorney for real legal matters.
