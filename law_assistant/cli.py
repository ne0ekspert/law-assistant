from __future__ import annotations

import os
import math
import argparse
from typing import List, Dict
from tqdm import tqdm

from .config import get_settings
from .document_loader import load_documents
from .text_splitter import chunk_document
from .embeddings import resolve_embeddings
from .vector_store import VectorStore
from .llm import chat


def build_index(args: argparse.Namespace) -> None:
    s = get_settings()
    laws_dir = args.source or s.laws_dir

    docs = list(load_documents(laws_dir))
    chunks: List[Dict] = []
    for d in docs:
        chunks.extend(chunk_document(d, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap))

    texts = [c["text"] for c in chunks]
    emb = resolve_embeddings(args.embeddings_provider or s.embeddings_provider, args.embeddings_model or s.embeddings_model)

    # Estimate dimensionality from first embedding
    first = emb.embed([texts[0]])[0]
    vs = VectorStore.create(dim=len(first), meta_path=os.path.join(s.data_dir, "meta.json"))

    batch = args.batch
    for i in tqdm(range(0, len(texts), batch), desc="Embedding"):
        batch_texts = texts[i:i+batch]
        vectors = emb.embed(batch_texts)
        metadatas = [
            {"chunk_id": chunks[i+j]["chunk_id"], "doc_id": chunks[i+j]["doc_id"], "text": chunks[i+j]["text"], "metadata": chunks[i+j]["metadata"]}
            for j in range(len(batch_texts))
        ]
        vs.add(vectors, metadatas)

    os.makedirs(s.data_dir, exist_ok=True)
    vs.save(index_path=os.path.join(s.data_dir, "index.faiss"), meta_path=os.path.join(s.data_dir, "meta.json"))
    print(f"Indexed {vs.size} chunks from {len(docs)} documents.")


def query_index(args: argparse.Namespace) -> None:
    s = get_settings()
    index_path = os.path.join(s.data_dir, "index.faiss")
    meta_path = os.path.join(s.data_dir, "meta.json")
    vs = VectorStore.load(index_path=index_path, meta_path=meta_path)

    emb = resolve_embeddings(args.embeddings_provider or s.embeddings_provider, args.embeddings_model or s.embeddings_model)
    q_vec = emb.embed([args.query])[0]
    results = vs.search(q_vec, k=args.top_k)

    contexts = []
    for _, score, md in results:
        contexts.append(f"[score={score:.3f}] {md.get('text','')}")

    system = "You are a helpful legal assistant. Use only the provided CONTEXT to answer. If the answer is not in the context, say you don't know."
    prompt = (
        "CONTEXT:\n" + "\n\n".join(contexts) + "\n\n" +
        f"QUESTION: {args.query}\n"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    answer = chat(
        provider=args.chat_provider or s.chat_provider,
        model=args.chat_model or s.chat_model,
        messages=messages,
    )
    print(answer)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="law-assistant", description="FAISS RAG over Korean laws")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build vector index from law corpus")
    p_index.add_argument("--source", type=str, default=None, help="Path to laws directory")
    p_index.add_argument("--embeddings-provider", type=str, default=None, choices=["ollama", "openai", "anthropic"], help="Embeddings provider")
    p_index.add_argument("--embeddings-model", type=str, default=None, help="Embeddings model name")
    p_index.add_argument("--chunk-size", type=int, default=800)
    p_index.add_argument("--chunk-overlap", type=int, default=100)
    p_index.add_argument("--batch", type=int, default=16, help="Embedding batch size")
    p_index.set_defaults(func=build_index)

    p_query = sub.add_parser("query", help="Query the vector index and chat")
    p_query.add_argument("query", type=str, help="User question")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--embeddings-provider", type=str, default=None, choices=["ollama", "openai", "anthropic"], help="Embeddings provider")
    p_query.add_argument("--embeddings-model", type=str, default=None, help="Embeddings model name")
    p_query.add_argument("--chat-provider", type=str, default=None, choices=["ollama", "openai", "anthropic"], help="Chat provider")
    p_query.add_argument("--chat-model", type=str, default=None, help="Chat model name")
    p_query.set_defaults(func=query_index)

    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

