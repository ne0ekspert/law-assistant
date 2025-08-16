from typing import List, Dict


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    if chunk_size <= 0:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def chunk_document(doc: Dict, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Dict]:
    chunks = split_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = []
    for i, ch in enumerate(chunks):
        out.append({
            "doc_id": doc["id"],
            "chunk_id": f"{doc['id']}::chunk_{i}",
            "text": ch,
            "metadata": {
                "path": doc.get("path", doc["id"]),
                "index": i,
                "total_chunks": len(chunks),
            },
        })
    return out

