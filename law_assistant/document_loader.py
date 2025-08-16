import os
from typing import Iterator, Dict


def iter_files(root: str, exts=(".md", ".txt", ".html", ".json")) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if exts and not any(fn.lower().endswith(e) for e in exts):
                continue
            yield os.path.join(dirpath, fn)


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp949", errors="ignore") as f:
            return f.read()


def load_documents(root: str) -> Iterator[Dict]:
    for path in iter_files(root):
        text = read_text(path)
        if not text.strip():
            continue
        yield {
            "id": path,
            "path": path,
            "text": text,
        }

