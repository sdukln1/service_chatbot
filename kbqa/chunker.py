from __future__ import annotations

SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，"]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    for sep in SEPARATORS:
        if sep and sep in text:
            parts = [p for p in text.split(sep) if p.strip()]
            if len(parts) > 1:
                chunks = []
                buffer = ""
                for part in parts:
                    piece = (buffer + sep + part) if buffer else part
                    if len(piece) <= size:
                        buffer = piece
                        continue
                    if buffer:
                        chunks.append(buffer)
                        tail = buffer[-overlap:]
                        buffer = tail if tail.strip() else ""
                    if len(part) > size:
                        chunks.extend(_hard_split(part, size, overlap))
                        buffer = ""
                    else:
                        buffer = part
                if buffer:
                    chunks.append(buffer)
                return [c.strip() for c in chunks if c.strip()]

    return _hard_split(text, size, overlap)


def chunk_text(
    text: str, size: int | None = None, overlap: int | None = None
) -> list[str]:
    from kbqa.config import CONFIG

    size = size if size is not None else CONFIG.chunk_size
    overlap = overlap if overlap is not None else CONFIG.chunk_overlap
    return split_text(text, size, overlap)
