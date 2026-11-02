from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kbqa.config import CONFIG


@dataclass
class Document:
    text: str
    metadata: dict


def load_documents(knowledge_dir: Path | None = None) -> list[Document]:
    knowledge_dir = knowledge_dir or CONFIG.knowledge_dir
    documents = []
    for path in sorted(knowledge_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        if text.strip():
            documents.append(
                Document(text=text, metadata={"source": path.name})
            )
    return documents
