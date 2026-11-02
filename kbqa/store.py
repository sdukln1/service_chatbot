from __future__ import annotations

from pathlib import Path

import chromadb

from kbqa.config import CONFIG


class VectorStore:
    def __init__(self, db_dir: Path | None = None, collection_name: str | None = None):
        db_dir = db_dir or CONFIG.db_dir
        collection_name = collection_name or CONFIG.collection_name
        client = chromadb.PersistentClient(path=str(db_dir))
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        start = self.collection.count()
        ids = [f"chunk_{start + i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def search(
        self, query_embedding: list[float], top_k: int | None = None
    ) -> list[dict]:
        top_k = top_k or CONFIG.top_k
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append({"text": doc, "metadata": meta, "score": 1.0 - dist})
        return hits
