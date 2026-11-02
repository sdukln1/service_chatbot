__all__ = ["RAGPipeline", "Retriever"]


def __getattr__(name):
    if name == "RAGPipeline":
        from kbqa.pipeline import RAGPipeline

        return RAGPipeline
    if name == "Retriever":
        from kbqa.retriever import Retriever

        return Retriever
    raise AttributeError(name)
