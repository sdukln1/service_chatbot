"""Phase 2: RAG 全链路评测脚本。

流程：
1. 读取 data/eval/testset.json（问题-标准答案-预期来源 三元组）
2. 走完整 RAG 链路（检索 → Rerank → LLM 生成）得到每题的答案与上下文
3. 用 Ragas 计算四指标：
   - Faithfulness（忠实度/防幻觉）
   - Context Precision（上下文精确度）
   - Context Recall（上下文召回率）
   - Answer Relevancy（答案相关性）
4. 输出报告 JSON + 控制台摘要

用法：
    python scripts/evaluate.py                 # 全量 20 题
    python scripts/evaluate.py --limit 5      # 快速冒烟
    python scripts/evaluate.py --model GLM-5.3 # 指定生成模型
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx

from kbqa.config import CONFIG
from kbqa.logger import get_logger
from kbqa.pipeline import RAGPipeline
from kbqa.retriever import Retriever

logger = get_logger("evaluate")

TESTSET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "testset.json"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "eval" / "results"


def build_ragas_wrappers():
    """远程 LLM + 本地 BGE，包装成 ragas 需要的 langchain 对象。"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    sync_http = httpx.Client(verify=CONFIG.verify_ssl, trust_env=False)
    async_http = httpx.AsyncClient(verify=CONFIG.verify_ssl, trust_env=False)

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=CONFIG.llm_model,
            api_key=CONFIG.api_key,
            base_url=CONFIG.api_base,
            temperature=0,
            http_client=sync_http,
            http_async_client=async_http,
        )
    )

    from kbqa.config import resolve_model

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=resolve_model(CONFIG.embedding_model),
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    return evaluator_llm, evaluator_embeddings


def run_pipeline_on_testset(pipeline: RAGPipeline, testset: list[dict], model: str) -> list[dict]:
    records = []
    for i, item in enumerate(testset, 1):
        question = item["question"]
        logger.info(f"[eval {i}/{len(testset)}] {question}")
        start = time.time()
        try:
            t0 = time.time()
            hits = pipeline.retriever.retrieve(question)
            retrieval_s = round(time.time() - t0, 2)
            contexts = [h["text"] for h in hits]
            context = Retriever.format_context(hits)
            t0 = time.time()
            answer = pipeline._get_generator().generate(question, context, model=model)
            llm_s = round(time.time() - t0, 2)
            retrieved_sources = [h["metadata"].get("source") for h in hits]
            hit_expected = item["source"] in retrieved_sources
            records.append(
                {
                    "user_input": question,
                    "retrieved_contexts": contexts,
                    "response": answer,
                    "reference": item["ground_truth"],
                    "expected_source": item["source"],
                    "retrieved_sources": retrieved_sources,
                    "source_hit": hit_expected,
                    "retrieval_s": retrieval_s,
                    "llm_s": llm_s,
                    "latency_s": round(time.time() - start, 2),
                }
            )
            logger.info(
                f"[eval {i}] 完成 检索={retrieval_s}s 生成={llm_s}s 来源命中={hit_expected}"
            )
        except Exception as e:
            logger.error(f"[eval {i}] 失败: {type(e).__name__}: {e}")
            records.append(
                {
                    "user_input": question,
                    "retrieved_contexts": [],
                    "response": f"ERROR: {e}",
                    "reference": item["ground_truth"],
                    "expected_source": item["source"],
                    "retrieved_sources": [],
                    "source_hit": False,
                    "retrieval_s": None,
                    "llm_s": None,
                    "latency_s": round(time.time() - start, 2),
                    "error": str(e),
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser(description="RAG 评测")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 题")
    parser.add_argument("--model", default=None, help="生成用 LLM")
    args = parser.parse_args()

    model = args.model or CONFIG.llm_model
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    if args.limit:
        testset = testset[: args.limit]
    logger.info(f"[eval] 开始评测 共 {len(testset)} 题 生成模型={model}")

    pipeline = RAGPipeline()
    if pipeline.store.count() == 0:
        logger.error("知识库为空，请先执行 python cli.py ingest")
        sys.exit(1)

    records = run_pipeline_on_testset(pipeline, testset, model)

    print("\n正在运行 Ragas 四指标评测（LLM-as-Judge，较慢）...")
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas import EvaluationDataset

    evaluator_llm, evaluator_embeddings = build_ragas_wrappers()
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "retrieved_contexts": r["retrieved_contexts"],
                "response": r["response"],
                "reference": r["reference"],
            }
            for r in records
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = result.to_pandas()

    ragas_scores = {}
    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        col = metric.name
        if col in df.columns:
            valid = df[col].dropna()
            ragas_scores[col] = round(float(valid.mean()), 3) if len(valid) else None

    ok_records = [r for r in records if "error" not in r]
    avg_retrieval = (
        round(sum(r["retrieval_s"] for r in ok_records) / len(ok_records), 2)
        if ok_records
        else None
    )
    avg_llm = (
        round(sum(r["llm_s"] for r in ok_records) / len(ok_records), 2)
        if ok_records
        else None
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "testset": str(TESTSET_PATH),
        "num_cases": len(records),
        "kb_chunks": pipeline.store.count(),
        "retrieval_source_hit_rate": round(
            sum(1 for r in records if r["source_hit"]) / len(records), 3
        ),
        "avg_retrieval_s": avg_retrieval,
        "avg_llm_s": avg_llm,
        "avg_latency_s": round(sum(r["latency_s"] for r in records) / len(records), 2),
        "ragas_scores": ragas_scores,
        "cases": records,
    }
    out = RESULTS_DIR / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    append_metrics_log(report)

    print("\n========== 评测报告 ==========")
    print(f"用例数: {len(records)}  生成模型: {model}  Chunk数: {report['kb_chunks']}")
    print(f"预期文档命中率: {report['retrieval_source_hit_rate']:.1%}")
    print(f"平均检索耗时: {avg_retrieval}s  平均生成耗时: {avg_llm}s")
    print(f"平均端到端耗时: {report['avg_latency_s']}s")
    print("\nRagas 四指标:")
    for col, score in ragas_scores.items():
        print(f"  {col}: {score if score is not None else 'N/A'}")
    print(f"\n详细结果: {out}")
    logger.info(f"[eval] 报告已保存: {out}")


def append_metrics_log(report: dict) -> None:
    metrics_dir = PROJECT_ROOT / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics_file = metrics_dir / "metrics_log.md"
    header = (
        "# 智能体指标记录\n\n"
        "> 每次评测后脚本会**自动追加**一条记录（`python scripts/evaluate.py`）。\n"
        "> 指标含义见 README「评测体系」章节。检索耗时含 embedding/向量检索/BM25/RRF/Rerank，"
        "生成耗时为 LLM 生成。\n\n"
        "| 日期 | 阶段/配置 | Chunk数 | 生成模型 | 命中率 | Faithfulness | "
        "Context Precision | Context Recall | Answer Relevancy | 平均检索(s) | 平均生成(s) | 平均端到端(s) |\n"
        "|------|-----------|---------|----------|--------|--------------|"
        "-------------------|----------------|------------------|-------------|-------------|---------------|\n"
    )
    if not metrics_dir.joinpath("metrics_log.md").exists():
        metrics_dir.joinpath("metrics_log.md").write_text(header, encoding="utf-8")

    s = report["ragas_scores"]
    row = (
        f"| {report['timestamp'][:10]} | 手动评测 | {report['kb_chunks']} "
        f"| {report['model']} | {report['retrieval_source_hit_rate']:.0%} "
        f"| {s.get('faithfulness', '-')} | {s.get('context_precision', '-')} "
        f"| {s.get('context_recall', '-')} | {s.get('answer_relevancy', '-')} "
        f"| {report['avg_retrieval_s']} | {report['avg_llm_s']} "
        f"| {report['avg_latency_s']} |\n"
    )
    with open(metrics_dir / "metrics_log.md", "a", encoding="utf-8") as f:
        f.write(row)
    logger.info(f"[eval] 指标已追加到 {metrics_dir / 'metrics_log.md'}")


if __name__ == "__main__":
    main()
