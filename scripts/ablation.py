"""消融实验：单变量关闭 Rerank / 混合检索，量化各组件贡献。

变体（同测试集、同生成模型、同日执行，单变量对照）：
  A  基线     混合检索（向量+BM25+RRF）+ LLM Rerank
  B  无Rerank 混合检索，跳过精排，直接取 RRF 融合 top_k
  C  纯向量   Phase 0 基线检索（无 BM25/RRF/Rerank）

用法：
    python scripts/ablation.py --variant A --limit 2   # 单变体冒烟
    python scripts/ablation.py --variant A             # 只跑变体A（全量）
    python scripts/ablation.py                         # A/B/C 顺序全量

结果：
    - 每变体明细: data/eval/results/ablation-{A|B|C}.json
    - 对比表:     data/eval/results/ablation-comparison.md
    - 指标台账:   metrics/metrics_log.md 自动追加（阶段列标注消融变体）
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kbqa.config import CONFIG
from kbqa.logger import get_logger
from kbqa.pipeline import RAGPipeline
from evaluate import (
    RESULTS_DIR,
    TESTSET_PATH,
    build_ragas_wrappers,
    run_pipeline_on_testset,
)

logger = get_logger("ablation")

VARIANTS = {
    "A": {"label": "消融A·基线(混合+Rerank)", "flags": {"use_hybrid": True, "use_rerank": True}},
    "B": {"label": "消融B·无Rerank(混合)", "flags": {"use_hybrid": True, "use_rerank": False}},
    "C": {"label": "消融C·纯向量(Phase0)", "flags": {"use_hybrid": False, "use_rerank": False}},
}


def run_ragas(records: list[dict]) -> dict:
    print("  Ragas LLM-as-Judge 评测中（较慢）...")
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

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
    df = result.to_pandas()
    scores = {}
    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        col = metric.name
        if col in df.columns:
            valid = df[col].dropna()
            scores[col] = round(float(valid.mean()), 3) if len(valid) else None
    return scores


def append_metrics_row(s: dict) -> None:
    metrics_dir = PROJECT_ROOT / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    sc = s["ragas_scores"]
    row = (
        f"| {s['timestamp'][:10]} | {s['label']} | {s['kb_chunks']} | {s['model']} "
        f"| {s['hit_rate']:.0%} | {sc.get('faithfulness', '-')} | {sc.get('context_precision', '-')} "
        f"| {sc.get('context_recall', '-')} | {sc.get('answer_relevancy', '-')} "
        f"| {s['avg_retrieval_s']} | {s['avg_llm_s']} | {s['avg_latency_s']} |\n"
    )
    with open(metrics_dir / "metrics_log.md", "a", encoding="utf-8") as f:
        f.write(row)
    logger.info(f"[ablation] 指标已追加到 metrics/metrics_log.md")


def run_variant(key: str, testset: list[dict], model: str) -> dict:
    spec = VARIANTS[key]
    for attr, value in spec["flags"].items():
        setattr(CONFIG, attr, value)
    logger.info(f"[ablation] 变体{key} {spec['label']} flags={spec['flags']} 共{len(testset)}题")

    pipeline = RAGPipeline()  # use_hybrid 在初始化时读取，每个变体必须重建
    if pipeline.store.count() == 0:
        logger.error("知识库为空，请先执行 python cli.py ingest")
        sys.exit(1)

    # 预热：触发本地 BGE 加载与 Rerank LLM 客户端初始化，
    # 避免首题冷启动（~17s）混入均值，保证测的是稳态耗时
    pipeline.embedder.embed_query("预热")
    if CONFIG.use_rerank:
        pipeline.retriever.retrieve("预热查询")

    records = run_pipeline_on_testset(pipeline, testset, model)
    scores = run_ragas(records)

    ok = [r for r in records if "error" not in r]
    summary = {
        "variant": key,
        "label": spec["label"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "num_cases": len(records),
        "kb_chunks": pipeline.store.count(),
        "flags": spec["flags"],
        "hit_rate": round(sum(1 for r in records if r["source_hit"]) / len(records), 3),
        "ragas_scores": scores,
        "avg_retrieval_s": round(sum(r["retrieval_s"] for r in ok) / len(ok), 2) if ok else None,
        "avg_llm_s": round(sum(r["llm_s"] for r in ok) / len(ok), 2) if ok else None,
        "avg_latency_s": round(sum(r["latency_s"] for r in records) / len(records), 2),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"ablation-{key}.json"
    out.write_text(
        json.dumps({"summary": summary, "cases": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"[ablation] 变体{key} 结果已保存: {out}")

    append_metrics_row(summary)
    return summary


def fmt(v, base=None, nd=3):
    if v is None:
        return "-"
    if base is None or base == "-":
        return f"{v}"
    d = v - base
    sign = "+" if d >= 0 else ""
    return f"{v} ({sign}{round(d, nd)})"


def print_comparison() -> None:
    summaries = []
    for key in VARIANTS:
        p = RESULTS_DIR / f"ablation-{key}.json"
        if p.exists():
            summaries.append(json.loads(p.read_text(encoding="utf-8"))["summary"])
    if not summaries:
        return
    base = next((s for s in summaries if s["variant"] == "A"), summaries[0])
    b = base["ragas_scores"]

    lines = [
        f"# 消融实验对比表",
        "",
        f"- 基准: {base['label']}  测试集: {base['num_cases']}题  生成模型: {base['model']}  Chunk数: {base['kb_chunks']}",
        f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}",
        f"- 变体B/C 括号内为相对基准的差值",
        "",
        "| 变体 | 配置 | 命中率 | Faithfulness | Context Precision | Context Recall | Answer Relevancy | 平均检索(s) | 平均生成(s) | 平均端到端(s) |",
        "|------|------|--------|--------------|-------------------|----------------|------------------|-------------|-------------|---------------|",
    ]
    for s in summaries:
        sc = s["ragas_scores"]
        ref = b if s["variant"] != "A" else None
        flags = "+".join(k for k, v in s["flags"].items() if v) or "全关"
        lines.append(
            f"| {s['variant']} | {flags} | {s['hit_rate']:.0%} "
            f"| {fmt(sc.get('faithfulness'), (ref or {}).get('faithfulness'))} "
            f"| {fmt(sc.get('context_precision'), (ref or {}).get('context_precision'))} "
            f"| {fmt(sc.get('context_recall'), (ref or {}).get('context_recall'))} "
            f"| {fmt(sc.get('answer_relevancy'), (ref or {}).get('answer_relevancy'))} "
            f"| {fmt(s['avg_retrieval_s'], base['avg_retrieval_s'] if ref else None, 2)} "
            f"| {fmt(s['avg_llm_s'], base['avg_llm_s'] if ref else None, 2)} "
            f"| {fmt(s['avg_latency_s'], base['avg_latency_s'] if ref else None, 2)} |"
        )
    out = RESULTS_DIR / "ablation-comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n消融对比表已写入: {out}")
    logger.info(f"[ablation] 对比表已保存: {out}")


def main():
    parser = argparse.ArgumentParser(description="RAG 消融实验")
    parser.add_argument("--limit", type=int, default=None, help="每个变体只评测前 N 题（冒烟用）")
    parser.add_argument("--variant", choices=list(VARIANTS), default=None, help="只跑指定变体")
    args = parser.parse_args()

    model = CONFIG.llm_model
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    if args.limit:
        testset = testset[: args.limit]

    keys = [args.variant] if args.variant else list(VARIANTS)
    for key in keys:
        run_variant(key, testset, model)
    print_comparison()


if __name__ == "__main__":
    main()
