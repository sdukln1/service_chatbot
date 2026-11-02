"""预下载 Embedding 模型到 data/models/，供离线/受限网络环境使用。

所用 LLM 服务不提供 embeddings 接口，
因此向量检索所需的 Embedding 模型仍需在有外网的环境（如手机热点）下载一次。
Rerank 为 LLM 打分实现，无需下载 reranker 模型。

用法：
    python scripts/download_models.py
    python scripts/download_models.py --endpoint https://hf-mirror.com
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kbqa.config import CONFIG, resolve_model  # noqa: E402

MODELS = [CONFIG.embedding_model]


def download(endpoint: str | None) -> None:
    if endpoint:
        import os

        os.environ["HF_ENDPOINT"] = endpoint

    from huggingface_hub import snapshot_download

    for name in MODELS:
        target = CONFIG.models_dir / name.split("/")[-1]
        if target.exists() and any(target.iterdir()):
            print(f"已存在，跳过: {target}")
            continue
        print(f"下载 {name} -> {target} ...")
        snapshot_download(
            repo_id=name,
            local_dir=str(target),
            ignore_patterns=["*.h5", "*.ot", "*.msgpack", "onnx/*", "openvino/*"],
        )
    print("\n全部模型就绪。当前生效路径：")
    for name in MODELS:
        print(f"  {name} -> {resolve_model(name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预下载 Embedding/Rerank 模型")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="HuggingFace 镜像地址，如 https://hf-mirror.com",
    )
    args = parser.parse_args()
    download(args.endpoint)
