import argparse
import sys


def _get_pipeline():
    try:
        from kbqa.pipeline import RAGPipeline

        return RAGPipeline()
    except ModuleNotFoundError as e:
        print(f"缺少依赖 {e.name}，请先执行: pip install -r requirements.txt")
        sys.exit(1)


def cmd_ingest(_args):
    pipeline = _get_pipeline()
    pipeline.ingest()


def cmd_ask(args):
    pipeline = _get_pipeline()
    result = pipeline.query(args.question, model=args.model)
    print(result["answer"])


def cmd_chat(_args):
    pipeline = _get_pipeline()
    if pipeline.store.count() == 0:
        print("向量库为空，请先执行: python cli.py ingest")
        sys.exit(1)
    print("智能客服已启动（输入 q 退出）")
    while True:
        try:
            question = input("\n用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"q", "quit", "exit"}:
            break
        print("\n客服: ", end="", flush=True)
        for token in pipeline.stream_query(question):
            print(token, end="", flush=True)
        print()
    print("\n再见！")


def main():
    parser = argparse.ArgumentParser(description="kbqa 智能客服知识问答系统")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="解析知识库文档并写入向量库")
    subparsers.add_parser("chat", help="启动交互式问答")
    ask = subparsers.add_parser("ask", help="单次提问")
    ask.add_argument("question", help="问题内容")
    ask.add_argument("--model", default=None, help="指定 LLM 模型（如 GLM-5.3）")

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "ask":
        cmd_ask(args)


if __name__ == "__main__":
    main()
