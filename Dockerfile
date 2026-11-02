FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://mirrors.tools.huawei.com/pypi/simple \
    HF_ENDPOINT=https://hf-mirror.com

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 离线环境：把本地模型目录一起拷入（构建前先执行 scripts/download_models.py）
COPY data/models/ /app/data/models/

COPY kbqa/ /app/kbqa/
COPY static/ /app/static/
COPY scripts/ /app/scripts/
COPY cli.py server.py /app/

# 知识库文档由挂载卷提供，这里只建目录
RUN mkdir -p data/knowledge data/chroma log data/feedback

EXPOSE 8080

CMD ["python", "server.py"]
