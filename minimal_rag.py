import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from dotenv import load_dotenv

load_dotenv()


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STORAGE_DIR = ROOT / "storage"
INDEX_PATH = STORAGE_DIR / "index.json"


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Reading PDF files requires pypdf. Run: pip install -r requirements.txt") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use .txt, .md, or .pdf")


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower())
    tokens: list[str] = []
    for token in raw_tokens:
        tokens.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.extend(token)
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
    return tokens


def local_hash_embedding(text: str, dimensions: int = 512) -> list[float]:
    """A tiny local embedding for learning RAG without an embedding API."""
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dimensions
        vector[index] += 1.0
    return vector


def mimo_chat(question: str, context: str) -> str:
    api_key = (os.getenv("MIMO_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("没有找到 MIMO_API_KEY。请先在终端设置环境变量：$env:MIMO_API_KEY=\"你的apikey\"")

    prompt = f"""你是一个严谨的知识库问答助手。
请只根据【参考资料】回答问题。
如果用户询问“知识库里有什么、包含什么、能做什么”，请根据【参考资料】概括知识库内容。
如果参考资料里确实没有答案，请直接说“资料中没有提到”。

【参考资料】
{context}

【用户问题】
{question}
"""

    default_url = (
        "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
        if api_key.startswith("tp-")
        else "https://api.xiaomimimo.com/v1/chat/completions"
    )
    url = (os.getenv("MIMO_CHAT_URL") or default_url).strip()
    if api_key.startswith("tp-") and "api.xiaomimimo.com" in url:
        print("检测到 tp- 开头的 Token Plan Key，已自动切换到 Token Plan 中国区接口。")
        url = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
    headers = {
        "api-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
            "model": os.getenv("MIMO_CHAT_MODEL", "mimo-v2.5-pro"),
            "messages": [
                {"role": "system", "content": "你是 MiMo 驱动的 RAG 知识库问答助手。"},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False,
            "thinking": {"type": "disabled"},
        }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )
    if response.status_code >= 400:
        print("\nMiMo API 请求失败，下面是原始调试信息：")
        print(f"请求地址：{url}")
        print(f"模型名称：{payload['model']}")
        print(f"状态码：{response.status_code}")
        print(f"返回内容：{response.text}")
        response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def save_index(items: list[dict[str, Any]]) -> None:
    STORAGE_DIR.mkdir(exist_ok=True)
    INDEX_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index() -> list[dict[str, Any]]:
    if not INDEX_PATH.exists():
        raise RuntimeError("No index found. Run: python minimal_rag.py ingest data/example.txt")
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def ingest(path: Path) -> None:
    text = load_document(path)
    chunks = split_text(text)

    items = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"正在为文本块生成向量 {index}/{len(chunks)}...")
        items.append(
            {
                "source": str(path),
                "chunk_id": index,
                "text": chunk,
                "embedding": local_hash_embedding(chunk),
            }
        )

    save_index(items)
    print(f"已将 {len(items)} 个文本块写入索引：{INDEX_PATH}")


def retrieve(question: str, top_k: int = 3) -> list[dict[str, Any]]:
    items = load_index()
    question_embedding = local_hash_embedding(question)

    scored = []
    for item in items:
        score = cosine_similarity(question_embedding, item["embedding"])
        scored.append({**item, "score": score})

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def ask(question: str, top_k: int = 3) -> None:
    docs = retrieve(question, top_k=top_k)
    context = "\n\n".join(
        f"[source={doc['source']} chunk={doc['chunk_id']} score={doc['score']:.4f}]\n{doc['text']}"
        for doc in docs
    )
    answer = mimo_chat(question, context)

    print("\n回答：\n")
    print(answer)
    print("\n检索到的参考文本块：\n")
    for doc in docs:
        print(f"- 文本块={doc['chunk_id']} 相似度={doc['score']:.4f} 来源={doc['source']}")


def interactive_mode() -> None:
    api_key = os.getenv("MIMO_API_KEY", "")
    print("欢迎使用最小 RAG 学习程序")
    if api_key:
        print(f"已读取 MIMO_API_KEY：{api_key[:4]}...{api_key[-4:]}")
    else:
        print("未读取到 MIMO_API_KEY，请检查 .env 或环境变量设置。")
    print("请先选择操作：")
    print("1. 建立知识库索引")
    print("2. 基于知识库提问")

    choice = input("请输入 1 或 2：").strip()

    if choice == "1":
        default_file = DATA_DIR / "example.txt"
        user_file = input(f"请输入文档路径，直接回车使用默认文档 {default_file}：").strip()
        file_path = Path(user_file) if user_file else default_file
        ingest(file_path)
        return

    if choice == "2":
        question = input("请输入你的问题：").strip()
        if not question:
            print("问题不能为空。")
            return
        top_k_text = input("请输入要检索的参考文本块数量，直接回车默认 3：").strip()
        top_k = int(top_k_text) if top_k_text else 3
        ask(question, top_k=top_k)
        return

    print("输入无效，请重新运行程序并输入 1 或 2。")


def main() -> None:
    load_dotenv(ROOT / ".env")

    if len(sys.argv) == 1:
        interactive_mode()
        return

    parser = argparse.ArgumentParser(description="一个不依赖 LangChain 的最小 RAG 学习项目。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="读取文档并建立向量索引。")
    ingest_parser.add_argument("file", type=Path)

    ask_parser = subparsers.add_parser("ask", help="基于已经入库的文档进行提问。")
    ask_parser.add_argument("question", help="你的中文问题，例如：RAG 的基本流程是什么？")
    ask_parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.file)
    elif args.command == "ask":
        ask(args.question, top_k=args.top_k)


if __name__ == "__main__":
    main()
