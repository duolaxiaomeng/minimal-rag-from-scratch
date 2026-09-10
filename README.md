# Minimal RAG From Scratch

这是一个用来学习 RAG 原理的最小项目。它暂时不使用 LangChain，目的是让你看清楚 RAG 的完整链路：

```text
文档 -> 切块 -> Embedding -> 保存向量 -> 问题 Embedding -> 相似度检索 -> 拼 Prompt -> 大模型回答
```

## 1. 安装依赖

```powershell
git clone https://github.com/duolaxiaomeng/minimal-rag-from-scratch.git
cd minimal-rag-from-scratch
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 设置小米 MiMo API Key

这个版本默认使用小米 MiMo 的 OpenAI 兼容聊天接口，并使用 `python-dotenv` 自动读取项目根目录下的 `.env` 文件。

注意：目前这个最小学习版只用 MiMo 负责“生成回答”。检索阶段先使用本地 `local_hash_embedding()` 生成简单向量，这样你只需要一个 MiMo API Key 就能跑通完整 RAG 流程。

```powershell
$env:MIMO_API_KEY="your_api_key_here"
```

如果你的 Key 是 Coding Plan / Token Plan，一般是 `tp-xxxxx` 开头。代码会自动使用中国区 Token Plan 地址：

```text
https://token-plan-cn.xiaomimimo.com/v1/chat/completions
```

如果你的订阅页显示的是新加坡区或欧洲区，请手动设置：

```powershell
$env:MIMO_CHAT_URL="https://token-plan-sgp.xiaomimimo.com/v1/chat/completions"
```

或：

```powershell
$env:MIMO_CHAT_URL="https://token-plan-ams.xiaomimimo.com/v1/chat/completions"
```

可选模型配置：

```powershell
$env:MIMO_CHAT_MODEL="mimo-v2.5-pro"
```

## 3. 建立知识库索引

如果你使用 PyCharm，可以直接运行 `minimal_rag.py`，然后在终端里输入：

```text
1
```

再按提示输入文档路径，或者直接回车使用默认文档。

也可以使用命令行参数：

```powershell
python minimal_rag.py ingest data/example.txt
```

运行后会生成：

```text
storage/index.json
```

里面保存了每个文本块和它对应的 embedding。

## 4. 提问

如果你使用 PyCharm，可以直接运行 `minimal_rag.py`，然后在终端里输入：

```text
2
RAG 的基本流程是什么？
```

也可以使用命令行参数：

```powershell
python minimal_rag.py ask "RAG 的基本流程是什么？"
```

程序会输出中文内容：

- 大模型回答
- 被检索到的文本块
- 每个文本块的相似度分数

## 你需要重点理解的代码

- `split_text()`：把长文本切成小块。
- `local_hash_embedding()`：用本地 hashing 方法把文本变成简单向量。
- `cosine_similarity()`：计算问题和文本块的相似度。
- `retrieve()`：找出最相关的 top-k 文本块。
- `mimo_chat()`：把检索结果塞进 prompt，再调用 MiMo 大模型回答。

## 下一步升级方向

1. 支持多个文件同时入库。
2. 把 `local_hash_embedding()` 换成真正的 embedding 模型。
3. 把 `storage/index.json` 换成 ChromaDB。
4. 用 FastAPI 包装成 `/upload` 和 `/chat` 接口。
5. 保存聊天历史到 MySQL。
6. 加一个 Streamlit 或 Vue 前端。
