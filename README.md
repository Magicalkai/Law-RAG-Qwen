# 🏛 Law-RAG-Qwen: 基于 Qwen + LoRA + RAG 的法律智能问答系统

本项目旨在构建一个专业的法律智能问答系统，通过结合 Qwen-6B 大型语言模型、LoRA 轻量级微调技术和 RAG（检索增强生成）架构，实现基于垂直领域知识库的高质量、可信赖的法律咨询服务。

## ⚙️ 核心技术栈

*   **基座模型 (LLM):** Qwen-3.0-6B (阿里通义千问)
*   **微调技术:** LoRA (Low-Rank Adaptation)
*   **向量模型 (Embedding):** BAAI/bge-small-zh-v1.5
*   **向量数据库 (Vector DB):** FAISS (Facebook AI Similarity Search)
*   **框架:** PyTorch, Hugging Face `transformers`, `peft`, `datasets`

## 🛠️ 环境配置

请首先安装必要的依赖：

```bash
pip install -r requirements.txt
