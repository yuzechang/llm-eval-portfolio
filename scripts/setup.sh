#!/bin/bash
# ============================================================
# 大模型测试工程师培训 · 环境搭建脚本
# 用法: bash setup.sh
# 说明: 一键创建 venv 并安装核心依赖(P0/W1 使用)
# ============================================================

set -e

echo "==> 创建 Python 虚拟环境(用 python3.11,DeepEval/Ragas 要求 3.10+)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

echo "==> 升级 pip"
pip install --upgrade pip

echo "==> 安装 RAG 框架(二选一,建议先都装,后续锁定一个)"
pip install langchain langchain-community langchain-openai
pip install llama-index

echo "==> 安装向量库(Chroma 本地轻量)"
pip install chromadb

echo "==> 安装评测框架(核心)"
pip install ragas deepeval

echo "==> 安装 LLM 客户端(OpenAI 兼容,兼容 DeepSeek/通义/智谱)"
pip install openai

echo "==> 安装工具库"
pip install pandas python-dotenv pytest

echo ""
echo "==> 完成!下一步:"
echo "    1. 创建 .env 文件,填入你的 API key:"
echo "       OPENAI_API_KEY=sk-xxx"
echo "       OPENAI_BASE_URL=https://api.deepseek.com  # 或通义/智谱的地址"
echo "    2. 验证安装: python -c 'import deepeval, ragas, chromadb; print(\"ok\")'"
