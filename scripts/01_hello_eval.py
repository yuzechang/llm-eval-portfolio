# -*- coding: utf-8 -*-
"""
01_hello_eval.py —— 你的第一个 LLM 评测(最小可运行 demo)

作用:调 DeepSeek(OpenAI 兼容协议)跑一个 DeepEval 评测指标,
     走通「环境 → 调 LLM 打分 → 输出评测结果」这条完整链路。

前提条件:
  1. 已运行 scripts/setup.sh 装好依赖(deepeval、openai、python-dotenv 等)
  2. 在项目根目录(stu_ai_test/)下建了 .env 文件,内容:
        OPENAI_API_KEY=sk-你的DeepSeek key
        OPENAI_BASE_URL=https://api.deepseek.com

运行方式:
  cd other/stu_ai_test
  source .venv/bin/activate
  python scripts/01_hello_eval.py
"""

import os
from dotenv import load_dotenv

# 1. 加载 .env(默认从当前工作目录找,请在 stu_ai_test/ 目录下运行)
load_dotenv()

# 2. 配置 DeepEval 使用 DeepSeek(OpenAI 兼容协议)
#    DeepEval 默认用 OpenAI 接口,这里把 base_url 指到 DeepSeek 即可复用
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


def main():
    # 3. 构造一个「测试用例」:
    #    - input:   用户问的问题
    #    - actual_output: 模型给出的回答(这里手写一个,方便你理解评测对象)
    #    - retrieval_context: 检索到的参考上下文(评测会用它对答案做判断)
    test_case = LLMTestCase(
        input="什么是 RAG?",
        actual_output="RAG 是检索增强生成,先从知识库检索相关内容,再让模型基于这些内容生成答案。",
        retrieval_context=[
            "RAG(Retrieval-Augmented Generation,检索增强生成)是一种先检索外部知识、再基于检索结果生成回答的技术,能有效缓解大模型的知识过时和幻觉问题。"
        ],
    )

    # 4. 选一个评测指标:答案相关性(Answer Relevancy),阈值 0.5(0~1 之间)
    metric = AnswerRelevancyMetric(threshold=0.5)

    # 5. 跑评测,evaluate() 会自动打印结果报告(分数 + 是否通过)
    evaluate([test_case], [metric])

    print("\n>>> 恭喜!你已跑通第一个 LLM 评测。")
    print(">>> 下一步:把 actual_output 改成一句「答非所问」的话,再看分数会怎么变。")


if __name__ == "__main__":
    main()
