Okay, let's see. The original code was a short comment and README title. Now apply the suggested edit fully, only output the modified markdown code as requested.```
# agent-apply
对不同agent范式进行代码实现和效果评估

## 项目简介

本项目实现了三种不同的智能体控制流，并提供了完整的评估框架来对比它们的性能表现。

## 实现的智能体

### 1. DirectPromptAgent（基础直接提示）
- 最简单的Agent实现
- 直接将用户问题传递给LLM获取回答
- 适用于简单直接的问答场景

### 2. ReActAgent（ReAct范式）
- 结合推理(Reasoning)和行动(Acting)
- 实现了工具调用机制（搜索、计算）
- 通过思考-行动-观察循环解决复杂问题
- 可配置最大步数（默认为3步）

### 3. ReflexionAgent（Reflexion架构）
- 带有自我反思机制
- 生成回答后会反思回答质量
- 根据反思结果迭代改进答案
- 可配置最大反思次数（默认为2次）

## 文件结构

模型：测试模型使用Qwen-flash，评估模型使用Qwen-plus。
测试集：使用GSM8K测试集中的25个题目和HotpotQA数据集中的25个题目作为评估题目。
实践优化过程：
原始agent和评估代码实现；
ReactAgent的工具的具体实现，特别是搜索工具从本地知识库模拟转为真实搜索。
agent结果提取函数优化，由于Reactagent的输出往往是答案在前，然后补充步骤，导致对答案的提取往往会出现后续步骤，影响评分模型的判断，于是我寻找规律，使用正则字符串精确匹配"最终答案"到最近"Step"的中间字段，达成对答案的精准锁定。。
evaluate评估由原本的机械式关键词匹配改为更先进的大模型评估。
交互轮次由广义上的解决问题的轮次改为agent和大模型的交流次数。
由于现有模型的能力都很强，单纯的判断对错不足以区分不同agent的答案，于是采取分数制。
