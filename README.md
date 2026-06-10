# Agent-Apply 智能体范式对比评估平台
本项目是一个专业的大语言模型智能体（Agent）架构对比评估系统，实现了三种主流Agent范式的完整实现，支持权威数据集接入、大模型横向对比裁判、多维度性能分析与自动可视化报告，帮助开发者客观评测不同Agent架构的能力差异。

## 🚀 核心特性
### 🧠 三种Agent架构完整实现
- **DirectPromptAgent**：基础直接提示智能体，无工具调用能力与反思机制，作为性能基线对照
- **ReActAgent**：Reasoning + Acting 范式实现，支持多步工具调用与链式推理，适合需要外部信息的复杂场景
- **ReflexionAgent**：带自我反思迭代机制的智能体，可自动发现回答错误并修正，复杂推理场景性能显著提升
### 🛠️ 内置5类工具支持
所有Agent均可调用以下工具：
- 🔍 `search`：事实性信息搜索，内置知识库可模拟真实搜索效果
- 🧮 `calculate`：安全数学表达式计算，支持复杂运算
- ⏰ `datetime`：当前日期时间查询工具
- 📐 `math`：数学常数与专业公式查询工具
### 📊 权威测试数据集支持
- **云端权威数据集**：支持自动拉取两大行业标准测试集：
  - GSM8K 数学推理数据集：25道中小学数学计算题，考察计算推理能力
  - HotpotQA 多跳搜索数据集：25道事实类多跳查询题，考察信息检索与整合能力
### ⚖️ 智能评估体系
- **大模型横向对比裁判**：引入大模型作为中立裁判，同时对比三个Agent的回答，避免单独打分的不公平性，可精准区分回答质量差异
- **匹配评估策略**：
  - `llm_judge`：大模型裁判评估，适合开放类、复杂推理类题目
### 📈 多维度分析与可视化
- 自动统计多维度指标：总成功率、按难度/工具依赖拆分成功率、平均交互轮数、平均处理耗时
- 自动生成4维度可视化对比图表：各Agent成功率对比、平均耗时对比、不同难度成功率对比、工具依赖对成功率影响
- 评估结果自动导出为JSON格式，支持后续分析
## 📁 项目目录结构
```
agent-apply/
├── agent.py               # 三种Agent核心实现代码
├── api.py                 # 大语言模型API封装层，可自定义接入不同模型
├── evaluate.py            # 核心评估框架，包含测试用例、评估逻辑、结果分析
├── plot.py                # 可视化图表生成模块
├── recalculate.py         # 评估结果重计算模块，支持二次分析
├── test.py                # 单功能测试脚本
├── report/                # 评估报告输出目录
├── result/                # 评估结果JSON文件存储目录
├── __pycache__/           # Python缓存目录
```
## 🛠️ 环境要求
- Python 3.8+
- 系统环境变量需配置 `HF_TOKEN` 用于拉取Hugging Face公开数据集
- 依赖库：
  ```
  matplotlib >= 3.5.0
  requests >= 2.28.0
  datasets >= 2.10.0
  ```
## 📦 安装与配置
### 1. 下载项目
```bash
git clone <https://github.com/jiling-shenyi/agent-apply>
cd agent-apply
```
### 2. 安装依赖
```bash
pip install -r requirements.txt
```
### 3. 配置环境变量
配置Hugging Face Token用于拉取公开数据集：
```bash
# Windows PowerShell
$env:HF_TOKEN = "你的HuggingFace Token"
# Linux/macOS
export HF_TOKEN="你的HuggingFace Token"
```
### 4. 配置大模型API
编辑 `api.py` 文件，实现 `api_call` 函数，可接入任意主流大模型接口（OpenAI、Anthropic、通义千问、文心一言等），默认已适配通义千问系列模型。
## ⚡ 快速开始
### 1. 交互式使用Agent
直接运行 `agent.py` 即可进入交互模式，自由切换不同Agent类型测试效果：
```bash
python agent.py
```
支持命令：
- `switch`：切换Agent类型
- `quit/exit`：退出交互模式
### 2. 运行完整评估
运行 `evaluate.py` 执行完整评估流程：
```bash
python evaluate.py
```
默认执行模式：自动拉取GSM8K 25题 + HotpotQA 25题（共50题）进行横向对比评估。
若要使用本地测试用例，修改 `evaluate.py` 中 `EvaluationFramework` 初始化参数：
```python
evaluator = EvaluationFramework(use_benchmark=False, num_tests=13)
```
### 3. 结果输出
评估完成后将自动：
1. 在控制台打印各Agent详细性能统计
2. 生成可视化对比图表 `agent_evaluation_dashboard.png`
3. 保存完整评估结果到 `evaluation_results_<timestamp>.json`
## 📊 评估指标说明
| 指标 | 说明 |
|------|------|
| 总得分率 | 所有测试用例的加权平均得分，范围0-100% |
| 按难度成功率 | 分别统计不同难度等级题目的得分率 |
| 工具依赖成功率 | 分别统计需要工具调用/不需要工具调用场景的得分率 |
| 平均交互轮数 | 完成单个任务平均需要的对话/推理轮次 |
| 平均处理耗时 | 完成单个任务平均耗时（单位：秒） |
## 🧩 自定义扩展
### 添加新的Agent类型
在 `agent.py` 中继承 `BaseAgent` 抽象类，实现 `ask` 方法即可，评估框架会自动识别。
### 添加新工具
在 `ReActAgent` 的 `_init_tools` 方法中注册新工具函数，同时在对应提示词模板中添加工具使用说明即可。
### 添加自定义测试用例
在 `evaluate.py` 的 `_generate_test_cases` 方法中添加新的 `TestCase` 实例，支持自定义匹配策略与评估规则。
### 自定义大模型裁判
修改 `_call_llm_judge_comparative_score` 方法中的模型参数，使用更高性能的大模型作为裁判可提升评估准确性。
## 💡 预期性能表现
正常评估结果应符合以下规律：
**得分率：ReflexionAgent > ReActAgent > DirectPromptAgent**
- DirectPromptAgent：适合简单纯推理场景，工具类题目正确率极低
- ReActAgent：在需要工具调用的场景相比基础版本有40%以上的性能提升
- ReflexionAgent：通过反思机制进一步降低错误率，复杂场景优势尤为明显
## ⚠️ 注意事项
1. 可根据模型能力调整 `ReActAgent(max_steps=)` 和 `ReflexionAgent(max_reflections=)` 参数，算力充足时可适当调大提升复杂任务表现
2. 如遇API速率限制，可在 `api.py` 中添加重试与延迟逻辑
3. 内置知识库仅作演示使用，生产环境可替换为真实搜索引擎接口
4. 首次运行会自动下载GSM8K与HotpotQA数据集，需保持网络畅通
## 📄 开源协议
本项目仅供学习研究使用。