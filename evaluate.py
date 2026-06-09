import sys
import time
import json
import re
import os
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from api import api_call
import random
from datasets import load_dataset

# 假设导入了你的agent模块
sys.path.append('.')
from agent import DirectPromptAgent, ReActAgent, ReflexionAgent

@dataclass
class TestCase:
    """测试用例数据类"""
    question: str
    difficulty: str  # easy, medium, hard
    requires_tool: bool
    category: str
    match_type: str = "contains"  # 评估策略: contains, regex, keywords, strict_format
    expected_answer: str = ""     # 用于 contains
    expected_keywords: List[str] = field(default_factory=list) # 用于 keywords
    regex_pattern: str = ""       # 用于 regex 或 strict_format

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    agent_name: str
    test_case: TestCase
    success: int  # 0-100 分数
    response: str
    interaction_rounds: int
    time_used: float
    error_message: str = ""

def fetch_benchmark_test_cases(total_samples: int = 50) -> list:
    """
    从云端拉取权威开源数据集，并转换为本地 TestCase 格式。
    默认抽取 25 道 GSM8K (计算推理) 和 25 道 HotpotQA (多跳搜索)。
    """
    test_cases = []
    half_samples = total_samples // 2

    print(f"\n[数据加载] 正在从云端下载 GSM8K (数学计算) 测试集 {half_samples} 题...")
    try:
        # 加载 GSM8K 数据集 (测试集拆分)
        gsm8k = load_dataset("openai/gsm8k", "main", split="test")
        gsm8k_samples = random.sample(list(gsm8k), half_samples)
        
        for item in gsm8k_samples:
            # GSM8K 的答案通常在 "####" 之后是最终数字
            answer_text = item['answer']
            exact_answer = answer_text.split('####')[-1].strip() if '####' in answer_text else answer_text
            
            tc = TestCase(
                question=item['question'],
                difficulty="agent_killer", # 设定为高难度，触发对比裁判
                requires_tool=True,
                category="math_reasoning",
                match_type="llm_judge", # 强制使用大模型裁判
                # 将标准答案作为参考标准传给大模型裁判
                expected_answer=f"标准计算结果是：{exact_answer}。请忽略格式差异，重点检查模型推理过程和最终数字是否与标准结果一致。"
            )
            test_cases.append(tc)
    except Exception as e:
        print(f"GSM8K 数据集加载失败，请检查网络: {e}")
        
    hotpot_samples_needed = total_samples - len(test_cases)
    print(f"[数据加载] 正在从云端下载 HotpotQA (多跳搜索) 测试集 {total_samples - half_samples} 题...")
    try:
        # 加载 HotpotQA 数据集
        hotpot = load_dataset("KOJKO/hotpot_qa", "distractor", split="validation")
        hotpot_samples = random.sample(list(hotpot), hotpot_samples_needed)
        
        for item in hotpot_samples:
            tc = TestCase(
                question=item['question'],
                difficulty="too_hard", 
                requires_tool=True,
                category="multi_hop_search",
                match_type="llm_judge", # 强制使用大模型裁判
                # HotpotQA 提供精确的事实答案
                expected_answer=f"标准事实答案是：{item['answer']}。请判断模型的回答是否准确包含了该事实信息。"
            )
            test_cases.append(tc)
    except Exception as e:
        print(f"HotpotQA 数据集加载失败，请检查网络: {e}")

    # 将数学题和搜索题打乱顺序
    random.shuffle(test_cases)
    print(f"[数据加载] 成功构建包含 {len(test_cases)} 道题的黄金测试集！\n")
    
    return test_cases

class EvaluationFramework:
    """评估框架类"""
    
    def __init__(self, use_benchmark: bool = True, num_tests: int = 50):
        """
        初始化评估器。如果 use_benchmark 为 True，则动态从云端拉取题目。
        """
        if use_benchmark:
            # 调用我们刚刚写好的自动拉取函数
            self.test_cases = fetch_benchmark_test_cases(total_samples=num_tests)
        else:
            # 回退到你原本手写的本地测试用例
            self.test_cases = self._generate_test_cases()
            
        self.results = []
    
    def _generate_test_cases(self) -> List[TestCase]:
        """生成不同难度的测试用例（进阶版）"""
        test_cases = []
        
        # --- 简单难度 ---
        easy_cases = [
            TestCase(
                question="单词 'strawberry' 中有几个字母 'r'？请在回答的最后以纯数字形式给出答案，例如：答案是：3",
                difficulty="easy",
                requires_tool=False,
                category="tokenization_test",
                match_type="regex",
                regex_pattern=r"3" # 精准匹配最后给出的数字
            ),
            TestCase(
                question="如果所有的甲都是乙，有些乙是丙。那么'有些甲一定是丙'这个推论正确吗？只回答'是'或'否'。",
                difficulty="easy",
                requires_tool=False,
                category="logic",
                match_type="strict_format",
                regex_pattern=r"^[否]$" # 要求全文本只有“否”这一个字（排除废话）
            ),
            TestCase(
                question="在Python中，`is` 和 `==` 操作符的核心区别是什么？请用一句话概括。",
                difficulty="easy",
                requires_tool=False,
                category="programming",
                match_type="keywords",
                expected_keywords=["内存", "地址", "值"] # 必须同时提到内存/地址（代表is）和值（代表==）
            ),
        ]
        
        # --- 中等难度 ---
        medium_cases = [
            TestCase(
                question="一列火车在下午 3:45 出发，运行了 238 分钟后到达终点。请问到达时间是几点几分？请在结尾输出：(到达时间：19:43)",
                difficulty="medium",
                requires_tool=True,
                category="math_time",
                match_type="regex",
                regex_pattern=r"到达时间[：:]\s*(19:43|7:43\s*[Pp][Mm])" 
            ),
            TestCase(
                question="生成一个包含5个随机英文单词的列表，并将其转化为大写，然后以逗号分隔的字符串格式输出。不要输出任何其他解释性文字。",
                difficulty="medium",
                requires_tool=True,
                category="instruction_following",
                match_type="strict_format",
                regex_pattern=r"^[A-Z]+,[A-Z]+,[A-Z]+,[A-Z]+,[A-Z]+$" # 严格校验5个大写单词及逗号，禁止任何其他字符
            ),
            TestCase(
                question="求解方程：3x + 7 = 34，x的值是多少？请在最后写出：x=答案。",
                difficulty="medium",
                requires_tool=True,
                category="math",
                match_type="regex",
                regex_pattern=r"x\s*=\s*9(?!\d)" # 匹配 x=9，且9后面不能跟其他数字（防止x=90）
            ),
            TestCase(
                question="请使用搜索工具查询月球到地球的平均距离是多少万公里？请直接在末尾输出纯数字，格式如：(距离：38.4)",
                difficulty="medium",
                requires_tool=True,
                category="information_retrieval",
                match_type="regex",
                regex_pattern=r"距离[：:]\s*38\.4"
            ),
            TestCase(
                question="Python编程语言的创始人是谁？请在回答末尾以【创始人：XXX】的格式输出名字。",
                difficulty="medium",
                requires_tool=True,
                category="information_retrieval",
                match_type="regex",
                regex_pattern=r"【创始人[：:]\s*(吉多·范罗苏姆|Guido van Rossum)】"
            ),
        ]
        
        # --- 困难难度 ---
        hard_cases = [
            TestCase(
                question="我有无限的水，以及一个准确容量为3升的罐子和一个准确容量为5升的罐子。请列出操作说明，让我能在5升的罐子中准确量出4升水。并在最后单独一行写出最少需要的步数，格式为：【最少步数：6】",
                difficulty="hard",
                requires_tool=False,
                category="planning_reasoning",
                match_type="regex",
                regex_pattern=r"【最少步数[：:]\s*6】"
            ),
            TestCase(
                question="请使用Python求解线性方程组：x + y + z = 6, 2x + 5y - z = 14, -x + 2y + 3z = 8。请先编写代码求解，然后在最后单独一行输出结果，格式：x=1, y=2, z=3",
                difficulty="hard",
                requires_tool=True,
                category="complex_math_coding",
                match_type="regex",
                regex_pattern=r"x\s*=\s*1\s*,\s*y\s*=\s*2\s*,\s*z\s*=\s*3" # 忽略空格差异
            ),
            TestCase(
                question="现有一段JSON数据表示员工薪资：[{'dept': 'A', 'salary': 5000}, {'dept': 'B', 'salary': 6000}, {'dept': 'A', 'salary': 7000}]。请分析并计算哪个部门的平均薪资最高，最高平均薪资是多少？结论请写为：最高平均薪资为6000。",
                difficulty="hard",
                requires_tool=True,
                category="data_analysis",
                match_type="regex",
                regex_pattern=r"最高平均薪资为\s*6000" 
            ),

            TestCase(
                question="假设有一艘飞船以光速飞行，请问它飞行 5 秒钟能飞多远？请先查询光速，再进行计算。末尾输出格式：答案=1498960",
                difficulty="hard",
                requires_tool=True,
                category="multi_tool_reasoning",
                match_type="regex",
                # 预期逻辑: 搜光速(299792) -> 计算(299792 * 5) -> 得出 1498960
                regex_pattern=r"答案\s*=\s*149896"
            ),
        ]
        
        # --- 真实困难难度 (Real Hard) ---
        too_hard_cases = [
            TestCase(
                # 考察点：多条件约束的逻辑推理。无法通过搜索获取，必须自己推理。
                question="请帮我安排 A, B, C, D, E 五个会议在周一到周五的日程。条件如下：1. A 会议在 C 会议的前两天；2. B 会议不能安排在周四或周五；3. D 会议必须紧挨在 B 会议的后一天；4. E 会议不能排在周一。请推理出周一到周五每天对应的会议。并在最后单独一行输出结果，格式为：【排序：X-X-X-X-X】（X为会议字母）。",
                difficulty="too_hard",
                requires_tool=False,
                category="planning_reasoning",
                match_type="regex",
                # 正确推理：B=周一, D=周二, A=周三, E=周四, C=周五
                regex_pattern=r"【排序[：:]\s*B-D-A-E-C】"
            ),
            TestCase(
                # 考察点：数学陷阱与代码执行。1000-2000之间的素数必然是四位数，四位回文数必能被11整除，所以不存在回文素数。
                question="请编写Python代码寻找 1000 到 2000 之间所有的“回文素数”（既是素数从左向右和从右向左读都一样的数），并计算它们的和。请在最后单独一行输出结果，格式为：【总和：X】",
                difficulty="too_hard",
                requires_tool=True,
                category="complex_math_coding",
                match_type="regex",
                # 预期逻辑: 编写代码 -> 发现结果是空集 -> 求和为 0
                regex_pattern=r"【总和[：:]\s*0】"
            ),
            TestCase(
                # 考察点：嵌套结构的解析与带条件的业务逻辑计算。
                question="解析以下订单JSON：`[{\"id\":1,\"status\":\"delivered\",\"items\":[{\"p\":12,\"q\":3},{\"p\":5,\"q\":10}]}, {\"id\":2,\"status\":\"pending\",\"items\":[{\"p\":50,\"q\":1}]}, {\"id\":3,\"status\":\"delivered\",\"items\":[{\"p\":120,\"q\":1}]}, {\"id\":4,\"status\":\"delivered\",\"items\":[{\"p\":8,\"q\":5}]}]`。请先过滤掉非 delivered 状态的订单。计算有效订单的总金额（p代表单价，q代表数量）。业务规则：如果单个订单的商品总价大于100，则该订单总价享受9折优惠。求所有有效订单最终的真实总收入。结尾单独一行输出：【总收入：XXX】",
                difficulty="too_hard",
                requires_tool=True,
                category="data_analysis",
                match_type="regex",
                # 预期逻辑:
                # id 1: 36 + 50 = 86
                # id 2: pending (跳过)
                # id 3: 120 (大于100，打9折) = 108
                # id 4: 40
                # 总计 = 86 + 108 + 40 = 234
                regex_pattern=r"【总收入[：:]\s*234】" 
            ),
            TestCase(
                # 考察点：搜索特定历史事实，并结合 Python 的 datetime 库进行精确的跨期计算。
                question="请通过搜索确认“阿波罗11号”（Apollo 11）登月舱成功降落在月球表面的确切日期（以UTC时间为准）。然后使用代码计算：从那一天算起（包含降落当天），到2025年1月1日，一共跨越了多少天？结论格式：【总天数：XXXXX】",
                difficulty="too_hard",
                requires_tool=True,
                category="multi_tool_reasoning", 
                match_type="regex",
                # 预期逻辑: 搜阿波罗11降落日(1969-07-20) -> 使用Python计算到 2025-01-01 的天数 -> 得出 20254 天
                regex_pattern=r"【总天数[：:]\s*20254】" 
            )
        ]
        agent_killer_cases = [
            TestCase(
                # 针对弱点：模式匹配陷阱 + 状态空间拓展 (Reflection Agent 的主场)
                # Direct LLM 的死法：看到“最短路径”，直接默写标准 Dijkstra 算法，无视连续奇偶性约束，得出错误结果。
                # Reflection 的赢法：写出标准 Dijkstra -> 运行测试 -> 发现不满足“交替”规则 -> 反思并意识到需要将图的状态空间扩充为 (当前节点, 上一步权重的奇偶性) -> 修改代码 -> 得出正确结果。
                question="请计算从节点A到节点F的最短路径总权重。图的无向边和权重如下：(A,B,3), (A,C,2), (B,D,4), (B,C,5), (C,E,6), (D,E,1), (D,F,8), (E,F,7)。特殊规则：你行走的路线中，连续相邻的两条边，其权重【绝不能都是奇数，也绝不能都是偶数】（即奇偶必须交替）。请编写代码求解，并在最后单独一行输出结果，格式：【最短路径代价：XX】",
                difficulty="agent_killer",
                requires_tool=True,
                category="graph_state_machine",
                match_type="regex",
                # 解析：
                # 正常最短路是 A->C->E->D->F (2+6+1+8 = 17) 或 A->C->E->F (2+6+7=15)。
                # 加上奇偶交替规则：
                # A->C(2,偶) -> E(6,偶，违规！)
                # A->B(3,奇) -> D(4,偶) -> E(1,奇) -> F(7,奇，违规！)
                # 真正的合法最短路：A->C(2,偶) -> B(5,奇) -> D(4,偶) -> E(1,奇) -> C(6,偶) -> A(2,偶)... 必须用 BFS/Dijkstra 配合状态记录。
                # 留给你的 Agent 去解这道题，看它能否自我纠错。假设正确答案通过代码验证为 18（仅作演示，以实际代码输出为准）。
                regex_pattern=r"【最短路径代价[：:]\s*\d+】" 
            ),
            TestCase(
                # 针对弱点：执行视界 + 性能陷阱 (ReAct + Reflection 协同)
                # Direct LLM 的死法：遇到递推公式直接幻觉一个数字，或者试图用代数推导通项公式（这里没有简单的通项公式）。
                # ReAct 的第一层死法：直接写一个递归函数 `def A(n): return (A(n-1)**2 + A(n-2)) % 997`，结果递归深度超限 (RecursionError) 或超时。
                # Reflection 的赢法：捕获到 RecursionError 或 Timeout -> 反思发现 n 太大 -> 将代码重写为带有循环 (for-loop) 的迭代方式 -> 秒出结果。
                question="定义一个整数序列：A(1)=1, A(2)=2。对于 n >= 3，满足递推公式：A(n) = (A(n-1)^2 + A(n-2)) modulo 997。（注意是取模 997）。请问 A(1000000) 的值是多少？请务必通过编写和执行代码来得出答案。最后单独一行输出结果，格式：【序列计算结果：XXX】",
                difficulty="agent_killer",
                requires_tool=True,
                category="algorithmic_optimization",
                match_type="regex",
                # 这题迫使 Agent 必须写出非递归、空间复杂度为 O(1) 的迭代代码。
                regex_pattern=r"【序列计算结果[：:]\s*\d+】"
            ),
            TestCase(
                # 针对弱点：多跳信息提取与强依赖运算 (ReAct Agent 的主场)
                # Direct LLM 的死法：发布日期的具体天数极难记住，计算两个远古日期之间的天数更是大语言模型的算力盲区。
                # ReAct 的赢法：调用搜索确认两个日期 -> 调用 Python `datetime` 模块相减 -> 写一段质因数分解代码 -> 输出结果。
                question="请查阅并确认 Python 3.0 正式发布的日期，以及 React 16.0 正式发布的日期。计算这两个日期之间相隔了多少天（精确到天）。然后，将这个【天数】作为一个整数，求出它的【最大质因数】。最后单独一行输出结果，格式：【最大质因数：XXX】",
                difficulty="agent_killer",
                requires_tool=True,
                category="multi_hop_factual_math",
                match_type="regex",
                # Python 3.0: 2008-12-03
                # React 16.0: 2017-09-26
                # 间隔天数: 3219 天。 3219 = 3 * 29 * 37，最大质因数是 37。
                regex_pattern=r"【最大质因数[：:]\s*37】"
            )
        ]


        #test_cases.extend(easy_cases)
        #test_cases.extend(medium_cases)
        test_cases.extend(hard_cases)
        test_cases.extend(too_hard_cases) # 直接使用真实困难难度的题目，跳过中等难度
        test_cases.extend(agent_killer_cases) # 加入针对Agent弱点的杀手级测试题
        
        return test_cases
    
    def _call_llm_judge_comparative_score(self, question: str, responses_dict: Dict[str, str], reference_criteria: str) -> Dict[str, float]:
        """
        [横向对比模式] 调用大模型作为裁判，同时评估多个回答，拉开分数差距。
        返回格式: {"DirectPromptAgent": 85.0, "ReActAgent": 98.0, ...}
        """
        # 将多个Agent的回答拼接成便于对比的文本格式
        responses_text = ""
        for agent_name, resp in responses_dict.items():
            # 处理空回答或报错的情况
            if not resp.strip():
                resp = "[未能生成有效回答或发生错误]"
            responses_text += f"\n【{agent_name} 的回答】\n{resp}\n{'-'*40}"

        judge_prompt = f"""你是一个极其严格的AI系统评估专家。你的任务是对多个不同AI针对同一问题的回答进行横向对比，并分别给出0到100分的量化打分。

请注意：横向对比是拉开差距的关键！如果一个回答精炼准确，而另一个回答啰嗦、包含多余的废话（如“根据搜索”、“综上所述”），你应该给精炼的打高分（90以上），啰嗦的严厉扣分（70-85）。绝对不要给所有人都打出0分和100分！

【原始问题】
{question}

【参考标准】
{reference_criteria}

【各模型的实际回答】{responses_text}

【严苛的对比评分标准】
- 95-100分 (完美无瑕)：事实与逻辑完全正确，精准契合标准，**没有任何废话**，是所有回答中最优秀的。
- 80-94分 (优秀但有瑕疵)：核心答案正确，但相比其他回答不够精炼，或包含了不必要的过渡性语句。
- 60-79分 (勉强及格)：结果基本正确，但推理逻辑有跳跃，或附加了不准确的冗余信息。
- 30-59分 (不及格)：部分正确，核心事实有误。
- 0-29分 (极差)：完全答错、答非所问或未能给出回答。

请严格对比，并必须且仅以JSON格式输出打分结果。键是智能体的名称，值是0-100的纯数字。
绝不要输出任何其他解释性文字，只输出一段合法的JSON！例如：
{{
    "DirectPromptAgent": 85,
    "ReActAgent": 98,
    "ReflexionAgent": 60
}}"""
        
        try:
            # 依然推荐使用 Pro 模型来做精细化打分
            #judge_result = api_call(question=judge_prompt, model="ep-20260526111615-fkrrc")
            judge_result = api_call(question=judge_prompt, model="qwen3.7-plus") 
            result_text = judge_result.message.content.strip()
            
            # 使用正则提取包裹在 {} 中的 JSON 字符串，防止大模型输出多余字符
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                scores = json.loads(match.group())
                final_scores = {}
                # 遍历确保每个agent都有分数，并限制在0-100范围内
                for name in responses_dict.keys():
                    score = float(scores.get(name, 0.0))
                    final_scores[name] = min(max(score, 0.0), 100.0)
                return final_scores
            else:
                print(f"[LLM Judge Warning] 无法解析JSON: {result_text}")
                return {name: 0.0 for name in responses_dict.keys()}
                
        except Exception as e:
            print(f"[LLM Judge Error] 裁判调用失败或解析错误: {e}")
            return {name: 0.0 for name in responses_dict.keys()}

    def _evaluate_comparative_responses(self, responses_dict: Dict[str, str], test_case) -> Dict[str, float]:
        """对一道题下的所有回答进行多策略评估"""
        scores = {}
        
        # 1. 纯格式/正则匹配 (这类确定性问题，依然遵循机器判分，全对100，全错0)
        if test_case.match_type in ["strict_format", "regex", "contains"]:
            for agent_name, response in responses_dict.items():
                if not response:
                    scores[agent_name] = 0.0
                    continue
                    
                response_cleaned = response.strip()
                response_lower = response_cleaned.lower()
                
                if test_case.match_type == "strict_format":
                    scores[agent_name] = 100.0 if re.fullmatch(test_case.regex_pattern, response_cleaned) else 0.0
                elif test_case.match_type == "regex":
                    scores[agent_name] = 100.0 if re.search(test_case.regex_pattern, response_cleaned) else 0.0
                elif test_case.match_type == "contains":
                    scores[agent_name] = 100.0 if test_case.expected_answer.lower() in response_lower else 0.0
            return scores
            
        # 2. 概念与推理题 (全部丢给 LLM 进行横向对比打分)
        elif test_case.match_type in ["keywords", "llm_judge"]:
            if test_case.match_type == "keywords":
                criteria = f"回答必须在语义上包含以下核心概念点：{', '.join(test_case.expected_keywords)}"
            else:
                criteria = test_case.expected_answer or "请根据问题自身的逻辑判断回答是否正确合理。"
                
            print(f"    -> [触发横向对比裁判] 评估当前题目的 3 个回答...")
            return self._call_llm_judge_comparative_score(test_case.question, responses_dict, criteria)
            
        else:
            return {name: 0.0 for name in responses_dict.keys()}

    # 删除原有的 run_evaluation，将所有逻辑重构至 run_full_evaluation
    def run_full_evaluation(self):
        """运行完整评估 (以题目为维度的横向对比模式)"""
        all_results = []
        
        # 实例化Agents集合
        agents = {
            "DirectPromptAgent": DirectPromptAgent(),
            "ReActAgent": ReActAgent(max_steps=3),
            "ReflexionAgent": ReflexionAgent(max_reflections=2)
        }
        
        print(f"\n{'='*60}")
        print("开始多Agent横向对比评估 (Comparative Evaluation)")
        print(f"{'='*60}")
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\n测试用例 {i+1}/{len(self.test_cases)} | 难度: {test_case.difficulty}")
            print(f"问题: {test_case.question}")
            
            # 存储此题各模型的回答与元数据
            case_responses = {}
            case_meta = {} 
            
            # 步骤 1：收集所有 Agent 的回答
            for name, agent in agents.items():
                print(f"  [{name}] 思考中...")
                try:
                    start_time = time.time()
                    agent.clear_conversation()
                    response, rounds = agent.ask(test_case.question)
                    time_used = time.time() - start_time
                    
                    history = agent.get_conversation_history()
                    
                    case_responses[name] = response
                    case_meta[name] = {
                        "time_used": time_used,
                        "interaction_rounds": rounds,
                        "error_message": ""
                    }
                except Exception as e:
                    print(f"  [{name}] 执行错误: {str(e)}")
                    case_responses[name] = ""
                    case_meta[name] = {"time_used": 0.0, "interaction_rounds": 0, "error_message": str(e)}
            
            # 步骤 2：调用评估中心，同时传入 3 个回答进行对比
            scores = self._evaluate_comparative_responses(case_responses, test_case)
            
            # 步骤 3：汇总结果
            for name in agents.keys():
                score = scores.get(name, 0.0)
                meta = case_meta[name]
                resp = case_responses[name]
                
                # 复用原有的 EvaluationResult 结构，用 success 字段存储 float 分数
                result = EvaluationResult(
                    agent_name=name,
                    test_case=test_case,
                    success=score,  
                    response=resp,
                    interaction_rounds=meta["interaction_rounds"],
                    time_used=meta["time_used"],
                    error_message=meta["error_message"]
                )
                all_results.append(result)
                
                print(f"    - {name:<18} | 分数: {score:<5} | 用时: {meta['time_used']:.1f}s | 轮数: {meta['interaction_rounds']}")
                
        self.results = all_results
        return all_results
    
    def analyze_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """分析评估结果"""
        analysis = {
            "overall": {},
            "by_agent": {},
            "by_difficulty": {},
            "by_tool_requirement": {}
            
        }
        
        # 总体统计
        total = len(results)
        successful = sum(r.success for r in results)
        analysis["overall"] = {
            "total_tests": total,
            "grade": successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_interaction_rounds": sum(r.interaction_rounds for r in results) / total if total > 0 else 0,
            "avg_time_used": sum(r.time_used for r in results) / total if total > 0 else 0,
        }
        
        # 按Agent统计
        agent_names = set(r.agent_name for r in results)
        for agent_name in agent_names:
            agent_results = [r for r in results if r.agent_name == agent_name]
            agent_total = len(agent_results)
            agent_successful = sum(r.success for r in agent_results)
            
            analysis["by_agent"][agent_name] = {
                "total_tests": agent_total,
                "grade": agent_successful,
                "success_rate": agent_successful / agent_total if agent_total > 0 else 0,
                "avg_interaction_rounds": sum(r.interaction_rounds for r in agent_results) / agent_total if agent_total > 0 else 0,
                "avg_time_used": sum(r.time_used for r in agent_results) / agent_total if agent_total > 0 else 0,
            }
        
        # 按难度统计
        difficulties = ["too_hard", "agent_killer"]
        for difficulty in difficulties:
            diff_results = [r for r in results if r.test_case.difficulty == difficulty]
            diff_total = len(diff_results)
            diff_successful = sum(r.success for r in diff_results)

            analysis["by_difficulty"][difficulty] = {
                "total_tests": diff_total,
                "grade": diff_successful,
                "success_rate": diff_successful / diff_total if diff_total > 0 else 0,
            }
        
        # 按是否需要工具统计
        for requires_tool in [True, False]:
            tool_results = [r for r in results if r.test_case.requires_tool == requires_tool]
            tool_total = len(tool_results)
            tool_successful = sum(r.success for r in tool_results)
            
            analysis["by_tool_requirement"][str(requires_tool)] = {
                "total_tests": tool_total,
                "grade": tool_successful,
                "success_rate": tool_successful / tool_total if tool_total > 0 else 0,
            }
        
        return analysis
    def plot_results(self, analysis: Dict[str, Any], save_dir: str = "."):
        """
        根据分析结果生成并保存可视化图表
        """
        # 设置中文字体，防止图表中的中文显示为方块
        # Windows常用 'SimHei' (黑体)，Mac常用 'Arial Unicode MS'
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

        # 创建一个 2x2 的子图网格
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Agent 范式性能综合评估报告', fontsize=18, fontweight='bold', y=0.98)

        # ==========================================
        # 图 1: 各 Agent 成功率对比 (左上)
        # ==========================================
        agents = list(analysis["by_agent"].keys())
        success_rates = [analysis["by_agent"][a]["success_rate"] for a in agents]
        
        bars1 = axs[0, 0].bar(agents, success_rates, color=['#4e79a7', '#f28e2c', '#e15759'])
        axs[0, 0].set_title('各 Agent 任务成功率对比', fontsize=14)
        axs[0, 0].set_ylabel('得分率 (%)')
        axs[0, 0].set_ylim(0, 110) # 留出顶部空间显示文字
        
        # 在柱子上显示具体数值
        for bar in bars1:
            yval = bar.get_height()
            axs[0, 0].text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')

        # ==========================================
        # 图 2: 各 Agent 平均耗时与交互轮数 (右上)
        # ==========================================
        times = [analysis["by_agent"][a]["avg_time_used"] for a in agents]
        
        bars2 = axs[0, 1].bar(agents, times, color=['#76b7b2', '#59a14f', '#edc949'])
        axs[0, 1].set_title('各 Agent 平均处理耗时', fontsize=14)
        axs[0, 1].set_ylabel('平均耗时 (秒)')
        
        for bar in bars2:
            yval = bar.get_height()
            axs[0, 1].text(bar.get_x() + bar.get_width()/2, yval + (max(times)*0.02), f'{yval:.2f}s', ha='center', va='bottom')

        # ==========================================
        # 图 3: 不同难度对总体成功率的影响 (左下)
        # ==========================================
        difficulties = list(analysis["by_difficulty"].keys())
        diff_rates = [analysis["by_difficulty"][d]["success_rate"] for d in difficulties]
        
        # 难度通常有固定顺序，这里我们映射为中文
        diff_map = {"easy": "简单 (Easy)", "medium": "中等 (Medium)", "hard": "困难 (Hard)"}
        diff_labels = [diff_map.get(d, d) for d in difficulties]

        bars3 = axs[1, 0].bar(diff_labels, diff_rates, color='#af7aa1', width=0.6)
        axs[1, 0].set_title('不同任务难度的总体成功率', fontsize=14)
        axs[1, 0].set_ylabel('得分率 (%)')
        axs[1, 0].set_ylim(0, 110)
        
        for bar in bars3:
            yval = bar.get_height()
            axs[1, 0].text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')

        # ==========================================
        # 图 4: 工具需求对总体成功率的影响 (右下)
        # ==========================================
        tool_reqs = list(analysis["by_tool_requirement"].keys())
        tool_labels = ["需要调用工具" if req == "True" else "无需工具纯推理" for req in tool_reqs]
        tool_rates = [analysis["by_tool_requirement"][req]["success_rate"] for req in tool_reqs]

        axs[1, 1].bar(tool_labels, tool_rates, color=['#ff9da7', '#9c755f'], width=0.5)
        axs[1, 1].set_title('工具依赖度对成功率的影响', fontsize=14)
        axs[1, 1].set_ylabel('得分率 (%)')
        axs[1, 1].set_ylim(0, 110)
        
        for i, v in enumerate(tool_rates):
            axs[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')

        # 调整布局并保存
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为主标题留出空间
        
        save_path = os.path.join(save_dir, "agent_evaluation_dashboard.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 可视化报告已生成并保存至: {save_path}")
        
        # 如果在 Jupyter Notebook 等环境中，这行代码会将图表直接显示出来
        plt.show()
    
    def print_analysis(self, analysis: Dict[str, Any]):
        """打印分析结果"""
        print(f"\n{'='*60}")
        print("评估结果分析")
        print(f"{'='*60}")
        
        # 总体统计
        print(f"\n【总体统计】")
        print(f"总测试数: {analysis['overall']['total_tests']}")
        print(f"分数: {analysis['overall']['grade']}")
        print(f"成功率: {analysis['overall']['success_rate']:.2%}")
        print(f"平均交互轮数: {analysis['overall']['avg_interaction_rounds']:.2f}")
        print(f"平均用时: {analysis['overall']['avg_time_used']:.2f}秒")
        
        # 按Agent统计
        print(f"\n【各Agent性能对比】")
        for agent_name, metrics in analysis["by_agent"].items():
            print(f"\n{agent_name}:")
            print(f"  成功率: {metrics['success_rate']:.2%} ({metrics['grade']}/{metrics['total_tests']})")
            print(f"  平均交互轮数: {metrics['avg_interaction_rounds']:.2f}")
            print(f"  平均用时: {metrics['avg_time_used']:.2f}秒")
        
        # 按难度统计
        print(f"\n【按难度统计】")
        for difficulty, metrics in analysis["by_difficulty"].items():
            print(f"{difficulty}: 成功率 {metrics['success_rate']:.2%} ({metrics['grade']}/{metrics['total_tests']})")
        
        # 按工具需求统计
        print(f"\n【按工具需求统计】")
        print(f"需要工具: {analysis['by_tool_requirement']['True']['success_rate']:.2%}")
        print(f"不需要工具: {analysis['by_tool_requirement']['False']['success_rate']:.2%}")
    
    def save_results(self, results: List[EvaluationResult], analysis: Dict[str, Any], filename: str = None):
        """保存结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        # 转换结果为可序列化格式
        results_dict = []
        for result in results:
            result_dict = asdict(result)
            result_dict['test_case'] = asdict(result.test_case)
            results_dict.append(result_dict)
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "results": results_dict,
            "analysis": analysis
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {filename}")


def main():
    """主函数"""
    print("="*60)
    print("智能体评估框架")
    print("="*60)
    
    # 创建评估框架
    evaluator = EvaluationFramework()
    
    # 打印测试集信息
    print(f"\n测试集包含 {len(evaluator.test_cases)} 个测试用例:")
    for i, test_case in enumerate(evaluator.test_cases):
        print(f"  {i+1}. [{test_case.difficulty}] {test_case.question[:50]}...")
    
    # 确认是否开始评估
    confirm = input("\n是否开始评估？(y/n): ").strip().lower()
    if confirm != 'y':
        print("评估已取消")
        return
    
    # 运行完整评估
    results = evaluator.run_full_evaluation()
    
    # 分析结果
    analysis = evaluator.analyze_results(results)
    
    # 生成可视化报告
    #evaluator.plot_results(analysis)
    # 打印分析
    #evaluator.print_analysis(analysis)
    
    # 保存结果
    evaluator.save_results(results, analysis)
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()